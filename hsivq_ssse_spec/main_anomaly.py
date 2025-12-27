from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader

from utils.config import load_yaml
from utils.cube_partition import make_cubes
from utils.gwpca import apply_gwpca_with_state, fit_gwpca
from utils.dataset import (
    build_fewshot_splits,
    load_hsi_from_paths,
    load_whu_hsi,
    resolve_dataset_name,
)
from utils.normalize import apply_zscore, fit_zscore_from_tensor
from models.hsi_vq_ssse import HSIVQSSSEModel


def _get(cfg, *keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_model(cfg):
    return HSIVQSSSEModel(
        patch_size=_get(cfg, "data", "patch_size", default=9),
        cube_size=_get(cfg, "data", "cube_size", default=3),
        T=_get(cfg, "data", "T", default=8),
        L=_get(cfg, "data", "L", default=8),
        Ks=_get(cfg, "vq", "Ks", default=512),
        Kx=_get(cfg, "vq", "Kx", default=512),
        D=_get(cfg, "encoder", "D", default=384),
        D_vq=_get(cfg, "vq", "D_vq", default=256),
        depth_spa=_get(cfg, "encoder", "depth_spa", default=6),
        depth_spe=_get(cfg, "encoder", "depth_spe", default=6),
        depth_fuse=_get(cfg, "encoder", "depth_fuse", default=2),
        heads=_get(cfg, "encoder", "heads", default=8),
        mlp_ratio=_get(cfg, "encoder", "mlp_ratio", default=4.0),
        dropout=_get(cfg, "encoder", "dropout", default=0.1),
        attn_dropout=_get(cfg, "encoder", "attn_dropout", default=0.1),
        mr_spa=_get(cfg, "mask", "mr_spa", default=0.5),
        mr_spe=_get(cfg, "mask", "mr_spe", default=0.5),
        num_classes=_get(cfg, "data", "num_classes", default=9),
        vq_decay=_get(cfg, "vq", "vq_decay", default=0.99),
        vq_beta=_get(cfg, "vq", "vq_beta", default=0.25),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/anomaly_hsivq_ssse.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    torch.manual_seed(_get(cfg, "train", "seed", default=42))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, please enable GPU or install CUDA build.")
    device = torch.device("cuda")
    model = build_model(cfg).to(device)
    model.train()

    b = _get(cfg, "train", "batch_size", default=4)
    num_workers = _get(cfg, "train", "num_workers", default=0)
    pin_memory = _get(cfg, "train", "pin_memory", default=True)
    prefetch_factor = _get(cfg, "train", "prefetch_factor", default=2)
    persistent_workers = _get(cfg, "train", "persistent_workers", default=False)
    c_in = _get(cfg, "data", "C_in", default=64)
    patch_size = _get(cfg, "data", "patch_size", default=9)
    pca_group = _get(cfg, "data", "pca_group", default=8)
    pca_nc = _get(cfg, "data", "pca_nc_per_group", default=8)
    pca_max_samples = _get(cfg, "data", "pca_max_samples", default=None)
    use_zscore = _get(cfg, "data", "use_zscore", default=True)
    zscore_max_samples = _get(cfg, "data", "zscore_max_samples", default=None)
    train_patches = _get(cfg, "data", "train_patches", default=20)
    val_patches = _get(cfg, "data", "val_patches", default=20)
    split_seed = _get(cfg, "data", "split_seed", default=42)

    data_path = _get(cfg, "data", "data_path", default=None)
    gt_path = _get(cfg, "data", "gt_path", default=None)
    data_key = _get(cfg, "data", "data_key", default=None)
    gt_key = _get(cfg, "data", "gt_key", default=None)
    dataset_name = _get(cfg, "data", "dataset_name", default=None) or resolve_dataset_name()
    if data_path is not None and gt_path is not None:
        data = load_hsi_from_paths(data_path, gt_path, data_key=data_key, gt_key=gt_key)
        dataset_name = dataset_name or "custom"
    elif dataset_name is not None:
        data = load_whu_hsi(dataset_name)
    else:
        data = None

    if data is not None and data.gt is not None:
        gwpca_state = fit_gwpca(
            data.cube,
            group=pca_group,
            nc_per_group=pca_nc,
            whiten=True,
            max_samples=pca_max_samples,
        )
        cube_feat = apply_gwpca_with_state(data.cube, gwpca_state)
        cube_feat = torch.as_tensor(cube_feat)
        if use_zscore:
            z_mean, z_std = fit_zscore_from_tensor(
                cube_feat, max_samples=zscore_max_samples
            )
            cube_feat = apply_zscore(cube_feat, z_mean, z_std)
        cube_feat = cube_feat.squeeze(0).permute(1, 2, 0).cpu().numpy()
        _, _, test_set = build_fewshot_splits(
            cube_feat,
            data.gt,
            patch_size=patch_size,
            train_patches=train_patches,
            val_patches=val_patches,
            seed=split_seed,
        )
        loader_args = {
            "batch_size": b,
            "shuffle": True,
            "drop_last": True,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers if num_workers > 0 else False,
        }
        if num_workers > 0:
            loader_args["prefetch_factor"] = prefetch_factor
        loader = DataLoader(test_set, **loader_args)
        x, _ = next(iter(loader))
        x = x.to(device, non_blocking=True)
        print("test", len(test_set))
    else:
        x = torch.randn(b, c_in, patch_size, patch_size, device=device)

    cubes, _ = make_cubes(
        x,
        patch_size=patch_size,
        cube_size=_get(cfg, "data", "cube_size", default=3),
        T=_get(cfg, "data", "T", default=8),
    )
    out = model.forward_anomaly(
        x,
        score_type=_get(cfg, "anomaly", "score_type", default="mean"),
        topk_ratio=_get(cfg, "anomaly", "topk_ratio", default=0.1),
    )
    loss = out["anomaly_score"].mean()
    loss.backward()

    print("cubes", tuple(cubes.shape))
    print("tok", tuple(out["tok"].shape))
    print("nll_map", tuple(out["nll_map"].shape))
    print("anomaly_map_spatial", tuple(out["anomaly_map_spatial"].shape))
    print("anomaly_score", tuple(out["anomaly_score"].shape))


if __name__ == "__main__":
    main()
