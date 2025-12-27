from __future__ import annotations

import argparse
import time
import torch
from torch.utils.data import DataLoader

from utils.config import load_yaml
from utils.cube_partition import make_cubes
from utils.gwpca import apply_gwpca_with_state, fit_gwpca
from utils.dataset import (
    build_pretrain_dataset,
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
    parser.add_argument("--config", default="configs/pretrain_hsivq_ssse.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    torch.manual_seed(_get(cfg, "train", "seed", default=42))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, please enable GPU or install CUDA build.")
    device = torch.device("cuda")
    model = build_model(cfg).to(device)
    tokenizer_ckpt = _get(cfg, "train", "tokenizer_ckpt", default=None)
    freeze_tokenizer = _get(cfg, "train", "freeze_tokenizer", default=False)
    if tokenizer_ckpt:
        state = torch.load(tokenizer_ckpt, map_location="cpu")
        if isinstance(state, dict) and "tokenizer" in state:
            state = state["tokenizer"]
        model.tokenizer.load_state_dict(state, strict=False)
        if freeze_tokenizer:
            for p in model.tokenizer.parameters():
                p.requires_grad = False
    model.train()

    b = _get(cfg, "train", "batch_size", default=4)
    num_workers = _get(cfg, "train", "num_workers", default=0)
    pin_memory = _get(cfg, "train", "pin_memory", default=True)
    prefetch_factor = _get(cfg, "train", "prefetch_factor", default=2)
    persistent_workers = _get(cfg, "train", "persistent_workers", default=False)
    c_in = _get(cfg, "data", "C_in", default=64)
    patch_size = _get(cfg, "data", "patch_size", default=9)
    pretrain_stride = _get(cfg, "data", "pretrain_stride", default=1)
    pca_group = _get(cfg, "data", "pca_group", default=8)
    pca_nc = _get(cfg, "data", "pca_nc_per_group", default=8)
    pca_max_samples = _get(cfg, "data", "pca_max_samples", default=None)
    use_zscore = _get(cfg, "data", "use_zscore", default=True)
    zscore_max_samples = _get(cfg, "data", "zscore_max_samples", default=None)

    data_path = _get(cfg, "data", "data_path", default=None)
    gt_path = _get(cfg, "data", "gt_path", default=None)
    data_key = _get(cfg, "data", "data_key", default=None)
    gt_key = _get(cfg, "data", "gt_key", default=None)
    dataset_name = _get(cfg, "data", "dataset_name", default=None) or resolve_dataset_name()
    if data_path is not None:
        data = load_hsi_from_paths(data_path, gt_path, data_key=data_key, gt_key=gt_key)
        dataset_name = dataset_name or "custom"
    elif dataset_name is not None:
        data = load_whu_hsi(dataset_name)
    else:
        data = None

    if data is not None:
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
        dataset = build_pretrain_dataset(
            cube_feat,
            patch_size=patch_size,
            stride=pretrain_stride,
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
        loader = DataLoader(dataset, **loader_args)
    else:
        loader = None

    epochs = _get(cfg, "train", "epochs", default=1)
    lr = _get(cfg, "train", "lr", default=0.0005)
    wd = _get(cfg, "train", "wd", default=0.0)
    steps_per_epoch = _get(cfg, "train", "steps_per_epoch", default=None)
    log_interval = _get(cfg, "train", "log_interval", default=200)
    usage_interval = _get(cfg, "train", "usage_interval", default=200)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    if loader is None:
        x = torch.randn(b, c_in, patch_size, patch_size, device=device)
        cubes, _ = make_cubes(
            x,
            patch_size=patch_size,
            cube_size=_get(cfg, "data", "cube_size", default=3),
            T=_get(cfg, "data", "T", default=8),
        )
        out = model.forward_pretrain(x)
        loss = out["loss_total"]
        loss.backward()
        print("cubes", tuple(cubes.shape))
        print("tok", tuple(out["tok"].shape))
        print("logits_k", tuple(out["logits_k"].shape))
        print("nll_map", tuple(out["nll_map"].shape))
        print("loss_total", float(loss.detach().cpu()))
        return

    printed_shapes = False
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_mtp = 0.0
        total_vq = 0.0
        total_count = 0
        start_time = time.time()
        max_steps = steps_per_epoch or len(loader)
        for step, x_raw in enumerate(loader, start=1):
            x = x_raw.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model.forward_pretrain(x)
            loss = out["loss_total"]
            loss.backward()
            optimizer.step()

            bs = x.shape[0]
            total_loss += float(loss.detach().cpu()) * bs
            total_mtp += float(out["loss_mtp"].detach().cpu()) * bs
            total_vq += float(out["vq_loss"].detach().cpu()) * bs
            total_count += bs

            if not printed_shapes:
                cubes, _ = make_cubes(
                    x,
                    patch_size=patch_size,
                    cube_size=_get(cfg, "data", "cube_size", default=3),
                    T=_get(cfg, "data", "T", default=8),
                )
                print("cubes", tuple(cubes.shape))
                print("tok", tuple(out["tok"].shape))
                print("logits_k", tuple(out["logits_k"].shape))
                print("nll_map", tuple(out["nll_map"].shape))
                printed_shapes = True

            if steps_per_epoch is not None and step >= steps_per_epoch:
                break
            if usage_interval and (step % usage_interval == 0):
                k_idx = out["tok_out"]["k_idx"]
                m_idx = out["tok_out"]["m_idx"]
                k_unique = torch.unique(k_idx).numel()
                m_unique = torch.unique(m_idx).numel()
                ks = _get(cfg, "vq", "Ks", default=512)
                kx = _get(cfg, "vq", "Kx", default=512)
                perplex_s = float(out["tok_out"]["perplexity_s"].detach().cpu())
                perplex_x = float(out["tok_out"]["perplexity_x"].detach().cpu())
                print(
                    f"codebook usage k {k_unique}/{ks} m {m_unique}/{kx} "
                    f"perplex_s {perplex_s:.2f} perplex_x {perplex_x:.2f}"
                )
            if log_interval and (step % log_interval == 0):
                avg_loss = total_loss / max(total_count, 1)
                avg_mtp = total_mtp / max(total_count, 1)
                avg_vq = total_vq / max(total_count, 1)
                elapsed = time.time() - start_time
                print(
                    f"epoch {epoch}/{epochs} step {step}/{max_steps} "
                    f"loss_total {avg_loss:.4f} loss_mtp {avg_mtp:.4f} "
                    f"vq_loss {avg_vq:.4f} time {elapsed:.1f}s"
                )

        avg_loss = total_loss / max(total_count, 1)
        avg_mtp = total_mtp / max(total_count, 1)
        avg_vq = total_vq / max(total_count, 1)
        print(
            f"epoch {epoch}/{epochs} "
            f"loss_total {avg_loss:.4f} loss_mtp {avg_mtp:.4f} vq_loss {avg_vq:.4f}"
        )


if __name__ == "__main__":
    main()
