from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader

from utils.config import load_yaml
from utils.gwpca import apply_gwpca_with_state, fit_gwpca
from utils.dataset import (
    build_fewshot_splits,
    load_hsi_from_paths,
    load_whu_hsi,
    resolve_dataset_name,
)
from utils.normalize import apply_zscore, fit_zscore_from_tensor
from models.hybridcnn import HybridSN


def _get(cfg, *keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_model(cfg):
    c_in = _get(cfg, "data", "C_in", default=64)
    num_classes = _get(cfg, "data", "num_classes", default=9)
    dropout = _get(cfg, "model", "dropout", default=0.3)
    fc_dim = _get(cfg, "model", "fc_dim", default=256)
    return HybridSN(c_in=c_in, num_classes=num_classes, fc_dim=fc_dim, dropout=dropout)


def _run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    if is_train:
        model.train()
    else:
        model.eval()
    for x_raw, y in loader:
        x = x_raw.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y = (y - 1).long()
        if is_train:
            optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * y.size(0)
        preds = logits.argmax(dim=1)
        total_correct += int((preds == y).sum().item())
        total_count += y.size(0)
    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hybridcnn.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    torch.manual_seed(_get(cfg, "train", "seed", default=42))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, please enable GPU or install CUDA build.")
    device = torch.device("cuda")
    model = build_model(cfg).to(device)

    b = _get(cfg, "train", "batch_size", default=4)
    num_workers = _get(cfg, "train", "num_workers", default=0)
    pin_memory = _get(cfg, "train", "pin_memory", default=True)
    prefetch_factor = _get(cfg, "train", "prefetch_factor", default=2)
    persistent_workers = _get(cfg, "train", "persistent_workers", default=False)
    c_in = _get(cfg, "data", "C_in", default=64)
    patch_size = _get(cfg, "data", "patch_size", default=9)
    num_classes = _get(cfg, "data", "num_classes", default=9)
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
        train_set, val_set, test_set = build_fewshot_splits(
            cube_feat,
            data.gt,
            patch_size=patch_size,
            train_patches=train_patches,
            val_patches=val_patches,
            seed=split_seed,
        )
        loader_args = {
            "batch_size": b,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers if num_workers > 0 else False,
        }
        if num_workers > 0:
            loader_args["prefetch_factor"] = prefetch_factor
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, **loader_args)
        test_loader = DataLoader(test_set, shuffle=False, **loader_args)
        print("train/val/test", len(train_set), len(val_set), len(test_set))

        epochs = _get(cfg, "train", "epochs", default=1)
        lr = _get(cfg, "train", "lr", default=0.001)
        wd = _get(cfg, "train", "wd", default=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = _run_epoch(
                model, train_loader, device, optimizer=optimizer
            )
            val_loss, val_acc = _run_epoch(model, val_loader, device, optimizer=None)
            print(
                f"epoch {epoch}/{epochs} "
                f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
                f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}"
            )
        test_loss, test_acc = _run_epoch(model, test_loader, device, optimizer=None)
        print(f"test_loss {test_loss:.4f} test_acc {test_acc:.4f}")
    else:
        x = torch.randn(b, c_in, patch_size, patch_size, device=device)
        y = torch.randint(0, num_classes, (b,), device=device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        print("x", tuple(x.shape))
        print("logits", tuple(logits.shape))
        print("loss", float(loss.detach().cpu()))


if __name__ == "__main__":
    main()
