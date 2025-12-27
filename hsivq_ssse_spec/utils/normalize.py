from __future__ import annotations

import torch

from .gwpca import apply_gwpca_with_state


def fit_zscore_from_tensor(
    x_pca: torch.Tensor,
    max_samples: int | None = None,
    seed: int = 42,
):
    if not torch.is_tensor(x_pca):
        x_pca = torch.as_tensor(x_pca)
    if x_pca.ndim == 3:
        x_pca = x_pca.unsqueeze(0)
    if x_pca.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W] or [C, H, W], got {tuple(x_pca.shape)}")
    b, c, h, w = x_pca.shape
    flat = x_pca.reshape(b, c, -1).reshape(c, -1)
    if max_samples is not None and flat.shape[1] > max_samples:
        gen = torch.Generator()
        gen.manual_seed(seed)
        idx = torch.randperm(flat.shape[1], generator=gen)[:max_samples]
        flat = flat[:, idx]
    mean = flat.mean(dim=1)
    std = flat.std(dim=1)
    std = torch.clamp(std, min=1e-6)
    return mean, std


def fit_zscore_from_pca(
    cube,
    gwpca_state,
    max_samples: int | None = None,
    seed: int = 42,
):
    x_pca = apply_gwpca_with_state(cube, gwpca_state)
    return fit_zscore_from_tensor(x_pca, max_samples=max_samples, seed=seed)


def apply_zscore(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    return (x - mean) / std
