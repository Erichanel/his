from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


def hsimae_consistent_mask(
    B: int,
    P: int = 9,
    T: int = 8,
    mr_spa: float = 0.5,
    mr_spe: float = 0.5,
    device: torch.device | None = None,
    seed: int | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = device or torch.device("cpu")
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    mp = int(math.ceil(mr_spa * P))
    mt = int(math.ceil(mr_spe * T))
    mask = torch.zeros((B, P, T), dtype=torch.bool, device=device)

    for b in range(B):
        p_idx = torch.randperm(P, generator=generator)[:mp].to(device)
        t_idx = torch.randperm(T, generator=generator)[:mt].to(device)
        mask[b, p_idx, :] = True
        mask[b, :, t_idx] = True

    mask_ratio = mask.float().mean().item()
    stats = {
        "mask_ratio": float(mask_ratio),
        "mr_spa": float(mr_spa),
        "mr_spe": float(mr_spe),
    }
    return mask, stats
