from __future__ import annotations

from typing import Dict, Tuple

import torch


def make_cubes(
    x: torch.Tensor,
    patch_size: int = 9,
    cube_size: int = 3,
    T: int = 8,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    if x.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")
    b, c, h, w = x.shape
    assert h == patch_size and w == patch_size, "Patch size mismatch"
    assert h % cube_size == 0 and w % cube_size == 0, "Non-overlap requires divisible"
    assert c % T == 0, "C must be divisible by T"
    l = c // T
    assert l * T == c, "C != T * L"

    x = x.reshape(b, T, l, h, w)
    x = x.reshape(b * T, l, h, w)
    unfold = torch.nn.Unfold(kernel_size=cube_size, stride=cube_size)
    patches = unfold(x)  # [B*T, L*cube_size*cube_size, P]
    p = patches.shape[-1]
    f = l * cube_size * cube_size
    patches = patches.transpose(1, 2)  # [B*T, P, F]
    patches = patches.reshape(b, T, p, f).permute(0, 2, 1, 3).contiguous()
    assert patches.shape == (b, p, T, f)

    meta = {
        "Hs": h // cube_size,
        "Ws": w // cube_size,
        "L": l,
        "F": f,
    }
    return patches, meta
