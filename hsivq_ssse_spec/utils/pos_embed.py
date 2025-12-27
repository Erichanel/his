from __future__ import annotations

import torch


def get_1d_sincos_pos_embed(T: int, D: int) -> torch.Tensor:
    assert D > 0
    half = D // 2
    pos = torch.arange(T, dtype=torch.float32)
    omega = torch.arange(half, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(half, 1)))
    out = pos[:, None] * omega[None, :]
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    if D % 2 == 1:
        emb = torch.cat([emb, torch.zeros((T, 1), dtype=emb.dtype)], dim=1)
    return emb


def get_2d_sincos_pos_embed(Hs: int, Ws: int, D: int) -> torch.Tensor:
    assert D > 0
    half = D // 2
    emb_h = get_1d_sincos_pos_embed(Hs, half)
    emb_w = get_1d_sincos_pos_embed(Ws, half)
    grid_h, grid_w = torch.meshgrid(
        torch.arange(Hs, dtype=torch.long),
        torch.arange(Ws, dtype=torch.long),
        indexing="ij",
    )
    pos_h = emb_h[grid_h.reshape(-1)]
    pos_w = emb_w[grid_w.reshape(-1)]
    emb = torch.cat([pos_h, pos_w], dim=1)
    if D % 2 == 1:
        emb = torch.cat([emb, torch.zeros((Hs * Ws, 1), dtype=emb.dtype)], dim=1)
    return emb


def build_separable_pos_embed(
    Hs: int,
    Ws: int,
    T: int,
    D: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    pos2d = get_2d_sincos_pos_embed(Hs, Ws, D)
    pos1d = get_1d_sincos_pos_embed(T, D)
    if device is not None:
        pos2d = pos2d.to(device)
        pos1d = pos1d.to(device)
    return pos2d[:, None, :] + pos1d[None, :, :]
