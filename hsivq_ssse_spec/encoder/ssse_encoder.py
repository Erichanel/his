from __future__ import annotations

import torch
from torch import nn

from .vit_blocks import TransformerBlock


class SSSEEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth_spa: int,
        depth_spe: int,
        depth_fuse: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.spa_blocks = nn.ModuleList(
            [
                TransformerBlock(dim, heads, mlp_ratio, dropout, attn_dropout)
                for _ in range(depth_spa)
            ]
        )
        self.spe_blocks = nn.ModuleList(
            [
                TransformerBlock(dim, heads, mlp_ratio, dropout, attn_dropout)
                for _ in range(depth_spe)
            ]
        )
        self.fuse_blocks = nn.ModuleList(
            [
                TransformerBlock(dim, heads, mlp_ratio, dropout, attn_dropout)
                for _ in range(depth_fuse)
            ]
        )

    def forward(self, tok, mask):
        b, p, t, d = tok.shape
        assert mask.shape == (b, p, t)
        tok = tok.clone()
        mask_token = self.mask_token.expand(b, p, t, d)
        tok = torch.where(mask.unsqueeze(-1), mask_token, tok)

        # Spatial encoding: [B, P, T, D] -> [B, T, P, D] -> [B*T, P, D]
        z_spa = tok.permute(0, 2, 1, 3).reshape(b * t, p, d)
        for blk in self.spa_blocks:
            z_spa = blk(z_spa)
        z_spa = z_spa.reshape(b, t, p, d).permute(0, 2, 1, 3)

        # Spectral encoding: [B, P, T, D] -> [B*P, T, D]
        z_spe = tok.reshape(b * p, t, d)
        for blk in self.spe_blocks:
            z_spe = blk(z_spe)
        z_spe = z_spe.reshape(b, p, t, d)

        z = z_spa + z_spe
        z = z.reshape(b, p * t, d)
        for blk in self.fuse_blocks:
            z = blk(z)
        z = z.reshape(b, p, t, d)
        return z
