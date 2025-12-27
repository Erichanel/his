from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from .vq import VectorQuantizerEMA


class SSPQTokenizer(nn.Module):
    def __init__(
        self,
        D_vq: int,
        Ks: int,
        Kx: int,
        vq_decay: float = 0.99,
        vq_beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.D_vq = D_vq
        self.Ks = Ks
        self.Kx = Kx

        self.s_conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.s_conv2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.s_act = nn.GELU()
        self.s_fc = nn.Linear(16 * 8, D_vq)
        self.vq_s = VectorQuantizerEMA(Ks, D_vq, decay=vq_decay, beta=vq_beta)
        self.s_dec_fc = nn.Linear(D_vq, 16 * 8)
        self.s_dec_conv1 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.s_dec_conv2 = nn.Conv1d(16, 1, kernel_size=3, padding=1)

        self.x_conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.x_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.x_act = nn.GELU()
        self.x_fc = nn.Linear(16 * 3 * 3, D_vq)
        self.vq_x = VectorQuantizerEMA(Kx, D_vq, decay=vq_decay, beta=vq_beta)
        self.x_dec_fc = nn.Linear(D_vq, 16 * 3 * 3)
        self.x_dec_conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.x_dec_conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)

    def forward(self, cubes: torch.Tensor, return_recon: bool = False) -> Dict[str, torch.Tensor]:
        if cubes.ndim != 4:
            raise ValueError(f"Expected [B, P, T, F], got {tuple(cubes.shape)}")
        b, p, t, f = cubes.shape
        assert p == 9 and t == 8 and f == 72, "Cubes must be [B,9,8,72]"

        cubes = (
            cubes.reshape(b, p, t, 8, 3, 3)
            .permute(0, 1, 2, 4, 5, 3)
            .contiguous()
        )
        assert cubes.shape == (b, p, t, 3, 3, 8)

        # Spectral branch
        spec = cubes.mean(dim=(3, 4))  # [B, P, T, 8]
        spec_target = spec
        spec_seq = spec.reshape(b * p * t, 1, 8)
        s = self.s_act(self.s_conv1(spec_seq))
        s = self.s_act(self.s_conv2(s))
        s = s.flatten(1)
        z_e_s = self.s_fc(s)
        z_q_s, k_idx, vq_loss_s, perplex_s = self.vq_s(z_e_s)
        emb_s = z_q_s.view(b, p, t, self.D_vq)
        k_idx = k_idx.view(b, p, t)
        recon_s = None
        recon_loss_s = None
        if return_recon:
            s_dec = self.s_dec_fc(z_q_s).view(-1, 16, 8)
            s_dec = self.s_act(self.s_dec_conv1(s_dec))
            s_dec = self.s_dec_conv2(s_dec).squeeze(1)
            recon_s = s_dec.view(b, p, t, 8)
            recon_loss_s = F.mse_loss(recon_s, spec_target)

        # Spatial branch
        spatial = cubes.permute(0, 1, 2, 5, 3, 4).reshape(b * p * t, 8, 3, 3)
        spatial_target = spatial
        x = self.x_act(self.x_conv1(spatial))
        x = self.x_act(self.x_conv2(x))
        x = x.flatten(1)
        z_e_x = self.x_fc(x)
        z_q_x, m_idx, vq_loss_x, perplex_x = self.vq_x(z_e_x)
        emb_x = z_q_x.view(b, p, t, self.D_vq)
        m_idx = m_idx.view(b, p, t)
        recon_x = None
        recon_loss_x = None
        if return_recon:
            x_dec = self.x_dec_fc(z_q_x).view(-1, 16, 3, 3)
            x_dec = self.x_act(self.x_dec_conv1(x_dec))
            recon_x = self.x_dec_conv2(x_dec)
            recon_loss_x = F.mse_loss(recon_x, spatial_target)
            recon_x = (
                recon_x.view(b, p, t, 8, 3, 3)
                .permute(0, 1, 2, 4, 5, 3)
                .contiguous()
            )

        assert emb_s.shape == (b, p, t, self.D_vq)
        assert emb_x.shape == (b, p, t, self.D_vq)

        out = {
            "k_idx": k_idx,
            "m_idx": m_idx,
            "emb_s": emb_s,
            "emb_x": emb_x,
            "vq_loss_dict": {"s": vq_loss_s, "x": vq_loss_x},
            "perplexity_s": perplex_s,
            "perplexity_x": perplex_x,
            "total": vq_loss_s + vq_loss_x,
        }
        if return_recon:
            out.update(
                {
                    "recon_s": recon_s,
                    "recon_x": recon_x,
                    "recon_loss_s": recon_loss_s,
                    "recon_loss_x": recon_loss_x,
                    "recon_loss_total": recon_loss_s + recon_loss_x,
                }
            )
        return out
