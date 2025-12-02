"""Joint tokenizer that combines spectral and spatial VQ-VAE branches."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .spectral_vqvae import SpectralTCNVQVAE
from .spatial_vqvae import SpatialCNNVQVAE


class SSPQTokenizer(nn.Module):
    def __init__(self, spectral_cfg: dict, spatial_cfg: dict) -> None:
        super().__init__()
        self.spectral_vqvae = SpectralTCNVQVAE(**spectral_cfg)
        self.spatial_vqvae = SpatialCNNVQVAE(**spatial_cfg)
        self.K_s = spectral_cfg["codebook_size"]
        self.K_x = spatial_cfg["codebook_size"]
        self.joint_vocab_size = self.K_s * self.K_x

    def forward(self, patches: torch.Tensor, return_recon: bool = False) -> dict:
        """Encode patches and return joint IDs plus optional reconstructions."""
        spec_out = self.spectral_vqvae(patches)
        spat_out = self.spatial_vqvae(patches)

        T_s = spec_out["indices"].shape[1]
        T_x = spat_out["indices"].shape[1]
        T = min(T_s, T_x)

        k = spec_out["indices"][:, :T]
        m = spat_out["indices"][:, :T]
        joint_ids = k + self.K_s * m

        loss = spec_out["loss"] + spat_out["loss"]
        out = {
            "joint_ids": joint_ids,
            "spectral_ids": spec_out["indices"],
            "spatial_ids": spat_out["indices"],
            "loss": loss,
            "loss_dict": {
                "spectral": spec_out["loss_dict"],
                "spatial": spat_out["loss_dict"],
            },
        }
        if return_recon:
            out["recon_s"] = spec_out.get("recon")
            out["recon_x"] = spat_out.get("recon")
        return out

    def decode_joint_ids(self, joint_ids: torch.LongTensor):
        """Reconstruct patches from joint IDs by recovering spectral/spatial indices."""
        m = joint_ids // self.K_s
        k = joint_ids % self.K_s
        spec_recon = self.spectral_vqvae.decode_indices(k)  # [batch, B]
        spatial_recon = self.spatial_vqvae.decode_indices(m)  # [batch, B, h, w]

        if spatial_recon.dim() == 4:
            spec_map = spec_recon.unsqueeze(-1).unsqueeze(-1)
            spec_map = F.interpolate(spec_map, size=spatial_recon.shape[-2:], mode="bilinear", align_corners=False)
            recon = (spec_map + spatial_recon) / 2
        else:
            recon = spec_recon
        return recon

    def save_codebooks(self, path: str) -> None:
        """Persist codebooks and relevant parameters."""
        state = {
            "spectral": self.spectral_vqvae.state_dict(),
            "spatial": self.spatial_vqvae.state_dict(),
            "K_s": self.K_s,
            "K_x": self.K_x,
        }
        torch.save(state, path)

    def load_codebooks(self, path: str) -> None:
        """Load codebooks and relevant parameters."""
        state = torch.load(path, map_location="cpu")
        self.spectral_vqvae.load_state_dict(state["spectral"])
        self.spatial_vqvae.load_state_dict(state["spatial"])
