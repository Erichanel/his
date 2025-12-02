"""Spatial CNN VQ-VAE for SSPQ Tokenizer (Stage 1)."""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class SpatialCNNVQVAE(nn.Module):
    def __init__(
        self,
        num_bands: int,
        hidden_dim: int,
        codebook_size: int,
        encoder_channels: list,
        commitment_beta: float,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.encoder_channels = encoder_channels
        self.commitment_beta = commitment_beta

        self.spectral_reduce = nn.Conv2d(num_bands, hidden_dim, kernel_size=1)

        enc_layers = []
        in_ch = hidden_dim
        for ch in encoder_channels:
            enc_layers.append(nn.Conv2d(in_ch, ch, kernel_size=3, stride=2, padding=1))
            enc_layers.append(nn.ReLU())
            in_ch = ch
        self.encoder = nn.Sequential(*enc_layers)
        self.proj = nn.Conv2d(in_ch, hidden_dim, kernel_size=1)

        self.codebook = nn.Embedding(codebook_size, hidden_dim)
        self.recon_head = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.final_recon = nn.Conv2d(hidden_dim, num_bands, kernel_size=1)

    def quantize(self, z_tokens: torch.Tensor):
        # z_tokens: [batch, T, D]
        flat = z_tokens.reshape(-1, self.hidden_dim)
        codebook = self.codebook.weight
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        quantized = self.codebook(indices).view(z_tokens.shape)
        return quantized, indices

    def forward(self, x: torch.Tensor):
        """Accepts [batch, B, S, S]; returns recon, quantized, indices, and loss dict."""
        batch, _, S, _ = x.shape
        feat = self.spectral_reduce(x)
        feat = self.encoder(feat)
        feat = self.proj(feat)  # [batch, hidden_dim, h, w]
        b, c, h, w = feat.shape
        tokens = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [batch, T, D]

        quantized, indices = self.quantize(tokens)
        quantized_st = tokens + (quantized - tokens).detach()

        quant_map = quantized_st.view(b, h, w, c).permute(0, 3, 1, 2)  # [batch, D, h, w]
        recon_feat = F.relu(self.recon_head(quant_map))
        recon_feat = F.interpolate(recon_feat, size=(S, S), mode="bilinear", align_corners=False)
        recon = self.final_recon(recon_feat)  # [batch, B, S, S]

        recon_loss = F.mse_loss(recon, x)
        vq_loss = F.mse_loss(quantized.detach(), tokens) + self.commitment_beta * F.mse_loss(
            quantized, tokens.detach()
        )
        loss = recon_loss + vq_loss

        return {
            "recon": recon,
            "quantized": quantized_st,
            "indices": indices.view(batch, -1),
            "loss": loss,
            "loss_dict": {"recon_loss": recon_loss, "vq_loss": vq_loss},
            "shape": (h, w),
        }

    def decode_indices(self, indices: torch.LongTensor) -> torch.Tensor:
        """Decode spatial indices to a reconstructed patch approximation."""
        batch, T = indices.shape
        quantized = self.codebook(indices)  # [batch, T, D]
        side = int(math.ceil(math.sqrt(T)))
        if side * side != T:
            pad = side * side - T
            pad_emb = torch.zeros((batch, pad, self.hidden_dim), device=quantized.device, dtype=quantized.dtype)
            quantized = torch.cat([quantized, pad_emb], dim=1)
        quant_map = quantized.view(batch, side, side, self.hidden_dim).permute(0, 3, 1, 2)
        recon_feat = F.relu(self.recon_head(quant_map))
        up_factor = 2 ** len(self.encoder_channels)
        target_size = side * up_factor
        recon_feat = F.interpolate(
            recon_feat, size=(target_size, target_size), mode="bilinear", align_corners=False
        )
        recon = self.final_recon(recon_feat)
        return recon
