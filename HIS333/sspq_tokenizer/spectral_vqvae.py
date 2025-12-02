"""Spectral TCN VQ-VAE for SSPQ Tokenizer (Stage 1)."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SpectralTCNVQVAE(nn.Module):
    def __init__(
        self,
        num_bands: int,
        hidden_dim: int,
        codebook_size: int,
        num_layers: int,
        kernel_size: int,
        commitment_beta: float,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.commitment_beta = commitment_beta

        layers = []
        in_ch = 1
        dilation = 1  # keep dilation at 1 to avoid effective kernel > num_bands on small inputs
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_ch, hidden_dim, kernel_size, padding=padding, dilation=dilation))
            layers.append(nn.ReLU())
            in_ch = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.codebook = nn.Embedding(codebook_size, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, num_bands)

    def quantize(self, z: torch.Tensor):
        # z: [batch, hidden_dim]
        flat_z = z.view(-1, self.hidden_dim)  # [B, D]
        codebook = self.codebook.weight  # [K, D]
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ codebook.t()
            + codebook.pow(2).sum(dim=1)
        )  # [B, K]
        indices = distances.argmin(dim=1)
        quantized = self.codebook(indices).view(z.shape)
        return quantized, indices

    def forward(self, x: torch.Tensor):
        """Accepts [batch, B, S, S]; returns recon, quantized, indices, and loss dict."""
        batch = x.shape[0]
        x_mean = x.mean(dim=(-1, -2))  # [batch, B]
        z_in = x_mean.unsqueeze(1)  # [batch, 1, B]
        z = self.encoder(z_in)  # [batch, hidden_dim, B]
        z = z.mean(dim=-1)  # [batch, hidden_dim] -> T_s = 1 token

        quantized, indices = self.quantize(z)
        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()

        recon = self.decoder(quantized_st)  # [batch, B]
        recon_loss = F.mse_loss(recon, x_mean)
        vq_loss = F.mse_loss(quantized.detach(), z) + self.commitment_beta * F.mse_loss(quantized, z.detach())

        loss = recon_loss + vq_loss
        return {
            "recon": recon,
            "quantized": quantized_st.unsqueeze(1),  # [batch, 1, D]
            "indices": indices.view(batch, 1),  # [batch, T_s]
            "loss": loss,
            "loss_dict": {"recon_loss": recon_loss, "vq_loss": vq_loss},
        }

    def decode_indices(self, indices: torch.LongTensor) -> torch.Tensor:
        """Decode spectral indices back to spectral vectors [batch, B]."""
        if indices.dim() == 1:
            indices = indices.unsqueeze(1)
        quantized = self.codebook(indices)  # [batch, T, D]
        recon = self.decoder(quantized)  # [batch, T, B]
        recon = recon.mean(dim=1)  # aggregate over tokens if any
        return recon
