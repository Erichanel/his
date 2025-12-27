from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        dim: int,
        decay: float = 0.99,
        beta: float = 0.25,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.decay = decay
        self.beta = beta
        self.eps = eps

        self.embedding = nn.Embedding(codebook_size, dim)
        nn.init.kaiming_uniform_(self.embedding.weight, a=5**0.5)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", self.embedding.weight.data.clone())

    def forward(self, z_e: torch.Tensor):
        if z_e.size(-1) != self.dim:
            raise ValueError(f"z_e last dim {z_e.size(-1)} != {self.dim}")
        z = z_e.reshape(-1, self.dim)
        distances = (
            z.pow(2).sum(dim=1, keepdim=True)
            - 2 * z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(indices, self.codebook_size).type_as(z)
        z_q = torch.matmul(encodings, self.embedding.weight).view_as(z_e)
        z_q_st = z_e + (z_q - z_e).detach()

        if self.training:
            cluster_size = encodings.sum(dim=0)
            self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            embed_sum = encodings.t() @ z
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps)
                / (n + self.codebook_size * self.eps)
                * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

        commitment_loss = F.mse_loss(z_q.detach(), z_e)
        vq_loss = self.beta * commitment_loss

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, indices.view(z_e.shape[:-1]), vq_loss, perplexity
