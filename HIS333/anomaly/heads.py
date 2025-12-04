"""Anomaly detection heads: LM-NLL, reconstruction, and one-class."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def compute_lm_nll_scores(
    encoder,
    joint_ids: torch.LongTensor,
    domain_ids: torch.LongTensor,
    mask_strategy: str = "structured",
) -> torch.Tensor:
    """Compute -log p(token) scores using a masked language model."""
    masked_input_ids, labels = encoder.mask_tokens(joint_ids, strategy=mask_strategy)
    batch, T = joint_ids.shape
    device = joint_ids.device
    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(batch, T)

    tok_emb = encoder.token_embedding(masked_input_ids)
    dom_emb = encoder.domain_embedding(domain_ids).unsqueeze(1)
    pos_emb = encoder.position_embedding(position_ids)
    inputs_embeds = encoder.dropout(tok_emb + dom_emb + pos_emb)
    outputs = encoder.encoder(inputs_embeds=inputs_embeds)
    logits = outputs.logits  # [batch, T, vocab]
    log_probs = F.log_softmax(logits, dim=-1)
    labels_flat = labels.view(-1)
    log_probs_flat = log_probs.view(-1, log_probs.size(-1))
    mask_flat = labels_flat != -100
    target_logp = torch.zeros_like(labels_flat, dtype=log_probs.dtype, device=device)
    target_logp[mask_flat] = log_probs_flat[mask_flat, labels_flat[mask_flat]]
    scores_flat = torch.zeros_like(labels_flat, dtype=log_probs.dtype, device=device)
    scores_flat[mask_flat] = -target_logp[mask_flat]
    return scores_flat.view(batch, T)


def compute_recon_error(tokenizer, patches: torch.Tensor, joint_ids: torch.LongTensor) -> torch.Tensor:
    """Compute reconstruction error using tokenizer decoders."""
    recon = tokenizer.decode_joint_ids(joint_ids)
    if recon.dim() == 3:  # [batch, B, h/w]
        recon = recon.unsqueeze(-1)
    if recon.dim() == 4 and patches.dim() == 4:
        err = (recon - patches).pow(2).mean(dim=(1, 2, 3))
    else:
        err = (recon.squeeze() - patches.mean(dim=(-1, -2))).pow(2).mean(dim=1)
    return err


class OneClassHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.center = None

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return one-class scores for patch-level embeddings."""
        logits = self.net(embeddings).squeeze(-1)
        if self.center is not None:
            logits = logits - self.center
        return logits
