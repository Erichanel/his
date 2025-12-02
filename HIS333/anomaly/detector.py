"""HSIAnomalyDetector combining multiple heads with the universal encoder."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from data.datasets import HSIPatchExtractor
from .heads import compute_lm_nll_scores, compute_recon_error, OneClassHead


class HSIAnomalyDetector(nn.Module):
    def __init__(
        self,
        universal_model,
        hidden_size: int,
        freeze_encoder: bool = True,
        use_lora: bool = False,
        head_weights: Optional[dict] = None,
        patch_size: int = 11,
        stride: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.universal_model = universal_model
        self.one_class_head = OneClassHead(hidden_size)
        self.freeze_encoder = freeze_encoder
        self.use_lora = use_lora
        self.head_weights = head_weights or {"lm_nll": 0.4, "recon": 0.3, "one_class": 0.3}
        self.patch_size = patch_size
        self.stride = stride or patch_size // 2

        if self.freeze_encoder:
            for p in self.universal_model.parameters():
                p.requires_grad = False

    def forward(self, patches: torch.Tensor, domain_ids: torch.LongTensor, mode: str = "train") -> dict:
        """Compute per-head scores and a combined anomaly score."""
        out = self.universal_model.encode_patches(patches, domain_ids, pretrain=False)
        hidden = out["hidden"]
        joint_ids = out["joint_ids"]
        lm_scores = compute_lm_nll_scores(self.universal_model.encoder, joint_ids, domain_ids)  # [batch, T]
        recon_scores = compute_recon_error(self.universal_model.tokenizer, patches, joint_ids)  # [batch]
        oc_scores = self.one_class_head(hidden.mean(dim=1))  # [batch]

        lm_patch = lm_scores.mean(dim=1)
        recon_patch = recon_scores if recon_scores.dim() == 1 else recon_scores.mean(dim=1)
        oc_patch = oc_scores

        def _zscore(x):
            mu = x.mean()
            sigma = x.std() + 1e-6
            return (x - mu) / sigma

        lm_norm = _zscore(lm_patch)
        recon_norm = _zscore(recon_patch)
        oc_norm = _zscore(oc_patch)

        combined = (
            self.head_weights.get("lm_nll", 0.4) * lm_norm
            + self.head_weights.get("recon", 0.3) * recon_norm
            + self.head_weights.get("one_class", 0.3) * oc_norm
        )

        return {
            "lm_nll": lm_scores,
            "recon": recon_scores,
            "one_class": oc_scores,
            "combined_patch": combined,
            "hidden": hidden,
        }

    def generate_anomaly_map(self, cube, domain_id: int, batch_size: int = 32):
        """Generate pixel-level anomaly map from full HSI cube."""
        if isinstance(cube, torch.Tensor):
            cube_np = cube.cpu().numpy()
        else:
            cube_np = np.asarray(cube)
        H, W, _ = cube_np.shape
        extractor = HSIPatchExtractor(self.patch_size, stride=self.stride)
        patches, coords = extractor.extract_patches(cube_np)
        device = next(self.parameters()).device

        score_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)

        for start in range(0, len(patches), batch_size):
            end = start + batch_size
            batch_patches = patches[start:end].to(device)
            batch_coords = coords[start:end]
            domain_ids = torch.full((len(batch_patches),), domain_id, dtype=torch.long, device=device)
            with torch.no_grad():
                out = self.forward(batch_patches, domain_ids, mode="eval")
                scores = out["combined_patch"].detach().cpu().numpy()
            for i, score in enumerate(scores):
                r, c = batch_coords[i].tolist()
                score_map[r : r + self.patch_size, c : c + self.patch_size] += score
                count_map[r : r + self.patch_size, c : c + self.patch_size] += 1

        anomaly_map = score_map / (count_map + 1e-6)
        return anomaly_map
