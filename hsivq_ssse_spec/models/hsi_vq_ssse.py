from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from utils.cube_partition import make_cubes
from utils.masking import hsimae_consistent_mask
from utils.pos_embed import build_separable_pos_embed
from sspq_tokenizer.tokenizer import SSPQTokenizer
from encoder.ssse_encoder import SSSEEncoder


class HSIVQSSSEModel(nn.Module):
    def __init__(
        self,
        patch_size: int = 9,
        cube_size: int = 3,
        T: int = 8,
        L: int = 8,
        Ks: int = 512,
        Kx: int = 512,
        D: int = 384,
        D_vq: int = 256,
        depth_spa: int = 6,
        depth_spe: int = 6,
        depth_fuse: int = 2,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        mr_spa: float = 0.5,
        mr_spe: float = 0.5,
        num_classes: int = 9,
        vq_decay: float = 0.99,
        vq_beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.cube_size = cube_size
        self.T = T
        self.L = L
        self.Ks = Ks
        self.Kx = Kx
        self.D = D
        self.D_vq = D_vq
        self.mr_spa = mr_spa
        self.mr_spe = mr_spe

        self.tokenizer = SSPQTokenizer(D_vq, Ks, Kx, vq_decay=vq_decay, vq_beta=vq_beta)
        self.token_fuse = nn.Sequential(
            nn.Linear(2 * D_vq, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.encoder = SSSEEncoder(
            dim=D,
            depth_spa=depth_spa,
            depth_spe=depth_spe,
            depth_fuse=depth_fuse,
            heads=heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        self.head_k = nn.Linear(D, Ks)
        self.head_m = nn.Linear(D, Kx)
        self.cls_head = nn.Linear(D, num_classes)

    def _encode_tokens(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        cubes, meta = make_cubes(x, patch_size=self.patch_size, cube_size=self.cube_size, T=self.T)
        # Ensure tokenizer is in correct mode for gradient flow
        # For training: keep tokenizer in train mode so VQ gradients flow
        tok_out = self.tokenizer(cubes)
        emb = torch.cat([tok_out["emb_s"], tok_out["emb_x"]], dim=-1)
        tok = self.token_fuse(emb)
        pos = build_separable_pos_embed(
            meta["Hs"], meta["Ws"], self.T, self.D, device=tok.device
        )
        tok = tok + pos[None, :, :, :]
        z = self.encoder(tok, mask)
        return z, {"meta": meta, **tok_out}

    def _nll_map(self, logits_k, logits_m, k_idx, m_idx) -> torch.Tensor:
        b, p, t, _ = logits_k.shape
        logp_k = F.log_softmax(logits_k, dim=-1)
        logp_m = F.log_softmax(logits_m, dim=-1)
        nll_k = -logp_k.gather(-1, k_idx.unsqueeze(-1)).squeeze(-1)
        nll_m = -logp_m.gather(-1, m_idx.unsqueeze(-1)).squeeze(-1)
        nll_map = nll_k + nll_m
        assert nll_map.shape == (b, p, t)
        return nll_map

    def forward_pretrain(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = x.shape[0]
        p = (self.patch_size // self.cube_size) ** 2
        mask, mask_stats = hsimae_consistent_mask(
            b, P=p, T=self.T, mr_spa=self.mr_spa, mr_spe=self.mr_spe, device=x.device
        )
        z, tok_out = self._encode_tokens(x, mask)
        logits_k = self.head_k(z)
        logits_m = self.head_m(z)
        k_idx = tok_out["k_idx"]
        m_idx = tok_out["m_idx"]

        mask_flat = mask.reshape(-1)
        logits_k_flat = logits_k.reshape(-1, self.Ks)
        logits_m_flat = logits_m.reshape(-1, self.Kx)
        k_flat = k_idx.reshape(-1)
        m_flat = m_idx.reshape(-1)
        if mask_flat.any().item():
            loss_k = F.cross_entropy(logits_k_flat[mask_flat], k_flat[mask_flat])
            loss_m = F.cross_entropy(logits_m_flat[mask_flat], m_flat[mask_flat])
            loss_mtp = loss_k + loss_m
        else:
            loss_mtp = torch.tensor(0.0, device=x.device)

        nll_map = self._nll_map(logits_k, logits_m, k_idx, m_idx)
        vq_loss = tok_out["total"]
        loss_total = loss_mtp + vq_loss

        return {
            "tok_out": tok_out,
            "tok": z,
            "logits_k": logits_k,
            "logits_m": logits_m,
            "mask": mask,
            "mask_stats": mask_stats,
            "nll_map": nll_map,
            "loss_mtp": loss_mtp,
            "vq_loss": vq_loss,
            "loss_total": loss_total,
        }

    def forward_classify(self, x: torch.Tensor, y: torch.Tensor | None = None, lambda_mtp: float = 0.0):
        b = x.shape[0]
        p = (self.patch_size // self.cube_size) ** 2
        mask = torch.zeros((b, p, self.T), dtype=torch.bool, device=x.device)
        z, tok_out = self._encode_tokens(x, mask)
        pooled = z.mean(dim=(1, 2))
        logits_cls = self.cls_head(pooled)

        out = {
            "logits_cls": logits_cls,
            "tok": z,
        }
        if y is None:
            return out

        loss_cls = F.cross_entropy(logits_cls, y)
        loss_total = loss_cls
        if lambda_mtp > 0:
            mask_mtp, _ = hsimae_consistent_mask(
                b, P=p, T=self.T, mr_spa=self.mr_spa, mr_spe=self.mr_spe, device=x.device
            )
            z_mtp, tok_out_mtp = self._encode_tokens(x, mask_mtp)
            logits_k = self.head_k(z_mtp)
            logits_m = self.head_m(z_mtp)
            nll_map = self._nll_map(logits_k, logits_m, tok_out_mtp["k_idx"], tok_out_mtp["m_idx"])
            mask_flat = mask_mtp.reshape(-1)
            if mask_flat.any().item():
                loss_k = F.cross_entropy(
                    logits_k.reshape(-1, self.Ks)[mask_flat],
                    tok_out_mtp["k_idx"].reshape(-1)[mask_flat],
                )
                loss_m = F.cross_entropy(
                    logits_m.reshape(-1, self.Kx)[mask_flat],
                    tok_out_mtp["m_idx"].reshape(-1)[mask_flat],
                )
                loss_mtp = loss_k + loss_m
            else:
                loss_mtp = torch.tensor(0.0, device=x.device)
            loss_total = loss_total + lambda_mtp * loss_mtp + tok_out_mtp["total"]
            out.update(
                {
                    "nll_map": nll_map,
                    "loss_mtp": loss_mtp,
                    "vq_loss": tok_out_mtp["total"],
                    "logits_k": logits_k,
                    "logits_m": logits_m,
                }
            )

        out.update({"loss_cls": loss_cls, "loss_total": loss_total})
        return out

    def forward_anomaly(
        self,
        x: torch.Tensor,
        score_type: str = "mean",
        topk_ratio: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        b = x.shape[0]
        p = (self.patch_size // self.cube_size) ** 2
        mask = torch.zeros((b, p, self.T), dtype=torch.bool, device=x.device)
        z, tok_out = self._encode_tokens(x, mask)
        logits_k = self.head_k(z)
        logits_m = self.head_m(z)
        nll_map = self._nll_map(logits_k, logits_m, tok_out["k_idx"], tok_out["m_idx"])
        nll_flat = nll_map.reshape(b, -1)
        if score_type == "topk":
            k = max(1, int(topk_ratio * nll_flat.shape[1]))
            topk_vals = torch.topk(nll_flat, k=k, dim=1).values
            anomaly_score = topk_vals.mean(dim=1)
        else:
            anomaly_score = nll_flat.mean(dim=1)

        nll_spa = nll_map.mean(dim=2)
        hs = self.patch_size // self.cube_size
        anomaly_map_spatial = nll_spa.reshape(b, hs, hs)
        return {
            "anomaly_score": anomaly_score,
            "anomaly_map_spatial": anomaly_map_spatial,
            "nll_map": nll_map,
            "tok": z,
        }
