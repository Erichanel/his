"""
Hyperspectral modeling pipeline inspired by CrossTimeNet.

The module mirrors the time-series tokenizer + BERT pretraining pipeline
but adapts it to 3D hyperspectral cubes (H, W, Bands):

1. Spectral tokenizer: a TCN VQ-VAE operating on per-pixel spectra.
2. Spatial tokenizer: a CNN VQ-VAE that quantizes local spatial patches.
3. Cross-domain pre-training: BERT-style masked token modeling on the
   concatenated spectral + spatial code streams.
4. Target heads: classification or anomaly detection heads that reuse the
   shared BERT encoder.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertForMaskedLM

from hsi_vq import Quantize, VQVAE


class SpectralTCNVQVAE(nn.Module):
    """Tokenize per-pixel spectra with the existing TCN VQ-VAE.

    Args:
        num_bands: Number of spectral bands in the hyperspectral cube.
        hidden_dim: Hidden size for the TCN encoder/decoder.
        n_embed: Codebook size for spectral tokens.
        wave_length: Patch length along the spectral axis.
        block_num: Number of residual blocks for the temporal backbone.
    """

    def __init__(
        self,
        num_bands: int,
        hidden_dim: int = 64,
        n_embed: int = 512,
        wave_length: int = 8,
        block_num: int = 4,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.n_embed = n_embed
        self.model = VQVAE(
            data_shape=(num_bands, 1),
            hidden_dim=hidden_dim,
            n_embed=n_embed,
            wave_length=wave_length,
            block_num=block_num,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns reconstruction, commit loss, and token ids.

        Input can be channel-first ``(B, C, H, W)`` or channel-last
        ``(B, H, W, C)``; it is converted internally to per-pixel spectra.
        """

        if x.dim() != 4:
            raise ValueError(f"Expected input of shape (B, C, H, W) or (B, H, W, C), got {tuple(x.shape)}")

        channel_first = x.shape[1] == self.num_bands
        if channel_first:
            x_flat = x.permute(0, 2, 3, 1)
        else:
            x_flat = x

        b, h, w, _ = x_flat.shape
        spectra = x_flat.reshape(b * h * w, self.num_bands, 1)
        recon, diff, token_ids = self.model(spectra)

        recon = recon.reshape(b, h, w, self.num_bands)
        token_ids = token_ids.reshape(b, h, w, -1)

        if channel_first:
            recon = recon.permute(0, 3, 1, 2)

        return recon, diff, token_ids


class SpatialCNNVQVAE(nn.Module):
    """CNN-based spatial VQ-VAE for hyperspectral patches."""

    def __init__(
        self,
        num_bands: int,
        hidden_dim: int = 64,
        n_embed: int = 512,
        patch_size: int = 4,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.n_embed = n_embed
        self.patch_size = patch_size

        self.encoder = nn.Sequential(
            nn.Conv2d(num_bands, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.quantize_input = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.quantize = Quantize(hidden_dim, n_embed)
        self.quantize_output = nn.ConvTranspose2d(
            hidden_dim,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_bands, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns reconstruction, commit loss, and patch token ids."""

        if x.dim() != 4:
            raise ValueError(f"Expected input of shape (B, C, H, W) or (B, H, W, C), got {tuple(x.shape)}")

        channel_first = x.shape[1] == self.num_bands
        if channel_first:
            feat = x
        else:
            feat = x.permute(0, 3, 1, 2)

        encoded = self.encoder(feat)
        patches = self.quantize_input(encoded).permute(0, 2, 3, 1)
        quantized, diff, token_ids = self.quantize(patches)
        quantized = quantized.permute(0, 3, 1, 2)

        decoded = self.decoder(self.quantize_output(quantized))
        token_ids = token_ids

        if not channel_first:
            decoded = decoded.permute(0, 2, 3, 1)

        return decoded, diff, token_ids


class HSIBert(nn.Module):
    """End-to-end hyperspectral encoder with BERT-style cross-domain pretraining."""

    def __init__(
        self,
        num_bands: int,
        num_classes: int,
        spectral_codebook: int = 512,
        spatial_codebook: int = 512,
        spectral_hidden: int = 64,
        spatial_hidden: int = 64,
        wave_length: int = 8,
        patch_size: int = 4,
        mask_ratio: float = 0.3,
        bert_model: str = "bert-base-uncased",
        pooling: str = "mean",
    ):
        super().__init__()
        if pooling not in {"mean", "max", "cls"}:
            raise ValueError("pooling must be one of {'mean', 'max', 'cls'}")

        self.pooling = pooling
        self.mask_ratio = mask_ratio

        self.spectral_tokenizer = SpectralTCNVQVAE(
            num_bands=num_bands,
            hidden_dim=spectral_hidden,
            n_embed=spectral_codebook,
            wave_length=wave_length,
        )
        self.spatial_tokenizer = SpatialCNNVQVAE(
            num_bands=num_bands,
            hidden_dim=spatial_hidden,
            n_embed=spatial_codebook,
            patch_size=patch_size,
        )

        config = BertConfig.from_pretrained(bert_model, output_hidden_states=True)
        vocab_size = spectral_codebook + spatial_codebook + 1
        self.mask_token = vocab_size - 1
        self.encoder = BertForMaskedLM.from_pretrained(
            bert_model,
            output_attentions=True,
            output_hidden_states=True,
        )
        self.encoder.resize_token_embeddings(vocab_size)
        self.encoder.config.vocab_size = vocab_size

        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.anomaly_head = nn.Linear(config.hidden_size, 1)

    def _tokenize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        spec_recon, spec_diff, spec_ids = self.spectral_tokenizer(x)
        spa_recon, spa_diff, spa_ids = self.spatial_tokenizer(x)

        b = x.shape[0]
        spec_tokens = spec_ids.reshape(b, -1)
        spa_tokens = spa_ids.reshape(b, -1) + self.spectral_tokenizer.n_embed
        tokens = torch.cat([spec_tokens, spa_tokens], dim=1)
        return spec_recon, spa_recon, spec_diff + spa_diff, tokens

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        pretrain: bool = False,
    ) -> Dict[str, torch.Tensor]:
        spec_recon, spa_recon, commit_loss, tokens = self._tokenize(x)
        tokens = tokens.to(dtype=torch.long)

        if pretrain:
            labels = tokens.clone()
            mask = torch.rand_like(labels.float()) < self.mask_ratio
            inputs = tokens.clone()
            inputs[mask] = self.mask_token
            labels[~mask] = -100

            outputs = self.encoder(input_ids=inputs, labels=labels, return_dict=True)
            loss = outputs.loss + commit_loss
            return {
                "loss": loss,
                "mlm_loss": outputs.loss,
                "commitment_loss": commit_loss,
                "spec_recon": spec_recon,
                "spa_recon": spa_recon,
                "logits": outputs.logits,
            }

        encoded = self.encoder.bert(input_ids=tokens, output_hidden_states=True, return_dict=True).last_hidden_state
        pooled = self._pool(encoded)
        logits = self.classifier(pooled)
        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "spec_recon": spec_recon,
            "spa_recon": spa_recon,
            "commitment_loss": commit_loss,
        }

        anomaly_score = torch.sigmoid(self.anomaly_head(pooled))
        outputs["anomaly_score"] = anomaly_score

        if targets is not None:
            outputs["classification_loss"] = F.cross_entropy(logits, targets)

        return outputs

    def _pool(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        if self.pooling == "max":
            return hidden.max(dim=1).values
        # "cls": take the first token
        return hidden[:, 0, :]


__all__ = [
    "SpectralTCNVQVAE",
    "SpatialCNNVQVAE",
    "HSIBert",
]
