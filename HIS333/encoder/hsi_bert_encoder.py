"""HSI token encoder (BERT style) and UniversalHSIModel wrapper."""

from __future__ import annotations

from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, BertForMaskedLM


class HSIEncoderBERT(nn.Module):
    def __init__(
        self,
        joint_vocab_size: int,
        num_domains: int,
        bert_model_name_or_path: str,
        load_pretrained_lm: bool,
        mask_ratio: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.joint_vocab_size = joint_vocab_size
        self.num_domains = num_domains
        self.bert_model_name_or_path = bert_model_name_or_path
        self.load_pretrained_lm = load_pretrained_lm
        self.mask_ratio = mask_ratio
        self.max_seq_len = max_seq_len
        self.mask_token_id = joint_vocab_size  # reserve last id as mask

        if load_pretrained_lm:
            self.encoder = BertForMaskedLM.from_pretrained(bert_model_name_or_path)
            self.encoder.config.output_hidden_states = True
            hidden_size = self.encoder.config.hidden_size
        else:
            config = BertConfig(
                vocab_size=joint_vocab_size + 1,
                max_position_embeddings=max_seq_len,
                hidden_size=256,
                num_hidden_layers=6,
                num_attention_heads=8,
                intermediate_size=512,
                output_hidden_states=True,
            )
            self.encoder = BertForMaskedLM(config)
            hidden_size = config.hidden_size

        self.token_embedding = nn.Embedding(joint_vocab_size + 1, hidden_size)
        self.domain_embedding = nn.Embedding(num_domains, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def mask_tokens(
        self, joint_ids: torch.LongTensor, mask_ratio: Optional[float] = None, strategy: str = "structured"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return masked input IDs and labels for MLM training."""
        mask_ratio = mask_ratio or self.mask_ratio
        device = joint_ids.device
        batch, T = joint_ids.shape

        if strategy == "structured":
            # simple structured approximation: select contiguous spans of length 2-5
            mask = torch.zeros_like(joint_ids, dtype=torch.bool)
            span_len = max(1, int(0.05 * T))
            num_spans = max(1, int(mask_ratio * T / max(span_len, 1)))
            for b in range(batch):
                for _ in range(num_spans):
                    start = torch.randint(0, max(T - span_len, 1), (1,), device=device).item()
                    end = min(T, start + span_len)
                    mask[b, start:end] = True
            if mask.float().mean() < mask_ratio:  # fallback to random mask
                random_mask = torch.rand_like(joint_ids.float()) < mask_ratio
                mask = mask | random_mask
        else:
            mask = torch.rand_like(joint_ids.float()) < mask_ratio

        labels = joint_ids.clone()
        labels[~mask] = -100
        masked_input_ids = joint_ids.clone()
        masked_input_ids[mask] = self.mask_token_id
        return masked_input_ids, labels

    def forward(self, tokens: torch.LongTensor, domain_ids: torch.LongTensor, pretrain: bool = True):
        """Run pretraining or encoding forward pass."""
        batch, T = tokens.shape
        position_ids = torch.arange(T, device=tokens.device).unsqueeze(0).expand(batch, T)

        if pretrain:
            masked_input_ids, labels = self.mask_tokens(tokens, strategy="structured")
            tok_emb = self.token_embedding(masked_input_ids)
            dom_emb = self.domain_embedding(domain_ids).unsqueeze(1)
            pos_emb = self.position_embedding(position_ids)
            inputs_embeds = self.dropout(tok_emb + dom_emb + pos_emb)
            outputs = self.encoder(inputs_embeds=inputs_embeds, labels=labels)
            return outputs.loss, outputs.logits, outputs.hidden_states[-1]
        else:
            tok_emb = self.token_embedding(tokens)
            dom_emb = self.domain_embedding(domain_ids).unsqueeze(1)
            pos_emb = self.position_embedding(position_ids)
            inputs_embeds = self.dropout(tok_emb + dom_emb + pos_emb)
            outputs = self.encoder(inputs_embeds=inputs_embeds, output_hidden_states=True, return_dict=True)
            hidden = outputs.hidden_states[-1] if outputs.hidden_states else outputs.last_hidden_state
            return hidden


class UniversalHSIModel(nn.Module):
    def __init__(self, tokenizer, encoder: HSIEncoderBERT) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder

    def encode_patches(self, patches: torch.Tensor, domain_ids: torch.LongTensor, pretrain: bool = False):
        """Tokenize patches then run through the encoder."""
        tok_out = self.tokenizer(patches, return_recon=False)
        joint_ids = tok_out["joint_ids"]
        if pretrain:
            loss, logits, hidden = self.encoder(joint_ids, domain_ids, pretrain=True)
            return {"loss": loss, "logits": logits, "hidden": hidden, "joint_ids": joint_ids}
        hidden = self.encoder(joint_ids, domain_ids, pretrain=False)
        return {"hidden": hidden, "joint_ids": joint_ids}
