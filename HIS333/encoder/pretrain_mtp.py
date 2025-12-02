"""Cross-domain self-supervised pretraining entry (Stage 2)."""

from __future__ import annotations

from typing import Optional
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from data.datasets import HSIPatchExtractor, HSIDataset
from encoder.hsi_bert_encoder import HSIEncoderBERT, UniversalHSIModel
from sspq_tokenizer.joint_tokenizer import SSPQTokenizer
from utils.config import load_config
from utils.logging import setup_logging
from utils.checkpoint import load_checkpoint, save_checkpoint


def load_tokenizer_from_ckpt(tokenizer_ckpt: str, num_bands: int) -> SSPQTokenizer:
    state = load_checkpoint(tokenizer_ckpt, map_location="cpu")
    tok_cfg = state.get("config")
    if tok_cfg is None:
        raise ValueError("Tokenizer checkpoint missing config; cannot rebuild model.")
    spectral_cfg = {
        "num_bands": num_bands,
        "hidden_dim": tok_cfg["hidden_dim"],
        "codebook_size": tok_cfg["K_s"],
        "num_layers": tok_cfg["num_layers"],
        "kernel_size": tok_cfg["kernel_size"],
        "commitment_beta": tok_cfg["commitment_beta"],
    }
    spatial_cfg = {
        "num_bands": num_bands,
        "hidden_dim": tok_cfg["hidden_dim"],
        "codebook_size": tok_cfg["K_x"],
        "encoder_channels": tok_cfg.get("encoder_channels", [64, 128, 256]),
        "commitment_beta": tok_cfg["commitment_beta"],
    }
    tokenizer = SSPQTokenizer(spectral_cfg, spatial_cfg)
    tokenizer.load_state_dict(state["model"])
    return tokenizer


def pretrain_universal_hsi(config_path: str, tokenizer_override: Optional[str] = None) -> None:
    """Load config, freeze tokenizer, and pretrain the HSI encoder."""
    config = load_config(config_path)
    logger = setup_logging()
    torch.manual_seed(int(config.get("seed", 42)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    extractor = HSIPatchExtractor(config.get("patch_size", 11), stride=config.get("stride"))
    datasets = []
    for ds in config["pretrain_datasets"]:
        datasets.append(
            HSIDataset(
                data_path=ds["data_path"],
                extractor=extractor,
                normalize="minmax",
                augment=True,
                domain_id=ds.get("domain_id", 0),
            )
        )
    num_bands = datasets[0].cube.shape[2]
    combined = ConcatDataset(datasets)
    dataloader = DataLoader(
        combined,
        batch_size=config["pretrain_batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        drop_last=False,
    )

    tokenizer_ckpt = tokenizer_override or config.get("tokenizer_ckpt")
    if tokenizer_ckpt is None:
        raise ValueError("tokenizer_ckpt must be provided in pretrain config.")
    tokenizer = load_tokenizer_from_ckpt(tokenizer_ckpt, num_bands=num_bands).to(device)
    for p in tokenizer.parameters():
        p.requires_grad = False
    tokenizer.eval()

    encoder = HSIEncoderBERT(
        joint_vocab_size=tokenizer.joint_vocab_size,
        num_domains=config["num_domains"],
        bert_model_name_or_path=config["bert_model_name_or_path"],
        load_pretrained_lm=config.get("load_pretrained_lm", True),
        mask_ratio=config["mask_ratio"],
        max_seq_len=config["max_seq_len"],
    ).to(device)

    universal = UniversalHSIModel(tokenizer, encoder).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, universal.parameters()), lr=config["learning_rate_pretrain"]
    )
    scaler = GradScaler(enabled=device.type == "cuda")
    total_steps = len(dataloader) * config["pretrain_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.get("num_warmup_steps", 0),
        num_training_steps=total_steps,
    )

    ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints/encoder"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(config["pretrain_epochs"]):
        universal.train()
        running = 0.0
        for batch in tqdm(dataloader, desc=f"Pretrain {epoch+1}/{config['pretrain_epochs']}"):
            patches = batch["patch"].to(device)
            domain_ids = batch["domain_id"].to(device)
            optimizer.zero_grad()
            with autocast(enabled=device.type == "cuda"):
                out = universal.encode_patches(patches, domain_ids, pretrain=True)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * patches.size(0)

        epoch_loss = running / len(combined)
        logger.info(f"Epoch {epoch+1}: pretrain loss={epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_path = ckpt_dir / "universal_hsi_encoder.pth"
            save_checkpoint(
                {"encoder": encoder.state_dict(), "config": config, "tokenizer_ckpt": tokenizer_ckpt},
                str(ckpt_path),
            )
            logger.info(f"Saved encoder checkpoint to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain Universal HSI Encoder")
    parser.add_argument("--config", required=True, help="Path to configs/pretrain.yaml")
    parser.add_argument("--tokenizer_ckpt", required=False, help="Optional path to tokenizer checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pretrain_universal_hsi(args.config, tokenizer_override=args.tokenizer_ckpt)
