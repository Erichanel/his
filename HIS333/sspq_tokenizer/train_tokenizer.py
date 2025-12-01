"""Training entry for Stage 1 SSPQ Tokenizer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.datasets import HSIPatchExtractor, HSIDataset
from sspq_tokenizer.joint_tokenizer import SSPQTokenizer
from utils.config import load_config
from utils.logging import setup_logging
from utils.checkpoint import save_checkpoint


def build_tokenizer(config, num_bands: int) -> SSPQTokenizer:
    spectral_cfg = {
        "num_bands": num_bands,
        "hidden_dim": config["hidden_dim"],
        "codebook_size": config["K_s"],
        "num_layers": config["num_layers"],
        "kernel_size": config["kernel_size"],
        "commitment_beta": config["commitment_beta"],
    }
    spatial_cfg = {
        "num_bands": num_bands,
        "hidden_dim": config["hidden_dim"],
        "codebook_size": config["K_x"],
        "encoder_channels": config.get("encoder_channels", [64, 128, 256]),
        "commitment_beta": config["commitment_beta"],
    }
    return SSPQTokenizer(spectral_cfg, spatial_cfg)


def train_sspq_tokenizer(config_path: str) -> None:
    """Load config, build datasets/dataloaders, and train tokenizer."""
    config = load_config(config_path)
    logger = setup_logging()

    torch.manual_seed(int(config.get("seed", 42)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    extractor = HSIPatchExtractor(config["patch_size"], stride=config.get("stride"))
    dataset = HSIDataset(
        data_path=config["data_path"],
        extractor=extractor,
        normalize="minmax",
        augment=True,
        domain_id=0,
    )
    num_bands = dataset.cube.shape[2]

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        drop_last=False,
    )

    tokenizer = build_tokenizer(config, num_bands).to(device)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=config["lr"])
    scaler = GradScaler(enabled=device.type == "cuda")

    ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints/tokenizer"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        tokenizer.train()
        running = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            patches = batch["patch"].to(device)
            optimizer.zero_grad()
            with autocast(enabled=device.type == "cuda"):
                out = tokenizer(patches, return_recon=True)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * patches.size(0)

        epoch_loss = running / len(dataset)
        logger.info(f"Epoch {epoch+1}: loss={epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_path = ckpt_dir / "best.pth"
            save_checkpoint({"model": tokenizer.state_dict(), "config": config}, str(ckpt_path))
            logger.info(f"Saved checkpoint to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SSPQ Tokenizer")
    parser.add_argument("--config", required=True, help="Path to configs/tokenizer.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_sspq_tokenizer(args.config)
