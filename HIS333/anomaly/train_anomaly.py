"""Training script for Stage 3 anomaly detector."""

from __future__ import annotations

from typing import Optional
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from anomaly.detector import HSIAnomalyDetector
from encoder.hsi_bert_encoder import HSIEncoderBERT, UniversalHSIModel
from sspq_tokenizer.joint_tokenizer import SSPQTokenizer
from data.datasets import HSIPatchExtractor, HSIDataset
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


def build_detector(config: dict, encoder_ckpt: str, num_bands: int) -> HSIAnomalyDetector:
    enc_state = load_checkpoint(encoder_ckpt, map_location="cpu")
    pre_cfg = enc_state.get("config", {})
    tokenizer_ckpt = enc_state.get("tokenizer_ckpt") or config.get("tokenizer_ckpt")
    if tokenizer_ckpt is None:
        raise ValueError("Tokenizer checkpoint must be provided via encoder ckpt or config.")
    tokenizer = load_tokenizer_from_ckpt(tokenizer_ckpt, num_bands=num_bands)
    encoder = HSIEncoderBERT(
        joint_vocab_size=tokenizer.joint_vocab_size,
        num_domains=pre_cfg.get("num_domains", config.get("num_domains", 1)),
        bert_model_name_or_path=pre_cfg.get("bert_model_name_or_path", "bert-base-uncased"),
        load_pretrained_lm=pre_cfg.get("load_pretrained_lm", True),
        mask_ratio=pre_cfg.get("mask_ratio", 0.35),
        max_seq_len=pre_cfg.get("max_seq_len", 256),
    )
    encoder.load_state_dict(enc_state["encoder"])
    universal = UniversalHSIModel(tokenizer, encoder)

    detector = HSIAnomalyDetector(
        universal_model=universal,
        hidden_size=encoder.encoder.config.hidden_size,
        freeze_encoder=config.get("freeze_encoder", True),
        use_lora=config.get("use_lora", False),
        head_weights=config.get(
            "head_weights", {"lm_nll": 0.4, "recon": 0.3, "one_class": 0.3}
        ),
        patch_size=config.get("patch_size", 11),
        stride=config.get("stride"),
    )
    return detector


def train_anomaly_detector(config_path: str, encoder_override: Optional[str] = None) -> None:
    """Load config, prepare datasets, and train anomaly heads."""
    config = load_config(config_path)
    if encoder_override:
        config["encoder_ckpt"] = encoder_override
    logger = setup_logging()
    torch.manual_seed(int(config.get("seed", 42)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    data_path = config.get("data_path") or config.get("target_data")
    if data_path is None:
        raise ValueError("data_path or target_data must be specified in anomaly config.")
    extractor = HSIPatchExtractor(config.get("patch_size", 11), stride=config.get("stride"))
    dataset = HSIDataset(
        data_path=data_path,
        extractor=extractor,
        normalize="minmax",
        augment=config.get("augment", False),
        domain_id=config.get("domain_id", 0),
    )
    num_bands = dataset.cube.shape[2]
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("anomaly_batch_size", 16),
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        drop_last=False,
    )

    encoder_ckpt = config.get("encoder_ckpt")
    if encoder_ckpt is None:
        raise ValueError("encoder_ckpt must be provided in anomaly config.")
    detector = build_detector(config, encoder_ckpt, num_bands=num_bands).to(device)

    params = [p for p in detector.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.get("learning_rate_anomaly", 1e-4))
    scaler = GradScaler(enabled=device.type == "cuda")
    bce = torch.nn.BCEWithLogitsLoss()

    ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints/anomaly"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(config.get("anomaly_epochs", 20)):
        detector.train()
        running = 0.0
        for batch in tqdm(dataloader, desc=f"Anomaly Train {epoch+1}/{config.get('anomaly_epochs', 20)}"):
            patches = batch["patch"].to(device)
            domain_ids = batch["domain_id"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()

            with autocast(enabled=device.type == "cuda"):
                out = detector(patches, domain_ids, mode="train")
                oc_scores = out["one_class"]
                if (labels >= 0).any():
                    lbl = labels.float()
                    loss = bce(oc_scores, lbl)
                else:
                    score_center = oc_scores.detach().mean()
                    loss = ((oc_scores - score_center) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * patches.size(0)

        epoch_loss = running / len(dataset)
        logger.info(f"Epoch {epoch+1}: anomaly loss={epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_path = ckpt_dir / "anomaly_detector.pth"
            save_checkpoint({"detector": detector.state_dict(), "config": config}, str(ckpt_path))
            logger.info(f"Saved anomaly detector to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HSI Anomaly Detector")
    parser.add_argument("--config", required=True, help="Path to configs/anomaly.yaml")
    parser.add_argument("--encoder_ckpt", required=False, help="Optional path to encoder checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_anomaly_detector(args.config, encoder_override=args.encoder_ckpt)
