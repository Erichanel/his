"""Inference entry for generating anomaly maps."""

from __future__ import annotations

from typing import Optional
import argparse
from pathlib import Path

import numpy as np
import torch

from anomaly.train_anomaly import load_tokenizer_from_ckpt
from anomaly.detector import HSIAnomalyDetector
from encoder.hsi_bert_encoder import HSIEncoderBERT, UniversalHSIModel
from utils.config import load_config
from utils.checkpoint import load_checkpoint
from utils.logging import setup_logging
from utils.metrics import compute_auc


def load_detector(detector_ckpt: str, config: dict, num_bands: int) -> HSIAnomalyDetector:
    state = load_checkpoint(detector_ckpt, map_location="cpu")
    det_cfg = state.get("config", config)
    encoder_ckpt = det_cfg.get("encoder_ckpt") or config.get("encoder_ckpt")
    if encoder_ckpt is None:
        raise ValueError("encoder_ckpt must be specified to load detector.")
    enc_state = load_checkpoint(encoder_ckpt, map_location="cpu")
    tokenizer_ckpt = enc_state.get("tokenizer_ckpt") or det_cfg.get("tokenizer_ckpt")
    tokenizer = load_tokenizer_from_ckpt(tokenizer_ckpt, num_bands=num_bands)
    pre_cfg = enc_state.get("config", {})
    encoder = HSIEncoderBERT(
        joint_vocab_size=tokenizer.joint_vocab_size,
        num_domains=pre_cfg.get("num_domains", det_cfg.get("num_domains", 1)),
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
        freeze_encoder=det_cfg.get("freeze_encoder", True),
        use_lora=det_cfg.get("use_lora", False),
        head_weights=det_cfg.get(
            "head_weights", {"lm_nll": 0.4, "recon": 0.3, "one_class": 0.3}
        ),
        patch_size=det_cfg.get("patch_size", 11),
        stride=det_cfg.get("stride"),
    )
    detector.load_state_dict(state["detector"])
    return detector


def run_inference(config_path: str, detector_ckpt: Optional[str] = None, data_path: Optional[str] = None) -> None:
    """Load trained detector, run inference on target scene, and save outputs."""
    config = load_config(config_path)
    logger = setup_logging()
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    detector_path = detector_ckpt or config.get("detector_ckpt")
    if detector_path is None:
        raise ValueError("detector_ckpt must be provided for inference.")
    data_path = data_path or config.get("target_data") or config.get("data_path")
    if data_path is None:
        raise ValueError("data_path/target_data must be provided for inference.")

    data = np.load(data_path)
    files = getattr(data, "files", [])
    cube = data["cube"] if "cube" in files else data[files[0]]
    gt = data["gt"] if "gt" in files else data["gt_mask"] if "gt_mask" in files else None
    num_bands = cube.shape[2]

    detector = load_detector(detector_path, config, num_bands=num_bands).to(device)
    detector.eval()

    with torch.no_grad():
        anomaly_map = detector.generate_anomaly_map(cube, domain_id=config.get("domain_id", 0))

    out_dir = Path(config.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "anomaly_map.npy"
    np.save(out_path, anomaly_map)
    logger.info(f"Saved anomaly map to {out_path}")

    if gt is not None:
        auc = compute_auc(anomaly_map.flatten(), gt.flatten())
        logger.info(f"AUC against provided gt: {auc:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HSI anomaly inference")
    parser.add_argument("--config", required=True, help="Path to configs/anomaly.yaml")
    parser.add_argument("--detector_ckpt", required=False, help="Path to anomaly detector checkpoint")
    parser.add_argument("--data_path", required=False, help="Target scene file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args.config, detector_ckpt=args.detector_ckpt, data_path=args.data_path)
