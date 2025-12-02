"""CLI entrypoint for Stage 3 anomaly detection."""

from __future__ import annotations

import argparse

from anomaly.train_anomaly import train_anomaly_detector
from anomaly.inference import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 anomaly detection")
    parser.add_argument("--mode", choices=["train", "infer"], required=True, help="Train or inference.")
    parser.add_argument("--config", required=True, help="Path to configs/anomaly.yaml")
    parser.add_argument("--encoder_ckpt", help="Optional encoder checkpoint for training")
    parser.add_argument("--detector_ckpt", help="Optional detector checkpoint for inference")
    parser.add_argument("--data_path", help="Optional data path for inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train_anomaly_detector(args.config, encoder_override=args.encoder_ckpt)
    else:
        run_inference(args.config, detector_ckpt=args.detector_ckpt, data_path=args.data_path)


if __name__ == "__main__":
    main()
