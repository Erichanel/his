"""CLI entrypoint for Stage 1 + Stage 2 workflows."""

from __future__ import annotations

import argparse

from sspq_tokenizer.train_tokenizer import train_sspq_tokenizer
from encoder.pretrain_mtp import pretrain_universal_hsi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SSPQ-CrossTimeNet-HSI Stage1/2")
    parser.add_argument(
        "--stage",
        choices=["tokenizer", "pretrain", "all"],
        required=True,
        help="Which stage to run.",
    )
    parser.add_argument("--tokenizer_config", help="Path to configs/tokenizer.yaml")
    parser.add_argument("--pretrain_config", help="Path to configs/pretrain.yaml")
    parser.add_argument("--tokenizer_ckpt", help="Optional tokenizer checkpoint for Stage2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"tokenizer", "all"}:
        if not args.tokenizer_config:
            raise SystemExit("tokenizer_config is required for Stage1.")
        train_sspq_tokenizer(args.tokenizer_config)
    if args.stage in {"pretrain", "all"}:
        if not args.pretrain_config:
            raise SystemExit("pretrain_config is required for Stage2.")
        pretrain_universal_hsi(args.pretrain_config, tokenizer_override=args.tokenizer_ckpt)


if __name__ == "__main__":
    main()
