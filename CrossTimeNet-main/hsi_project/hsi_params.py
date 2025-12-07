"""Centralized hyperparameter definitions for the hyperspectral pipeline.

Each stage (tokenizer training, BERT pretraining, downstream fine-tuning)
can load a JSON configuration file where keys match the dataclass names
(`"tokenizer"`, `"pretrain"`, `"finetune"`). Values override the defaults.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TokenizerConfig:
    data_path: str = "./hsi_train.pt"
    save_dir: str = "./hsi_tokenizers"
    batch_size: int = 8
    num_workers: int = 2
    num_epochs: int = 10
    lr: float = 1e-3
    num_bands: int = 128
    spectral_hidden: int = 64
    spectral_codebook: int = 512
    spectral_wave_length: int = 8
    spectral_block_num: int = 4
    spatial_hidden: int = 64
    spatial_codebook: int = 512
    spatial_patch_size: int = 4
    device: str = "cuda" if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available() else "cpu"


@dataclass
class PretrainConfig:
    data_path: str = "./hsi_train.pt"
    tokenizer_ckpt: str = "./hsi_tokenizers"
    save_dir: str = "./hsi_pretrain"
    batch_size: int = 4
    num_workers: int = 2
    num_epochs: int = 5
    lr: float = 2e-4
    num_bands: int = 128
    spectral_codebook: int = 512
    spatial_codebook: int = 512
    mask_ratio: float = 0.3
    bert_model: str = "bert-base-uncased"
    device: str = "cuda" if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available() else "cpu"


@dataclass
class FinetuneConfig:
    data_path: str = "./hsi_train.pt"
    pretrained_ckpt: str = "./hsi_pretrain/model.pt"
    save_dir: str = "./hsi_finetune"
    batch_size: int = 4
    num_workers: int = 2
    num_epochs: int = 5
    lr: float = 1e-4
    num_bands: int = 128
    num_classes: int = 10
    spectral_codebook: int = 512
    spatial_codebook: int = 512
    pooling: str = "mean"
    device: str = "cuda" if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available() else "cpu"


def _merge(dataclass_cls: Any, overrides: Optional[Dict[str, Any]]) -> Any:
    if overrides is None:
        return dataclass_cls()
    base = dataclasses.asdict(dataclass_cls())
    base.update(overrides)
    return dataclass_cls(**base)


def load_hsi_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load a JSON config file and merge with defaults.

    The JSON structure should be::

        {
          "tokenizer": {"batch_size": 16, ...},
          "pretrain": {"mask_ratio": 0.4, ...},
          "finetune": {"num_epochs": 20, ...}
        }
    """

    if config_path is None:
        return {
            "tokenizer": TokenizerConfig(),
            "pretrain": PretrainConfig(),
            "finetune": FinetuneConfig(),
        }

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r") as f:
        raw = json.load(f)

    return {
        "tokenizer": _merge(TokenizerConfig, raw.get("tokenizer")),
        "pretrain": _merge(PretrainConfig, raw.get("pretrain")),
        "finetune": _merge(FinetuneConfig, raw.get("finetune")),
    }


__all__ = ["TokenizerConfig", "PretrainConfig", "FinetuneConfig", "load_hsi_config"]
