"""YAML + CLI config loader."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml


def _coerce_value(value):
    """Recursively convert numeric-like strings to int/float to avoid type issues."""
    if isinstance(value, dict):
        return {k: _coerce_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_value(v) for v in value]
    if isinstance(value, str):
        # try int
        try:
            if value.strip().lstrip("+-").isdigit():
                return int(value)
            return float(value)
        except ValueError:
            return value
    return value


def load_config(default_path: str, cli_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load YAML config and allow CLI overrides."""
    path = Path(default_path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config = _coerce_value(config)

    if cli_args:
        parser = argparse.ArgumentParser(add_help=False)
        for key in config.keys():
            parser.add_argument(f"--{key}")
        parsed, _ = parser.parse_known_args(cli_args)
        for key, value in vars(parsed).items():
            if value is not None:
                config[key] = value
    return config
