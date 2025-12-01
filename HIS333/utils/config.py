"""YAML + CLI config loader."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml


def load_config(default_path: str, cli_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load YAML config and allow CLI overrides."""
    path = Path(default_path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if cli_args:
        parser = argparse.ArgumentParser(add_help=False)
        for key in config.keys():
            parser.add_argument(f"--{key}")
        parsed, _ = parser.parse_known_args(cli_args)
        for key, value in vars(parsed).items():
            if value is not None:
                config[key] = value
    return config
