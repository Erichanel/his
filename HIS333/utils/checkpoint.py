"""Checkpoint save/load helpers."""

from __future__ import annotations

import torch


def save_checkpoint(state: dict, path: str) -> None:
    """Save model state dict to disk."""
    torch.save(state, path)


def load_checkpoint(path: str, map_location=None) -> dict:
    """Load model state dict from disk."""
    return torch.load(path, map_location=map_location)
