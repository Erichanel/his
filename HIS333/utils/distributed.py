"""Distributed and DDP utilities (optional)."""

from __future__ import annotations

import torch
import torch.distributed as dist
import os


def init_distributed() -> bool:
    """Initialize torch.distributed if environment is set up."""
    if dist.is_available() and dist.is_initialized():
        return True
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        return True
    return False
