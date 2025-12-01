"""Lightweight logging setup helper."""

from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with a simple format if not already configured."""
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s - %(message)s")
    return logging.getLogger(__name__)
