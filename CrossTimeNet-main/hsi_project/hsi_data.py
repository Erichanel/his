"""Utilities for loading hyperspectral cubes for the three-stage pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset


class HSICubeDataset(Dataset):
    """Load hyperspectral cubes stored in a torch serialized file.

    Expected formats::

        # single split
        torch.save({"data": Tensor[N, C, H, W], "labels": Tensor[N]}, "hsi.pt")

        # multi-split
        torch.save({
            "train": {"data": Tensor, "labels": Tensor},
            "val": {"data": Tensor, "labels": Tensor},
            "test": {"data": Tensor, "labels": Tensor},
        }, "hsi.pt")

    Args:
        data_path: Path to the serialized torch file.
        split: Which split to read when multiple splits are present.
        normalize: If True, per-channel mean/std normalization is applied.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        normalize: bool = True,
    ):
        super().__init__()
        tensor_path = Path(data_path)
        if not tensor_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        raw = torch.load(tensor_path)
        if "data" in raw:
            data = raw["data"]
            labels = raw.get("labels")
        else:
            if split not in raw:
                raise KeyError(f"Split '{split}' not found in dataset file. Available keys: {list(raw.keys())}")
            split_content: Dict[str, torch.Tensor] = raw[split]
            data = split_content["data"]
            labels = split_content.get("labels")

        if normalize:
            data = self._normalize(data)

        self.data = data.float()
        self.labels = labels.long() if labels is not None else None

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected tensor of shape (N, C, H, W), got {tuple(x.shape)}")
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        std = x.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
        return (x - mean) / std

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.labels is None:
            return self.data[idx], None
        return self.data[idx], self.labels[idx]


__all__ = ["HSICubeDataset"]
