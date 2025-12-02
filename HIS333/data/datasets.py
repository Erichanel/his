"""Datasets and patch extraction utilities for SSPQ-CrossTimeNet-HSI."""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import torch

from .transforms import normalize_cube, augment_cube


class HSIPatchExtractor:
    """Extract overlapping patches and coordinates as described in the requirements."""

    def __init__(self, patch_size: int, stride: Optional[int] = None):
        self.patch_size = patch_size
        self.stride = stride or patch_size // 2

    def extract_patches(self, cube: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return patches shaped [N, B, S, S] and coords shaped [N, 2]."""
        H, W, B = cube.shape
        S = self.patch_size
        stride = self.stride
        patches = []
        coords = []
        r_positions = list(range(0, max(H - S + 1, 1), stride))
        if r_positions[-1] != max(H - S, 0):
            r_positions.append(max(H - S, 0))
        c_positions = list(range(0, max(W - S + 1, 1), stride))
        if c_positions[-1] != max(W - S, 0):
            c_positions.append(max(W - S, 0))

        for r in r_positions:
            for c in c_positions:
                patch = cube[r : r + S, c : c + S, :]  # (S, S, B)
                patch = np.transpose(patch, (2, 0, 1))  # (B, S, S)
                patches.append(patch.astype(np.float32))
                coords.append((r, c))
        patches_t = torch.from_numpy(np.stack(patches)) if patches else torch.empty(0)
        coords_t = torch.tensor(coords, dtype=torch.long) if coords else torch.empty((0, 2), dtype=torch.long)
        return patches_t, coords_t


class HSIDataset(torch.utils.data.Dataset):
    """Dataset for loading cubes and serving patch tensors."""

    def __init__(
        self,
        data_path: str,
        extractor: HSIPatchExtractor,
        normalize: str = "minmax",
        augment: bool = False,
        domain_id: Optional[int] = None,
    ):
        self.data_path = data_path
        self.extractor = extractor
        self.normalize = normalize
        self.augment = augment
        self.domain_id = domain_id if domain_id is not None else 0

        data = np.load(self.data_path)
        files = getattr(data, "files", [])
        self.cube = data["cube"] if "cube" in files else data[files[0]]
        if "gt" in files:
            self.gt_mask = data["gt"]
        elif "gt_mask" in files:
            self.gt_mask = data["gt_mask"]
        else:
            self.gt_mask = None
        self.cube = normalize_cube(self.cube, method=self.normalize)

        self.patches, self.coords = self.extractor.extract_patches(self.cube)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        patch = self.patches[idx].clone()
        coord = self.coords[idx]
        if self.augment:
            patch_np = patch.numpy().transpose(1, 2, 0)
            patch_np = augment_cube(patch_np)
            patch_np = np.ascontiguousarray(patch_np)  # ensure positive strides after flips/rotations
            patch = torch.from_numpy(np.transpose(patch_np, (2, 0, 1))).float()

        if self.gt_mask is not None:
            S = self.extractor.patch_size
            r, c = coord.tolist()
            submask = self.gt_mask[r : r + S, c : c + S]
            label = 1 if np.mean(submask) > 0 else 0
        else:
            label = -1

        sample = {
            "patch": patch,
            "coord": coord,
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }
        return sample
