from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class HSIDatasetInfo:
    cube: np.ndarray
    gt: np.ndarray | None


def _load_mat(path: Path, key: str | None = None) -> np.ndarray:
    import scipy.io as sio

    mat = sio.loadmat(path)
    if key is None:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if not keys:
            raise ValueError(f"No valid keys in {path}")
        key = keys[0]
    if key not in mat:
        raise KeyError(f"Key {key} not in {path}")
    return mat[key]


def load_hsi_from_paths(
    data_path: str | Path,
    gt_path: str | Path | None = None,
    data_key: str | None = None,
    gt_key: str | None = None,
) -> HSIDatasetInfo:
    data_path = Path(data_path)
    cube = _load_mat(data_path, key=data_key)
    gt = None
    if gt_path is not None:
        gt_path = Path(gt_path)
        gt = _load_mat(gt_path, key=gt_key)
    return HSIDatasetInfo(cube=cube, gt=gt)


def load_whu_hsi(dataset_name: str, root: str | Path = ".") -> HSIDatasetInfo:
    root = Path(root)
    if dataset_name == "WHU_Hi_LongKou":
        cube = _load_mat(root / "WHU_Hi_LongKou.mat", key="WHU_Hi_LongKou")
        gt = _load_mat(root / "WHU_Hi_LongKou_gt.mat", key="WHU_Hi_LongKou_gt")
    elif dataset_name == "WHU_Hi_HanChuan_270":
        cube = _load_mat(root / "WHU_Hi_HanChuan_270.mat", key="cube")
        gt = _load_mat(root / "WHU_Hi_HanChuan_gt.mat", key="WHU_Hi_HanChuan_gt")
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    return HSIDatasetInfo(cube=cube, gt=gt)


def resolve_dataset_name(root: str | Path = ".") -> str | None:
    root = Path(root)
    if (root / "WHU_Hi_LongKou.mat").exists():
        return "WHU_Hi_LongKou"
    if (root / "WHU_Hi_HanChuan_270.mat").exists():
        return "WHU_Hi_HanChuan_270"
    return None


def split_labeled_indices(
    gt: np.ndarray,
    train_patches: int,
    val_patches: int,
    seed: int = 42,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    rng = np.random.RandomState(seed)
    labels = np.unique(gt)
    labels = labels[labels > 0]
    train_idx: List[Tuple[int, int]] = []
    val_idx: List[Tuple[int, int]] = []
    test_idx: List[Tuple[int, int]] = []

    for label in labels:
        coords = np.argwhere(gt == label)
        if coords.shape[0] < train_patches + val_patches:
            raise ValueError(
                f"Label {label} has {coords.shape[0]} samples, "
                f"need {train_patches + val_patches}"
            )
        perm = rng.permutation(coords.shape[0])
        coords = coords[perm]
        train_idx.extend([tuple(x) for x in coords[:train_patches]])
        val_idx.extend([tuple(x) for x in coords[train_patches:train_patches + val_patches]])
        test_idx.extend([tuple(x) for x in coords[train_patches + val_patches:]])

    return train_idx, val_idx, test_idx


class HSIPatchDataset(Dataset):
    def __init__(
        self,
        cube: np.ndarray,
        indices: List[Tuple[int, int]],
        patch_size: int = 9,
        labels: np.ndarray | None = None,
    ) -> None:
        if cube.ndim != 3:
            raise ValueError(f"Expected cube [H, W, C], got {cube.shape}")
        self.cube = cube.astype(np.float32)
        self.labels = labels
        self.indices = indices
        self.patch_size = patch_size
        pad = patch_size // 2
        self.pad = pad
        self.cube_pad = np.pad(
            self.cube,
            ((pad, pad), (pad, pad), (0, 0)),
            mode="reflect",
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        r, c = self.indices[idx]
        r0 = r + self.pad
        c0 = c + self.pad
        patch = self.cube_pad[
            r0 - self.pad : r0 + self.pad + 1,
            c0 - self.pad : c0 + self.pad + 1,
            :,
        ]
        patch = torch.from_numpy(patch).permute(2, 0, 1)
        if self.labels is None:
            return patch
        label = int(self.labels[r, c])
        return patch, label


def build_pretrain_dataset(
    cube: np.ndarray,
    patch_size: int,
    stride: int = 1,
) -> HSIPatchDataset:
    if stride <= 0:
        raise ValueError("stride must be >= 1")
    h, w, _ = cube.shape
    indices = [(r, c) for r in range(0, h, stride) for c in range(0, w, stride)]
    return HSIPatchDataset(cube, indices, patch_size=patch_size, labels=None)


def build_fewshot_splits(
    cube: np.ndarray,
    gt: np.ndarray,
    patch_size: int,
    train_patches: int,
    val_patches: int,
    seed: int = 42,
):
    train_idx, val_idx, test_idx = split_labeled_indices(
        gt, train_patches=train_patches, val_patches=val_patches, seed=seed
    )
    train_set = HSIPatchDataset(cube, train_idx, patch_size=patch_size, labels=gt)
    val_set = HSIPatchDataset(cube, val_idx, patch_size=patch_size, labels=gt)
    test_set = HSIPatchDataset(cube, test_idx, patch_size=patch_size, labels=gt)
    return train_set, val_set, test_set
