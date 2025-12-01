"""Normalization and augmentation helpers for hyperspectral data."""

from __future__ import annotations

import numpy as np


def normalize_cube(cube: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize hyperspectral cube with minmax or zscore strategies."""
    if method not in {"minmax", "zscore"}:
        raise ValueError(f"Unknown normalization method: {method}")
    cube = cube.astype(np.float32)
    if method == "minmax":
        band_min = cube.min(axis=(0, 1), keepdims=True)
        band_max = cube.max(axis=(0, 1), keepdims=True)
        cube = (cube - band_min) / (band_max - band_min + 1e-6)
    else:  # zscore
        mean = cube.mean(axis=(0, 1), keepdims=True)
        std = cube.std(axis=(0, 1), keepdims=True)
        cube = (cube - mean) / (std + 1e-6)
    return cube


def augment_cube(cube: np.ndarray) -> np.ndarray:
    """
    Apply simple augmentations: random 90-degree rotations, flips, and Gaussian noise.
    Augmentations operate on (H, W, B) numpy arrays.
    """
    # Random rotation (0, 90, 180, 270)
    k = np.random.randint(0, 4)
    if k:
        cube = np.rot90(cube, k=k, axes=(0, 1))
    # Random horizontal/vertical flips
    if np.random.rand() < 0.5:
        cube = np.flip(cube, axis=0)
    if np.random.rand() < 0.5:
        cube = np.flip(cube, axis=1)
    # Add mild Gaussian noise
    if np.random.rand() < 0.5:
        noise = np.random.normal(loc=0.0, scale=0.01, size=cube.shape).astype(np.float32)
        cube = cube + noise
    return cube
