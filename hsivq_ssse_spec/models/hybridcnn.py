from __future__ import annotations

import torch
from torch import nn


class HybridSN(nn.Module):
    def __init__(
        self,
        c_in: int,
        num_classes: int,
        fc_dim: int = 256,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if c_in <= 12:
            raise ValueError("c_in must be > 12 for HybridSN kernels")
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        depth = c_in - 12
        in_ch_2d = 32 * depth
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_ch_2d, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(64, fc_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")
        b, c, h, w = x.shape
        x = x.unsqueeze(1)  # [B, 1, C, H, W]
        x = self.conv3d(x)
        b, c3, d, h3, w3 = x.shape
        x = x.reshape(b, c3 * d, h3, w3)
        x = self.conv2d(x)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
