"""Stage 1: train spectral + spatial VQ-VAEs for hyperspectral cubes."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from hsi_data import HSICubeDataset
from hsi_model import SpatialCNNVQVAE, SpectralTCNVQVAE
from hsi_params import TokenizerConfig, load_hsi_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hyperspectral tokenizers")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config overriding defaults")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use when multiple splits exist")
    return parser.parse_args()


def train_epoch(loader: DataLoader, spectral: nn.Module, spatial: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:
    spectral.train()
    spatial.train()
    total_loss = 0.0
    mse = nn.MSELoss()

    for cubes, _ in loader:
        cubes = cubes.to(device)
        optimizer.zero_grad()

        spec_recon, spec_diff, _ = spectral(cubes)
        spa_recon, spa_diff, _ = spatial(cubes)

        spec_loss = mse(spec_recon, cubes) + spec_diff
        spa_loss = mse(spa_recon, cubes) + spa_diff
        loss = spec_loss + spa_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * cubes.size(0)

    return total_loss / len(loader.dataset)


def main() -> None:
    args = parse_args()
    cfg: TokenizerConfig = load_hsi_config(args.config)["tokenizer"]
    device = torch.device(cfg.device)

    dataset = HSICubeDataset(cfg.data_path, split=args.split)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)

    spectral = SpectralTCNVQVAE(
        num_bands=cfg.num_bands,
        hidden_dim=cfg.spectral_hidden,
        n_embed=cfg.spectral_codebook,
        wave_length=cfg.spectral_wave_length,
        block_num=cfg.spectral_block_num,
    ).to(device)
    spatial = SpatialCNNVQVAE(
        num_bands=cfg.num_bands,
        hidden_dim=cfg.spatial_hidden,
        n_embed=cfg.spatial_codebook,
        patch_size=cfg.spatial_patch_size,
    ).to(device)

    optimizer = optim.Adam(
        list(spectral.parameters()) + list(spatial.parameters()), lr=cfg.lr
    )

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.num_epochs + 1):
        loss = train_epoch(loader, spectral, spatial, optimizer, device)
        print(f"[Tokenizer] Epoch {epoch}: loss={loss:.4f}")

        torch.save(spectral.state_dict(), save_dir / "spectral.pt")
        torch.save(spatial.state_dict(), save_dir / "spatial.pt")

    print(f"Tokenizers saved to {save_dir}")


if __name__ == "__main__":
    main()
