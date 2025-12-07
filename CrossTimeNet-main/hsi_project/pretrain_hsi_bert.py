"""Stage 2: BERT masked modeling on tokenizer outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader

from hsi_data import HSICubeDataset
from hsi_model import HSIBert
from hsi_params import PretrainConfig, load_hsi_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train HSIBert with masked token modeling")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config overriding defaults")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use when multiple splits exist")
    parser.add_argument("--freeze_tokenizers", action="store_true", help="Freeze tokenizer parameters during pretraining")
    return parser.parse_args()


def pretrain_epoch(loader: DataLoader, model: HSIBert, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for cubes, _ in loader:
        cubes = cubes.to(device)
        optimizer.zero_grad()
        outputs = model(cubes, pretrain=True)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * cubes.size(0)
    return total_loss / len(loader.dataset)


def main() -> None:
    args = parse_args()
    cfg: PretrainConfig = load_hsi_config(args.config)["pretrain"]
    device = torch.device(cfg.device)

    dataset = HSICubeDataset(cfg.data_path, split=args.split)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)

    model = HSIBert(
        num_bands=cfg.num_bands,
        num_classes=cfg.spectral_codebook,  # placeholder; heads unused during pretrain
        spectral_codebook=cfg.spectral_codebook,
        spatial_codebook=cfg.spatial_codebook,
        mask_ratio=cfg.mask_ratio,
        bert_model=cfg.bert_model,
    ).to(device)

    tokenizer_dir = Path(cfg.tokenizer_ckpt)
    if tokenizer_dir.exists():
        spec_path = tokenizer_dir / "spectral.pt"
        spa_path = tokenizer_dir / "spatial.pt"
        if spec_path.exists():
            model.spectral_tokenizer.load_state_dict(torch.load(spec_path, map_location=device))
        if spa_path.exists():
            model.spatial_tokenizer.load_state_dict(torch.load(spa_path, map_location=device))
    else:
        raise FileNotFoundError(f"Tokenizer checkpoint directory not found: {cfg.tokenizer_ckpt}")

    if args.freeze_tokenizers:
        for p in model.spectral_tokenizer.parameters():
            p.requires_grad = False
        for p in model.spatial_tokenizer.parameters():
            p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.num_epochs + 1):
        loss = pretrain_epoch(loader, model, optimizer, device)
        print(f"[Pretrain] Epoch {epoch}: loss={loss:.4f}")
        torch.save(model.state_dict(), save_dir / "model.pt")

    print(f"Pretrained HSIBert saved to {save_dir}")


if __name__ == "__main__":
    main()
