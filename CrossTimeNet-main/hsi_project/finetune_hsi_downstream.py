"""Stage 3: fine-tune pretrained HSIBert on downstream labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from hsi_data import HSICubeDataset
from hsi_model import HSIBert
from hsi_params import FinetuneConfig, load_hsi_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune HSIBert on labeled hyperspectral data")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config overriding defaults")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    return parser.parse_args()


def run_epoch(loader: DataLoader, model: HSIBert, optimizer: optim.Optimizer, device: torch.device, train: bool) -> float:
    criterion = nn.CrossEntropyLoss()
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    with torch.set_grad_enabled(train):
        for cubes, labels in loader:
            if labels is None:
                raise ValueError("Labeled data is required for fine-tuning")
            cubes = cubes.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(cubes, targets=labels, pretrain=False)
            loss = criterion(outputs["logits"], labels)
            preds = outputs["logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy


def main() -> None:
    args = parse_args()
    cfg: FinetuneConfig = load_hsi_config(args.config)["finetune"]
    device = torch.device(cfg.device)

    train_ds = HSICubeDataset(cfg.data_path, split=args.train_split)
    val_ds = HSICubeDataset(cfg.data_path, split=args.val_split)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    model = HSIBert(
        num_bands=cfg.num_bands,
        num_classes=cfg.num_classes,
        spectral_codebook=cfg.spectral_codebook,
        spatial_codebook=cfg.spatial_codebook,
        pooling=cfg.pooling,
    ).to(device)

    ckpt_path = Path(cfg.pretrained_ckpt)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        raise FileNotFoundError(f"Pretrained checkpoint not found: {cfg.pretrained_ckpt}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, model, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(val_loader, model, optimizer, device, train=False)

        print(
            f"[Finetune] Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_dir / "best.pt")

    print(f"Best checkpoint saved to {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
