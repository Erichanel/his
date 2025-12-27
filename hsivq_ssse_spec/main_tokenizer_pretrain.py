from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import load_yaml
from utils.cube_partition import make_cubes
from utils.gwpca import apply_gwpca_with_state, fit_gwpca
from utils.dataset import build_pretrain_dataset, load_hsi_from_paths, load_whu_hsi, resolve_dataset_name
from utils.normalize import apply_zscore, fit_zscore_from_tensor
from sspq_tokenizer.tokenizer import SSPQTokenizer


def _get(cfg, *keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_tokenizer(cfg):
    return SSPQTokenizer(
        D_vq=_get(cfg, "vq", "D_vq", default=256),
        Ks=_get(cfg, "vq", "Ks", default=64),
        Kx=_get(cfg, "vq", "Kx", default=64),
        vq_decay=_get(cfg, "vq", "vq_decay", default=0.99),
        vq_beta=_get(cfg, "vq", "vq_beta", default=0.25),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tokenizer_hsivq_ssse.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    torch.manual_seed(_get(cfg, "train", "seed", default=42))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, please enable GPU or install CUDA build.")
    device = torch.device("cuda")

    tokenizer = build_tokenizer(cfg).to(device)
    tokenizer.train()

    b = _get(cfg, "train", "batch_size", default=8)
    patch_size = _get(cfg, "data", "patch_size", default=9)
    pretrain_stride = _get(cfg, "data", "pretrain_stride", default=3)
    pca_group = _get(cfg, "data", "pca_group", default=8)
    pca_nc = _get(cfg, "data", "pca_nc_per_group", default=8)
    pca_max_samples = _get(cfg, "data", "pca_max_samples", default=None)
    use_zscore = _get(cfg, "data", "use_zscore", default=True)
    zscore_max_samples = _get(cfg, "data", "zscore_max_samples", default=None)

    num_workers = _get(cfg, "train", "num_workers", default=0)
    pin_memory = _get(cfg, "train", "pin_memory", default=True)
    prefetch_factor = _get(cfg, "train", "prefetch_factor", default=2)
    persistent_workers = _get(cfg, "train", "persistent_workers", default=False)

    data_path = _get(cfg, "data", "data_path", default=None)
    gt_path = _get(cfg, "data", "gt_path", default=None)
    data_key = _get(cfg, "data", "data_key", default=None)
    gt_key = _get(cfg, "data", "gt_key", default=None)
    dataset_name = _get(cfg, "data", "dataset_name", default=None) or resolve_dataset_name()
    if data_path is not None:
        data = load_hsi_from_paths(data_path, gt_path, data_key=data_key, gt_key=gt_key)
    elif dataset_name is not None:
        data = load_whu_hsi(dataset_name)
    else:
        data = None

    if data is None:
        raise RuntimeError("No dataset found for tokenizer pretrain.")

    gwpca_state = fit_gwpca(
        data.cube,
        group=pca_group,
        nc_per_group=pca_nc,
        whiten=True,
        max_samples=pca_max_samples,
    )
    cube_feat = apply_gwpca_with_state(data.cube, gwpca_state)
    cube_feat = torch.as_tensor(cube_feat)
    if use_zscore:
        z_mean, z_std = fit_zscore_from_tensor(cube_feat, max_samples=zscore_max_samples)
        cube_feat = apply_zscore(cube_feat, z_mean, z_std)
    cube_feat = cube_feat.squeeze(0).permute(1, 2, 0).cpu().numpy()

    dataset = build_pretrain_dataset(cube_feat, patch_size=patch_size, stride=pretrain_stride)
    loader_args = {
        "batch_size": b,
        "shuffle": True,
        "drop_last": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }
    if num_workers > 0:
        loader_args["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_args)

    epochs = _get(cfg, "train", "epochs", default=10)
    lr = _get(cfg, "train", "lr", default=0.0005)
    wd = _get(cfg, "train", "wd", default=0.0)
    steps_per_epoch = _get(cfg, "train", "steps_per_epoch", default=None)
    log_interval = _get(cfg, "train", "log_interval", default=None)
    print_shapes = _get(cfg, "train", "print_shapes", default=False)
    debug_update = _get(cfg, "train", "debug_update", default=False)
    debug_interval = _get(cfg, "train", "debug_interval", default=200)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=lr, weight_decay=wd)
    lambda_recon_s = _get(cfg, "train", "lambda_recon_s", default=1.0)
    lambda_recon_x = _get(cfg, "train", "lambda_recon_x", default=1.0)

    printed_shapes = False
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_recon_s = 0.0
        total_recon_x = 0.0
        total_count = 0
        k_set = set()
        m_set = set()
        max_steps = steps_per_epoch or len(loader)
        pbar = tqdm(loader, total=max_steps, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for step, x in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            cubes, _ = make_cubes(
                x,
                patch_size=patch_size,
                cube_size=_get(cfg, "data", "cube_size", default=3),
                T=_get(cfg, "data", "T", default=8),
            )
            optimizer.zero_grad()
            tok_out = tokenizer(cubes, return_recon=True)
            vq_loss = tok_out["total"]
            recon_loss_s = tok_out["recon_loss_s"]
            recon_loss_x = tok_out["recon_loss_x"]
            loss = vq_loss + lambda_recon_s * recon_loss_s + lambda_recon_x * recon_loss_x
            
            # CRITICAL FIX: Check for NaN/Inf in loss before backprop
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss is NaN/Inf at step {step}: vq_loss={vq_loss}, recon_s={recon_loss_s}, recon_x={recon_loss_x}")
            
            do_debug = debug_update and debug_interval and (step % int(debug_interval) == 0)
            if do_debug:
                w_prev = {
                    "s_conv1": tokenizer.s_conv1.weight.detach().clone(),
                    "x_conv1": tokenizer.x_conv1.weight.detach().clone(),
                    "s_fc": tokenizer.s_fc.weight.detach().clone(),
                    "x_fc": tokenizer.x_fc.weight.detach().clone(),
                    "vq_s_embed": tokenizer.vq_s.embedding.weight.detach().clone(),
                    "vq_x_embed": tokenizer.vq_x.embedding.weight.detach().clone(),
                }
            loss.backward()
            
            # CRITICAL FIX: Gradient anomaly detection
            gradient_norm = 0.0
            has_zero_grad = False
            for name, param in tokenizer.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    gradient_norm += grad_norm ** 2
                    if grad_norm == 0.0:
                        has_zero_grad = True
                    # Check for gradient explosion/NaN
                    if not torch.isfinite(param.grad).all():
                        raise RuntimeError(f"Non-finite gradient in {name} at step {step}")
            gradient_norm = gradient_norm ** 0.5
            
            if do_debug:
                grad_s = tokenizer.s_conv1.weight.grad
                grad_x = tokenizer.x_conv1.weight.grad
                grad_s_fc = tokenizer.s_fc.weight.grad
                grad_x_fc = tokenizer.x_fc.weight.grad
                grad_info = {
                    "grad_s": float(grad_s.norm().detach().cpu()) if grad_s is not None else 0.0,
                    "grad_x": float(grad_x.norm().detach().cpu()) if grad_x is not None else 0.0,
                    "grad_s_fc": float(grad_s_fc.norm().detach().cpu()) if grad_s_fc is not None else 0.0,
                    "grad_x_fc": float(grad_x_fc.norm().detach().cpu()) if grad_x_fc is not None else 0.0,
                }
            optimizer.step()
            if do_debug:
                with torch.no_grad():
                    delta_s = (tokenizer.s_conv1.weight - w_prev["s_conv1"]).abs().max().item()
                    delta_x = (tokenizer.x_conv1.weight - w_prev["x_conv1"]).abs().max().item()
                    delta_s_fc = (tokenizer.s_fc.weight - w_prev["s_fc"]).abs().max().item()
                    delta_x_fc = (tokenizer.x_fc.weight - w_prev["x_fc"]).abs().max().item()
                    delta_vq_s = (tokenizer.vq_s.embedding.weight - w_prev["vq_s_embed"]).abs().max().item()
                    delta_vq_x = (tokenizer.vq_x.embedding.weight - w_prev["vq_x_embed"]).abs().max().item()
                pbar.write(
                    f"debug step {step}: "
                    f"loss {float(loss.detach().cpu()):.3e} grad_norm {gradient_norm:.3e} "
                    f"grad_s {grad_info['grad_s']:.3e} grad_x {grad_info['grad_x']:.3e} "
                    f"grad_s_fc {grad_info['grad_s_fc']:.3e} grad_x_fc {grad_info['grad_x_fc']:.3e} "
                    f"delta_s {delta_s:.3e} delta_x {delta_x:.3e} "
                    f"delta_s_fc {delta_s_fc:.3e} delta_x_fc {delta_x_fc:.3e} "
                    f"delta_vq_s {delta_vq_s:.3e} delta_vq_x {delta_vq_x:.3e}"
                )

            bs = x.shape[0]
            total_loss += float(loss.detach().cpu()) * bs
            total_recon_s += float(recon_loss_s.detach().cpu()) * bs
            total_recon_x += float(recon_loss_x.detach().cpu()) * bs
            total_count += bs
            k_set.update(tok_out["k_idx"].detach().cpu().unique().tolist())
            m_set.update(tok_out["m_idx"].detach().cpu().unique().tolist())

            if print_shapes and not printed_shapes:
                print("cubes", tuple(cubes.shape))
                printed_shapes = True

            if steps_per_epoch is not None and step >= steps_per_epoch:
                break
            if log_interval and (step % log_interval == 0):
                avg_loss = total_loss / max(total_count, 1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

        avg_loss = total_loss / max(total_count, 1)
        avg_rs = total_recon_s / max(total_count, 1)
        avg_rx = total_recon_x / max(total_count, 1)
        ks = _get(cfg, "vq", "Ks", default=64)
        kx = _get(cfg, "vq", "Kx", default=64)
        usage_k = len(k_set)
        usage_m = len(m_set)
        print(
            f"epoch {epoch}/{epochs} loss {avg_loss:.4f} "
            f"recon_s {avg_rs:.4f} recon_x {avg_rx:.4f} "
            f"usage_k {usage_k}/{ks} usage_m {usage_m}/{kx}"
        )

    save_path = _get(cfg, "train", "save_path", default=None)
    if save_path:
        torch.save({"tokenizer": tokenizer.state_dict()}, save_path)
        print("saved", save_path)


if __name__ == "__main__":
    main()
