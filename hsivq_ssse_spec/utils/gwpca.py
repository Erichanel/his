from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise TypeError("Expected numpy.ndarray or torch.Tensor")


def _reshape_to_samples(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        # [H, W, C]
        return x.reshape(-1, x.shape[-1])
    if x.ndim == 4:
        # [B, C, H, W] or [B, H, W, C]
        if x.shape[-1] != x.shape[1]:
            # [B, H, W, C]
            return x.reshape(-1, x.shape[-1])
        # [B, C, H, W]
        return x.transpose(0, 2, 3, 1).reshape(-1, x.shape[1])
    raise ValueError(f"Unsupported input shape: {x.shape}")


def fit_gwpca(
    x,
    group: int = 8,
    nc_per_group: int = 8,
    whiten: bool = True,
    max_samples: int | None = None,
    seed: int = 42,
):
    x_np = _to_numpy(x).astype(np.float32)
    samples = _reshape_to_samples(x_np)
    c = samples.shape[1]
    base = c // group
    rem = c % group
    group_sizes = [base + (1 if i < rem else 0) for i in range(group)]
    rng = np.random.RandomState(seed)
    eps = 1e-6

    means = []
    components = []
    scales = []
    c_start = 0
    for c_per_group in group_sizes:
        c0 = c_start
        c1 = c_start + c_per_group
        c_start = c1
        if c_per_group == 0:
            means.append(torch.zeros(0))
            components.append(torch.zeros(0, 0))
            scales.append(torch.zeros(0))
            continue
        Xg = samples[:, c0:c1]
        if max_samples is not None and Xg.shape[0] > max_samples:
            idx = rng.choice(Xg.shape[0], size=max_samples, replace=False)
            Xg = Xg[idx]
        mu = Xg.mean(axis=0, keepdims=True)
        Xc = Xg - mu
        u, s, vh = np.linalg.svd(Xc, full_matrices=False)
        k = min(nc_per_group, vh.shape[0])
        comps = vh[:k].T  # [Cg, k]
        means.append(torch.from_numpy(mu.squeeze(0)))
        components.append(torch.from_numpy(comps))
        if whiten and k > 0:
            scale = 1.0 / (s[:k] + eps)
        else:
            scale = np.ones((k,), dtype=np.float32)
        scales.append(torch.from_numpy(scale))

    return {
        "group": group,
        "nc_per_group": nc_per_group,
        "group_sizes": group_sizes,
        "means": means,
        "components": components,
        "scales": scales,
        "whiten": whiten,
        "out_c": group * nc_per_group,
    }


def apply_gwpca_with_state(x, state):
    x_is_numpy = isinstance(x, np.ndarray)
    if x_is_numpy:
        x = torch.from_numpy(x)
    if not torch.is_tensor(x):
        raise TypeError("apply_gwpca_with_state expects numpy.ndarray or torch.Tensor")

    x = x.float()
    x = _ensure_bchw(x)
    b, c, h, w = x.shape
    out = torch.zeros((b, state["out_c"], h, w), dtype=x.dtype, device=x.device)

    c_start = 0
    out_start = 0
    for gi, c_per_group in enumerate(state["group_sizes"]):
        c0 = c_start
        c1 = c_start + c_per_group
        c_start = c1
        if c_per_group == 0:
            out_start += state["nc_per_group"]
            continue
        xg = x[:, c0:c1]
        xg = xg.permute(0, 2, 3, 1).reshape(-1, c_per_group)
        mean = state["means"][gi].to(x.device)
        comps = state["components"][gi].to(x.device)
        if comps.numel() == 0:
            out_start += state["nc_per_group"]
            continue
        xg = xg - mean
        z = xg @ comps
        if state["whiten"]:
            scale = state["scales"][gi].to(x.device)
            z = z * scale
        k = comps.shape[1]
        z = z.reshape(b, h, w, k).permute(0, 3, 1, 2)
        out[:, out_start:out_start + k] = z
        out_start += state["nc_per_group"]

    if x_is_numpy:
        return out.cpu().numpy()
    return out


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        # [H, W, C] -> [1, C, H, W]
        return x.permute(2, 0, 1).unsqueeze(0)
    if x.ndim == 4:
        # Heuristic: [B, H, W, C] if spatial dims match and last dim is channels.
        if x.shape[1] == x.shape[2] and x.shape[-1] != x.shape[1]:
            return x.permute(0, 3, 1, 2)
        return x
    raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")


def apply_gwpca(x, group: int = 8, nc_per_group: int = 8, whiten: bool = True):
    x_is_numpy = isinstance(x, np.ndarray)
    if x_is_numpy:
        x = torch.from_numpy(x)
    if not torch.is_tensor(x):
        raise TypeError("apply_gwpca expects numpy.ndarray or torch.Tensor")

    x = x.float()
    x = _ensure_bchw(x)
    b, c, h, w = x.shape
    base = c // group
    rem = c % group
    group_sizes = [base + (1 if i < rem else 0) for i in range(group)]
    out_c = group * nc_per_group
    out = torch.zeros((b, out_c, h, w), dtype=x.dtype, device=x.device)
    eps = 1e-6

    for bi in range(b):
        c_start = 0
        for gi, c_per_group in enumerate(group_sizes):
            c0 = c_start
            c1 = c_start + c_per_group
            c_start = c1
            if c_per_group == 0:
                continue
            xg = x[bi, c0:c1]  # [Cg, H, W]
            xg = xg.permute(1, 2, 0).reshape(-1, c_per_group)  # [N, Cg]
            xg = xg - xg.mean(dim=0, keepdim=True)
            if c_per_group == 1:
                comps = xg
                if whiten:
                    comps = comps / (xg.std(dim=0, keepdim=True) + eps)
                comps = comps[:, :1]
            else:
                u, s, vh = torch.linalg.svd(xg, full_matrices=False)
                v = vh.transpose(0, 1)
                k = min(nc_per_group, v.shape[1], s.numel())
                comps = xg @ v[:, :k]
                if whiten and k > 0:
                    denom = s[:k] + eps
                    comps = comps / denom
            k_out = min(nc_per_group, comps.shape[1])
            comps = comps[:, :k_out]
            comps = comps.reshape(h, w, k_out).permute(2, 0, 1)
            out_c0 = gi * nc_per_group
            out[bi, out_c0:out_c0 + k_out] = comps
            if k_out < nc_per_group:
                out[bi, out_c0 + k_out:out_c0 + nc_per_group] = 0.0

    if x_is_numpy:
        return out.cpu().numpy()
    return out
