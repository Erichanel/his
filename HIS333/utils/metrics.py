"""Metrics utilities for anomaly detection."""

from __future__ import annotations

import numpy as np


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC; falls back to trapezoidal rule if sklearn is unavailable."""
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels.flatten(), scores.flatten()))
    except Exception:
        scores = scores.flatten()
        labels = labels.flatten()
        order = np.argsort(scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(scores))
        pos = labels == 1
        neg = labels == 0
        pos_ranks = ranks[pos]
        n_pos = pos.sum()
        n_neg = neg.sum()
        if n_pos == 0 or n_neg == 0:
            return 0.5
        auc = (pos_ranks.sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
        return float(auc)
