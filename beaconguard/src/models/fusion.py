"""FR-5.1, FR-5.2 — Score fusion: combine the supervised probability and
the normalised anomaly score into one risk value in [0,1].
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score


def weighted_average(
    supervised_score: np.ndarray,
    anomaly_score: np.ndarray,
    supervised_weight: float,
    anomaly_weight: float,
) -> np.ndarray:
    total = supervised_weight + anomaly_weight
    return (supervised_weight * supervised_score + anomaly_weight * anomaly_score) / total


def max_score(supervised_score: np.ndarray, anomaly_score: np.ndarray) -> np.ndarray:
    return np.maximum(supervised_score, anomaly_score)


def fuse(
    supervised_score: np.ndarray, anomaly_score: np.ndarray, cfg: dict[str, Any]
) -> np.ndarray:
    method = cfg["fusion"]["method"]
    if method == "weighted_average":
        return weighted_average(
            supervised_score, anomaly_score,
            cfg["fusion"]["supervised_weight"], cfg["fusion"]["anomaly_weight"],
        )
    elif method == "max_score":
        return max_score(supervised_score, anomaly_score)
    raise ValueError(f"Unknown fusion method: {method}")


def compare_fusion_methods(
    supervised_score: np.ndarray, anomaly_score: np.ndarray, y_true: np.ndarray, cfg: dict[str, Any]
) -> dict[str, float]:
    """FR-5.2: compare weighted-average and max-score fusion by PR-AUC, so
    the fusion.method choice in config is empirically justified (FR-5.1)."""
    results = {
        "weighted_average": average_precision_score(
            y_true,
            weighted_average(
                supervised_score, anomaly_score,
                cfg["fusion"]["supervised_weight"], cfg["fusion"]["anomaly_weight"],
            ),
        ),
        "max_score": average_precision_score(y_true, max_score(supervised_score, anomaly_score)),
    }
    return results
