"""FR-8 — Evaluation and reporting: three splitting strategies, the full
metric suite with PR-AUC as primary, per-tier ablation, and reproducible
artefact generation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from .config import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------- #
# FR-8.6/8.7 — splitting strategies. §8.1: all three must be reported;
# the gap between split 1 and splits 2/3 is itself a finding.
# --------------------------------------------------------------------- #

def random_stratified_split(
    X: pd.DataFrame, y: pd.Series, ratios: list[float], seed: int
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Split 1: the optimistic baseline (§8.1)."""
    train_frac, val_frac, test_frac = ratios
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_frac + test_frac), stratify=y, random_state=seed
    )
    relative_test = test_frac / (val_frac + test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, stratify=y_temp, random_state=seed
    )
    return {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}


def temporal_split(
    X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, ratios: list[float]
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Split 2: train on chronologically earlier captures, test on later
    (Anderson & McGrew 2017). Ordered by each session's first_ts."""
    order = meta["first_ts"].sort_values().index
    n = len(order)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_idx, val_idx, test_idx = order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]
    return {
        "train": (X.loc[train_idx], y.loc[train_idx]),
        "val": (X.loc[val_idx], y.loc[val_idx]),
        "test": (X.loc[test_idx], y.loc[test_idx]),
    }


def held_out_family_split(
    X: pd.DataFrame,
    y: pd.Series,
    family_labels: pd.Series,
    test_fraction: float,
    seed: int,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Split 3: train on a subset of malware families, test on entirely
    unseen ones (EarlyCrow protocol). `family_labels` should be the
    detailed-label (or capture_id, as a proxy for family) aligned to X's
    index. Benign sessions are distributed across both sides in
    proportion, since they carry no family identity to hold out."""
    rng = np.random.default_rng(seed)
    positive_families = sorted(family_labels[y == 1].unique())
    n_test_families = max(1, round(len(positive_families) * test_fraction))
    test_families = set(rng.choice(positive_families, size=n_test_families, replace=False))

    is_test_family = family_labels.isin(test_families) & (y == 1)
    benign_idx = y[y == 0].index
    benign_train_idx, benign_test_idx = train_test_split(
        benign_idx, test_size=test_fraction, random_state=seed
    )

    test_idx = X.index[is_test_family].union(benign_test_idx)
    train_idx = X.index.difference(test_idx).union(benign_train_idx).difference(test_idx)

    logger.info(
        "Held-out-family split: %d families held out (%s), %d test rows, %d train rows",
        len(test_families), sorted(test_families), len(test_idx), len(train_idx),
    )
    return {
        "train": (X.loc[train_idx], y.loc[train_idx]),
        "test": (X.loc[test_idx], y.loc[test_idx]),
        "held_out_families": sorted(test_families),
    }


# --------------------------------------------------------------------- #
# FR-8.1-8.5 — metric suite
# --------------------------------------------------------------------- #

def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (scores >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, scores)  # FR-8.1: primary metric

    metrics = {
        "pr_auc": pr_auc,
        "precision_at_threshold": precision_score(y_true, y_pred, zero_division=0),
        "recall_at_threshold": recall_score(y_true, y_pred, zero_division=0),
        "f1_at_threshold": f1_score(y_true, y_pred, zero_division=0),
        # FR-8.3: secondary metrics only, flagged as misleading under imbalance.
        "roc_auc_SECONDARY": roc_auc_score(y_true, scores),
        "accuracy_SECONDARY": accuracy_score(y_true, y_pred),
        "accuracy_caveat": (
            "Accuracy/ROC-AUC are reported for completeness only. Under the "
            "severe class imbalance in IoT-23 (§4.5), a trivial all-benign "
            "classifier scores >99% accuracy while detecting zero C2 "
            "traffic. PR-AUC is the metric that should drive any "
            "conclusion about model quality here."
        ),
        "threshold": threshold,
    }
    return metrics


def false_positives_per_day(
    y_true: np.ndarray, scores: np.ndarray, threshold: float, assumed_daily_flow_volume: int
) -> float:
    """FR-8.4: projected false positives per day at a stated flow volume."""
    y_pred = (scores >= threshold).astype(int)
    fp_rate_in_test = ((y_pred == 1) & (y_true == 0)).sum() / max(1, (y_true == 0).sum())
    return fp_rate_in_test * assumed_daily_flow_volume


def compute_confusion_matrix(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """FR-8.5."""
    y_pred = (scores >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def pr_roc_curve_data(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    return {
        "precision": precision.tolist(), "recall": recall.tolist(),
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
    }


# --------------------------------------------------------------------- #
# FR-8.8 — per-tier ablation
# --------------------------------------------------------------------- #

def run_ablation(
    tiers_list: list[list[str]],
    fit_and_score_fn,
    select_tiers_fn,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> pd.DataFrame:
    """FR-8.8: Tier 1 only, Tier 1+2, Tier 1+2+4 — quantify each tier's
    marginal contribution, i.e. the encryption-resistant performance floor.

    `fit_and_score_fn(X_train, y_train, X_test) -> scores_on_test` and
    `select_tiers_fn(X, tiers) -> X_subset` are injected so this module
    doesn't import models/pipeline directly (kept for testability).
    """
    rows = []
    for tiers in tiers_list:
        X_train_sub = select_tiers_fn(X_train, tiers)
        X_test_sub = select_tiers_fn(X_test, tiers)
        scores = fit_and_score_fn(X_train_sub, y_train, X_test_sub)
        pr_auc = average_precision_score(y_test, scores)
        rows.append({"tiers": "+".join(tiers), "n_features": X_train_sub.shape[1], "pr_auc": pr_auc})
        logger.info("Ablation %s: PR-AUC=%.4f (%d features)", tiers, pr_auc, X_train_sub.shape[1])
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# FR-8.9 — reproducible artefact generation
# --------------------------------------------------------------------- #

def save_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Evaluation report written to %s", path)
