"""FR-6 — Adversarial evaluation harness: synthetic jitter injection and
packet-padding simulation, producing the jitter degradation curve that is
this project's headline robustness result (AC-5, GAP-2).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import get_logger

logger = get_logger(__name__)


def inject_jitter(
    flows: pd.DataFrame,
    jitter_pct: float,
    session_id_col: str = "session_id",
    seed: int = 42,
) -> pd.DataFrame:
    """FR-6.1: perturb positive-class flow timestamps by ±jitter_pct% of
    the session's base (mean) inter-arrival interval, then re-derive `ts`
    ordering. Operates on the flow-level (pre-aggregation) frame so the
    perturbed timestamps flow naturally into sessionize.py + features/*
    on the next pipeline pass.
    """
    rng = np.random.default_rng(seed)
    df = flows.copy()
    if jitter_pct == 0:
        return df

    def _jitter_session(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("ts").copy()
        deltas = group["ts"].diff().dropna()
        base_interval = deltas.mean() if len(deltas) else 0.0
        if base_interval <= 0:
            return group
        noise = rng.uniform(-jitter_pct / 100, jitter_pct / 100, size=len(group)) * base_interval
        group["ts"] = group["ts"] + noise
        return group.sort_values("ts")

    jittered = df.groupby(session_id_col, group_keys=False).apply(_jitter_session)
    return jittered.reset_index(drop=True)


def inject_padding(
    flows: pd.DataFrame, padding_factor: float, seed: int = 42
) -> pd.DataFrame:
    """FR-6.4: inflate byte counts by a random factor in [1, padding_factor]
    to simulate C2 frameworks that pad payloads to defeat volume-based
    detection."""
    rng = np.random.default_rng(seed)
    df = flows.copy()
    factors = rng.uniform(1.0, padding_factor, size=len(df))
    for col in ["orig_bytes", "orig_ip_bytes"]:
        if col in df.columns:
            df[col] = (df[col].fillna(0) * factors).round()
    return df


def recall_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Fraction of true-positive rows scored at or above a fixed
    threshold.

    This intentionally does NOT derive a threshold from
    `precision_recall_curve` on the sweep's own scored set. Jitter/padding
    sweeps are, by construction (FR-6.1: "perturb C2 flow timestamps"),
    applied only to positive-class sessions — every row in `y_true` here
    is 1. With no negatives present, precision is trivially 1.0 at every
    threshold, so a 'recall at target precision' computed from this set
    would be meaningless (it would always return recall at threshold=0,
    i.e. 1.0, regardless of how badly jitter degraded the scores).

    Instead the threshold is fixed *before* the sweep starts — the
    caller passes in an operating threshold already calibrated against a
    real positive+negative test set (see run_pipeline.py, which uses the
    decision engine's tau_high). Holding that threshold constant while
    jitter increases is what actually answers FR-6.2/AC-5's question:
    "how much does recall against our real operating point degrade as
    an attacker randomises their beacon timing?"
    """
    positive_mask = y_true == 1
    if positive_mask.sum() == 0:
        return float("nan")
    return float((scores[positive_mask] >= threshold).mean())


def jitter_sweep(
    flows_positive_only: pd.DataFrame,
    rebuild_and_score: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray]],
    jitter_percentages: list[float],
    threshold: float,
    seed: int = 42,
) -> pd.DataFrame:
    """FR-6.2, FR-6.3: sweep jitter from 0-50% and measure recall against
    a fixed operating threshold at each level, producing the degradation
    curve.

    `rebuild_and_score(jittered_flows) -> (y_true, risk_scores)` is
    supplied by the caller (run_pipeline.py) so this module stays agnostic
    to exactly how features/model scoring are wired. `threshold` should
    be a threshold calibrated against a real positive+negative test set
    (e.g. decision.py's tau_high) — see `recall_at_threshold`'s docstring
    for why it can't be derived from this sweep's own data.
    """
    rows = []
    for pct in jitter_percentages:
        jittered = inject_jitter(flows_positive_only, pct, seed=seed)
        y_true, scores = rebuild_and_score(jittered)
        recall = recall_at_threshold(y_true, scores, threshold)
        rows.append({"jitter_pct": pct, "recall_at_threshold": recall, "threshold": threshold})
        logger.info("Jitter %d%%: recall@threshold(%.4f) = %.4f", pct, threshold, recall)
    return pd.DataFrame(rows)


def padding_sweep(
    flows_positive_only: pd.DataFrame,
    rebuild_and_score: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray]],
    padding_factors: list[float],
    threshold: float,
    seed: int = 42,
) -> pd.DataFrame:
    """FR-6.4 sweep. See `jitter_sweep` for why `threshold` is fixed and
    supplied by the caller rather than derived here."""
    rows = []
    for factor in padding_factors:
        padded = inject_padding(flows_positive_only, factor, seed=seed)
        y_true, scores = rebuild_and_score(padded)
        recall = recall_at_threshold(y_true, scores, threshold)
        rows.append({"padding_factor": factor, "recall_at_threshold": recall, "threshold": threshold})
        logger.info("Padding x%.2f: recall@threshold(%.4f) = %.4f", factor, threshold, recall)
    return pd.DataFrame(rows)
