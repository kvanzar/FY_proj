"""FR-5.3-5.6 — Threshold calibration and the offline enforcement
simulation (Phase 1 stands in for genuine inline blocking; see PRD §13.4
/ FS-4 for the Phase 2 real-firewall integration this offline simulation
is designed to slot into unchanged).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import get_logger

logger = get_logger(__name__)


def calibrate_thresholds(
    risk_scores: np.ndarray,
    alerts_per_day_budget: int,
    assumed_daily_flow_volume: int,
) -> tuple[float, float]:
    """FR-5.3: derive tau_low/tau_high from an operator alerts-per-day
    budget projected onto a stated flow volume.

    tau_high is the score at the percentile that yields exactly
    `alerts_per_day_budget` ALERT+BLOCK verdicts out of
    `assumed_daily_flow_volume` flows/day, assuming risk_scores'
    empirical distribution is representative of one day of traffic.
    tau_low is set at twice that alert rate, marking the ALLOW/ALERT
    boundary below which flows are not surfaced to an analyst at all.
    """
    alert_fraction = alerts_per_day_budget / assumed_daily_flow_volume
    tau_high = float(np.quantile(risk_scores, 1 - alert_fraction)) if alert_fraction < 1 else 0.0
    tau_low_fraction = min(1.0, alert_fraction * 2)
    tau_low = float(np.quantile(risk_scores, 1 - tau_low_fraction))
    tau_low = min(tau_low, tau_high)
    logger.info(
        "Calibrated thresholds from budget=%d/day over %d flows/day: "
        "tau_low=%.4f, tau_high=%.4f",
        alerts_per_day_budget, assumed_daily_flow_volume, tau_low, tau_high,
    )
    return tau_low, tau_high


def decide(risk_scores: np.ndarray, tau_low: float, tau_high: float) -> np.ndarray:
    """FR-5.4: map risk to ALLOW / ALERT / BLOCK."""
    actions = np.full(len(risk_scores), "ALLOW", dtype=object)
    actions[risk_scores >= tau_low] = "ALERT"
    actions[risk_scores >= tau_high] = "BLOCK"
    return actions


def build_decision_log(
    meta: pd.DataFrame,
    risk_scores: np.ndarray,
    supervised_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    actions: np.ndarray,
    explanations: list[str] | None = None,
) -> pd.DataFrame:
    """FR-5.5: every decision logged with flow ID, risk score, branch
    scores, action and (if provided by explain.py) top contributing
    features."""
    log = pd.DataFrame({
        "session_id": meta.index,
        "flow_id": meta["last_uid"].to_numpy(),
        "capture_id": meta["capture_id"].to_numpy(),
        "risk_score": risk_scores,
        "supervised_score": supervised_scores,
        "anomaly_score": anomaly_scores,
        "action": actions,
        "decided_at": datetime.now(timezone.utc).isoformat(),
    })
    if explanations is not None:
        log["explanation"] = explanations
    return log


def save_decision_log(log: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    logger.info("Decision log written to %s (%d entries)", path, len(log))


def save_thresholds(tau_low: float, tau_high: float, path: str | Path) -> None:
    """Persists the last-computed thresholds alongside config for
    reproducibility/audit (mirrors decision.tau_low/tau_high in
    config/default.yaml, which stay null until a calibration run fills
    them in)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tau_low": tau_low, "tau_high": tau_high}, indent=2))
