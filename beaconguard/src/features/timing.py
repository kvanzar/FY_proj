"""FR-2.2 — Tier 1 timing features.

Design note: the unit of analysis for the whole feature matrix is the
*session* (a host-pair/port conversation, per sessionize.py), not the
individual flow. Periodicity is a property of a sequence of connections,
not of a single one, so every timing feature here is a per-session
aggregate over that session's inter-arrival-time (IAT) series. Tier 2-4
modules aggregate to the same session granularity so all tiers join on
session_id. See exec.md for the full rationale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIER = 1


def _autocorrelation_peak(x: np.ndarray) -> float:
    """FR-2.2.6: normalised autocorrelation peak of the IAT series.
    A strongly periodic series has a high-magnitude peak away from lag 0;
    an irregular series decays toward zero immediately.

    A (near-)constant IAT series — i.e. a beacon with negligible jitter —
    is the maximally periodic case, not the degenerate one: demeaning it
    yields a ~zero vector, which the autocorrelation is undefined over
    (0/0). That edge case must resolve to 1.0, not 0.0, or the score
    would rank a perfect beacon as perfectly *aperiodic*."""
    if len(x) < 2:
        return 0.0
    if np.allclose(x, x.mean()):
        return 1.0
    x = x - x.mean()
    full = np.correlate(x, x, mode="full")
    mid = len(full) // 2
    acf = full[mid:]
    acf = acf / acf[0] if acf[0] != 0 else acf
    return float(np.max(np.abs(acf[1:]))) if len(acf) > 1 else 0.0


def _fft_dominant_power(x: np.ndarray) -> float:
    """FR-2.2.7: power at the dominant frequency of the IAT series,
    normalised by total spectral power so it is comparable across
    sessions of different lengths/scales."""
    if len(x) < 2:
        return 0.0
    if np.allclose(x, x.mean()):
        return 1.0
    x = x - x.mean()
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    total = spectrum.sum()
    if total == 0 or len(spectrum) < 2:
        return 0.0
    return float(spectrum[1:].max() / total)


def _session_agg(group: pd.DataFrame) -> pd.Series:
    iat = group["iat"].dropna().to_numpy()
    ts = group["ts"].to_numpy()

    iat_mean = float(iat.mean()) if len(iat) else 0.0
    iat_std = float(iat.std()) if len(iat) else 0.0
    iat_cv = iat_std / iat_mean if iat_mean > 0 else 0.0
    span_seconds = float(ts.max() - ts.min()) if len(ts) else 0.0
    duration_sum = float(group["duration"].fillna(0).sum())

    hour = pd.to_datetime(ts[0], unit="s").hour if len(ts) else np.nan
    dow = pd.to_datetime(ts[0], unit="s").dayofweek if len(ts) else np.nan

    return pd.Series({
        "iat_mean": iat_mean,
        "iat_std": iat_std,
        "iat_cv": iat_cv,                                          # FR-2.2.3
        "iat_median": float(np.median(iat)) if len(iat) else 0.0,
        "iat_min": float(iat.min()) if len(iat) else 0.0,
        "iat_max": float(iat.max()) if len(iat) else 0.0,
        "iat_mad": float(np.median(np.abs(iat - np.median(iat)))) if len(iat) else 0.0,
        "periodicity_score": _autocorrelation_peak(iat) if len(iat) >= 3 else 0.0,
        "fft_dominant_power": _fft_dominant_power(iat) if len(iat) >= 3 else 0.0,
        "session_flow_count": float(len(group)),                   # FR-2.2.8
        "session_span_hours": span_seconds / 3600.0,                # FR-2.2.9
        "duty_cycle": duration_sum / span_seconds if span_seconds > 0 else 0.0,
        "hour_of_day": float(hour) if not np.isnan(hour) else -1.0,
        "is_weekend": float(dow >= 5) if not np.isnan(dow) else 0.0,
    })


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per session_id with Tier 1 timing features."""
    return df.groupby("session_id", sort=False).apply(_session_agg, include_groups=False)


TIER1_COLUMNS = [
    "iat_mean", "iat_std", "iat_cv", "iat_median", "iat_min", "iat_max",
    "iat_mad", "periodicity_score", "fft_dominant_power",
    "session_flow_count", "session_span_hours", "duty_cycle",
    "hour_of_day", "is_weekend",
]
