"""FR-2.1 — Session grouping. Prerequisite for every timing feature: a
beacon can only be recognised as periodic in the context of the sequence
of connections between the same host pair on the same destination port.
"""
from __future__ import annotations

import pandas as pd

from .config import get_logger

logger = get_logger(__name__)


def build_session_id(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    """FR-2.1.1: group flows into sessions keyed by (orig host, resp host,
    resp port). Using resp_port (not the full 5-tuple) deliberately groups
    together repeated connections to the same service, which is exactly
    the granularity at which beaconing manifests."""
    df = df.copy()
    df["session_id"] = df[group_keys].astype(str).agg("|".join, axis=1)
    return df


def sort_and_derive_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """FR-2.1.2, FR-2.1.3: sort each session by ts ascending and compute
    inter-arrival deltas between consecutive flows within the session."""
    df = df.sort_values(["session_id", "ts"]).copy()
    df["iat"] = df.groupby("session_id")["ts"].diff()
    return df


def filter_short_sessions(
    df: pd.DataFrame, min_flows: int = 4
) -> tuple[pd.DataFrame, float]:
    """FR-2.1.4: discard sessions with fewer than min_flows flows —
    periodicity cannot be meaningfully assessed from 1-3 samples. Returns
    the filtered frame and the discard rate for logging (R-2 risk)."""
    session_sizes = df.groupby("session_id").size()
    keep_sessions = session_sizes[session_sizes >= min_flows].index
    kept = df[df["session_id"].isin(keep_sessions)].copy()

    discard_rate = 1 - (len(kept) / len(df)) if len(df) else 0.0
    n_sessions_total = df["session_id"].nunique()
    n_sessions_kept = kept["session_id"].nunique()
    logger.info(
        "Session filter (min_flows=%d): kept %d/%d sessions, "
        "flow discard rate %.2f%%",
        min_flows, n_sessions_kept, n_sessions_total, discard_rate * 100,
    )
    return kept, discard_rate


def sessionize(
    df: pd.DataFrame, group_keys: list[str], min_flows: int = 4
) -> tuple[pd.DataFrame, float]:
    """Full FR-2.1 pipeline: group -> sort/derive intervals -> filter."""
    df = build_session_id(df, group_keys)
    df = sort_and_derive_intervals(df)
    return filter_short_sessions(df, min_flows)
