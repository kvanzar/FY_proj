"""FR-2.3 — Tier 2 volume features, aggregated to session granularity
(see timing.py docstring for the session-as-unit-of-analysis rationale).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIER = 2


def _session_agg(group: pd.DataFrame) -> pd.Series:
    orig_bytes = group["orig_ip_bytes"].fillna(0)
    resp_bytes = group["resp_ip_bytes"].fillna(0)
    # Payload bytes (excl. IP/TCP headers) — the dataset's own C&C-HeartBeat
    # labelling criterion is "response bytes under 1 byte" measured on this
    # field. resp_ip_bytes always includes header overhead and is virtually
    # never exactly zero, so it cannot stand in for this feature.
    resp_payload_bytes = group["resp_bytes"].fillna(0)
    orig_pkts = group["orig_pkts"].fillna(0)
    resp_pkts = group["resp_pkts"].fillna(0)
    duration = group["duration"].fillna(0)

    total_orig = float(orig_bytes.sum())
    total_resp = float(resp_bytes.sum())
    total_bytes = total_orig + total_resp
    total_orig_pkts = float(orig_pkts.sum())
    total_resp_pkts = float(resp_pkts.sum())
    total_duration = float(duration.sum())

    return pd.Series({
        "orig_resp_byte_ratio": total_orig / (total_resp + 1),      # FR-2.3.1
        "total_bytes": total_bytes,                                  # FR-2.3.2 (log-transformed later)
        "orig_bytes": total_orig,
        "resp_bytes": total_resp,
        "orig_pkts": total_orig_pkts,                                # FR-2.3.3
        "resp_pkts": total_resp_pkts,
        "pkt_ratio": total_orig_pkts / (total_resp_pkts + 1),
        "mean_pkt_size_orig": total_orig / (total_orig_pkts + 1),    # FR-2.3.4
        "mean_pkt_size_resp": total_resp / (total_resp_pkts + 1),
        "bytes_per_second": total_bytes / (total_duration + 1e-6),   # FR-2.3.5
        "resp_bytes_is_zero": float((resp_payload_bytes == 0).mean()),  # FR-2.3.6
        "session_byte_variance": float((orig_bytes + resp_bytes).var(ddof=0)),  # FR-2.3.7
        "duration": float(duration.mean()),                          # FR-2.3.8
    })


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per session_id with Tier 2 volume features."""
    return df.groupby("session_id", sort=False).apply(_session_agg, include_groups=False)


def log_transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """FR-2.6.4: log-transform heavy-tailed volume columns in place
    (log1p handles zeros without producing -inf)."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


TIER2_COLUMNS = [
    "orig_resp_byte_ratio", "total_bytes", "orig_bytes", "resp_bytes",
    "orig_pkts", "resp_pkts", "pkt_ratio", "mean_pkt_size_orig",
    "mean_pkt_size_resp", "bytes_per_second", "resp_bytes_is_zero",
    "session_byte_variance", "duration",
]
