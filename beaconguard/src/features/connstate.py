"""FR-2.4 — Connection-state features, aggregated to session granularity.

Categorical columns (conn_state_mode, proto_mode, service_mode) are left
as strings here and one-hot encoded later by encode_categoricals(), which
must be fit on the training split only (FR-2.6.5) — encoding inside this
module would risk leaking test-set categories into the training vocabulary.
"""
from __future__ import annotations

import pandas as pd

TIER = "connstate"  # cross-cuts tiers 1/2 in the ablation; treated as always-on


def _mode_or_unknown(s: pd.Series) -> str:
    m = s.mode(dropna=True)
    return str(m.iloc[0]) if not m.empty else "unknown"


def _session_agg(group: pd.DataFrame, ephemeral_floor: int, common_ports: set[int]) -> pd.Series:
    history = group["history"].fillna("").astype(str)
    resp_port = group["id.resp_p"].iloc[0]
    orig_ports = group["id.orig_p"]

    return pd.Series({
        "conn_state_mode": _mode_or_unknown(group["conn_state"]),      # FR-2.4.1
        "proto_mode": _mode_or_unknown(group["proto"]),                # FR-2.4.3
        "service_mode": _mode_or_unknown(group["service"]),
        "history_len": float(history.str.len().mean()),                # FR-2.4.2
        "history_has_S": float(history.str.contains("S").mean()),
        "history_has_D": float(history.str.contains("D").mean()),
        "resp_port": float(resp_port),                                  # FR-2.4.4
        "is_ephemeral_orig_port": float((orig_ports >= ephemeral_floor).mean()),
        "is_common_service_port": float(resp_port in common_ports),
    })


def build(df: pd.DataFrame, ephemeral_floor: int, common_ports: list[int]) -> pd.DataFrame:
    """Returns one row per session_id with connection-state features."""
    common_ports_set = set(common_ports)
    return df.groupby("session_id", sort=False).apply(
        lambda g: _session_agg(g, ephemeral_floor, common_ports_set), include_groups=False
    )


CATEGORICAL_COLUMNS = ["conn_state_mode", "proto_mode", "service_mode"]
NUMERIC_COLUMNS = [
    "history_len", "history_has_S", "history_has_D", "resp_port",
    "is_ephemeral_orig_port", "is_common_service_port",
]


def encode_categoricals(
    train_df: pd.DataFrame,
    other_dfs: list[pd.DataFrame],
    columns: list[str] = CATEGORICAL_COLUMNS,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """FR-2.4.1/2.4.3 + FR-2.6.5: one-hot encode with an explicit 'unknown'
    category, using only categories observed in the training split. Any
    category in other_dfs not seen in training collapses to unknown, so
    the test-time feature space can never silently grow.

    Preserves each input frame's original index (session_id) on the way
    out, since callers align the result back against `y`/`meta` via
    `.loc[session_id]` — a plain reset_index here would silently break
    that alignment.
    """
    train_out = train_df.copy()
    others_out = [d.copy() for d in other_dfs]

    for col in columns:
        train_out[col] = train_out[col].fillna("unknown")
        vocab = set(train_out[col].unique())
        for d in others_out:
            d[col] = d[col].where(d[col].isin(vocab), other="unknown").fillna("unknown")

    all_dummies = pd.get_dummies(
        pd.concat([train_out[columns]] + [d[columns] for d in others_out], ignore_index=True),
        columns=columns,
    )

    n_train = len(train_out)
    train_dummies = all_dummies.iloc[:n_train]
    train_dummies.index = train_out.index
    train_result = pd.concat([train_out.drop(columns=columns), train_dummies], axis=1)

    offset = n_train
    results = [train_result]
    for d in others_out:
        n = len(d)
        d_dummies = all_dummies.iloc[offset: offset + n]
        d_dummies.index = d.index
        results.append(pd.concat([d.drop(columns=columns), d_dummies], axis=1))
        offset += n

    return results[0], results[1:]
