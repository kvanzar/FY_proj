"""Top-level orchestration: raw flows -> sessionized flows -> joined,
tiered feature matrix + labels + metadata. This is the one place that
knows how ingest.py, sessionize.py and features/* compose, so run_pipeline.py
and evaluate.py both call through here rather than re-deriving the wiring.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .config import get_logger
from .features import connstate, context, timing, volume
from .ingest import build_binary_target, coerce_numeric, fold_rare_classes, load_conn_logs, stratified_sample
from .sessionize import sessionize

logger = get_logger(__name__)

# Columns explicitly excluded from the model-facing feature matrix.
# FR-2.6.1: no raw IP addresses. FR-2.6.2: no uid / capture filename.
IDENTITY_COLUMNS = ["id.orig_h", "id.resp_h", "uid", "capture_id"]


def load_and_prepare(cfg: dict[str, Any]) -> pd.DataFrame:
    """FR-1 + PRD §4.4: load, validate, and binarise the target."""
    df = load_conn_logs(
        cfg["data"]["raw_paths"],
        cfg["data"]["required_columns"],
        cfg["data"]["null_token"],
    )
    df = build_binary_target(
        df,
        cfg["data"]["positive_labels"],
        cfg["data"]["negative_labels"],
        cfg["data"]["excluded_labels"],
    )
    df = fold_rare_classes(df, min_count=cfg["data"]["rare_class_min_count"])

    # FR-1.6: subsample for fast dev iteration. Applied at flow (row)
    # granularity, before sessionize.py groups flows into sessions — on a
    # heavily reduced working_sample_size this can shrink or fragment
    # some sessions below the min_session_flows filter, which is an
    # accepted trade-off for iteration speed. Use the full dataset
    # (working_sample_size: null) for any result that will be reported.
    df = stratified_sample(
        df,
        cfg["data"]["working_sample_size"],
        label_column="detailed-label",
        seed=cfg["seed"],
    )
    return df


def build_feature_matrix(
    flows: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Sessionizes `flows` and joins all Tier 1/2/connstate/Tier 4 feature
    blocks into one session-level matrix.

    Returns (X, y, meta):
      X    — model-facing feature matrix, one row per session, no identity
             columns (FR-2.6.1, FR-2.6.2).
      y    — binary target per session: 1 if any flow in the session was
             labelled positive (a compromised host's beacon session is
             internally label-consistent in IoT-23).
      meta — session_id, capture_id, first/last ts, representative uid,
             and each feature's tier tag, kept OUT of X but needed by
             decision.py (flow ID in the decision log) and evaluate.py
             (temporal / held-out-family splitting).
    """
    sessionized, discard_rate = sessionize(
        flows, cfg["sessionize"]["group_keys"], cfg["sessionize"]["min_session_flows"]
    )
    logger.info("Sessionization discard rate: %.2f%%", discard_rate * 100)

    tier1 = timing.build(sessionized)
    tier2 = volume.build(sessionized)
    cs = connstate.build(
        sessionized,
        cfg["features"]["ephemeral_port_floor"],
        cfg["features"]["common_service_ports"],
    )
    tier4 = context.build(sessionized)

    feature_block = tier1.join(tier2).join(cs).join(tier4)
    tier2_log = volume.log_transform(feature_block, cfg["features"]["log_transform_columns"])
    feature_block[cfg["features"]["log_transform_columns"]] = tier2_log[
        cfg["features"]["log_transform_columns"]
    ]

    session_target = sessionized.groupby("session_id", sort=False)["target"].max()
    session_meta = sessionized.groupby("session_id", sort=False).agg(
        capture_id=("capture_id", "first"),
        first_ts=("ts", "min"),
        last_ts=("ts", "max"),
        last_uid=("uid", "last"),
        # Malware-family signal for held_out_family_split (FR-8.7). This is
        # more reliable than capture_id when raw_paths is a single combined
        # file (e.g. the Kaggle iot23preprocesseddata export) with no
        # per-capture separation — capture_id would then be constant across
        # every session and the family split would have nothing to hold out.
        detailed_label=("detailed-label", "first"),
    )

    X = feature_block.loc[session_target.index]
    y = session_target
    meta = session_meta.loc[session_target.index]

    logger.info("Feature matrix built: %d sessions, %d features", len(X), X.shape[1])
    return X, y, meta


FEATURE_TIER_MAP = {
    "tier1": timing.TIER1_COLUMNS,
    "tier2": volume.TIER2_COLUMNS,
    "tier4": context.TIER4_COLUMNS,
    # connstate columns are always included; not gated by the tier ablation
    # since they represent protocol structure rather than an encryption-
    # sensitive visibility tier in the PRD's own taxonomy (§7 module list).
}


def select_tiers(X: pd.DataFrame, tiers: list[str]) -> pd.DataFrame:
    """FR-8.8: select the columns belonging to the requested tiers, plus
    always-on connstate columns, for per-tier ablation."""
    always_on = [c for c in connstate.NUMERIC_COLUMNS if c in X.columns]
    always_on += [c for c in X.columns if any(c.startswith(f"{cat}_") for cat in connstate.CATEGORICAL_COLUMNS)]
    selected = list(always_on)
    for tier in tiers:
        selected += [c for c in FEATURE_TIER_MAP.get(tier, []) if c in X.columns]
    return X[sorted(set(selected), key=selected.index)]
