"""FR-1 — Data ingestion and validation.

Loads Zeek conn.log(.labeled) files or CSV exports of IoT-23, validates
that timing/identity columns survived preprocessing (see PRD §4.1 critical
note), coerces Zeek's ``-`` null convention, and emits a data profile.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import get_logger

logger = get_logger(__name__)

# Zeek conn.log.labeled columns, in order, when reading the raw
# whitespace-separated TSV format (as opposed to a pre-flattened CSV).
ZEEK_CONN_COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "tunnel_parents", "label", "detailed-label",
]

NUMERIC_COLUMNS = [
    "ts", "id.orig_p", "id.resp_p", "duration", "orig_bytes", "resp_bytes",
    "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
]


class SchemaValidationError(RuntimeError):
    """Raised when a required column (FR-1.2) is absent from the source file.

    See PRD §4.1: several circulating preprocessed IoT-23 variants drop
    ts/uid/id.orig_h/id.resp_h. Those variants cannot support timing
    features and must be rejected loudly rather than silently degraded.
    """


def _read_single_file(path: Path, null_token: str) -> pd.DataFrame:
    if path.suffix in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        df = pd.read_csv(path, na_values=[null_token], low_memory=False)
    else:
        # Native Zeek conn.log.labeled: whitespace-delimited, '#fields' header,
        # 'label\tdetailed-label' collapsed into one trailing tab-separated pair.
        df = pd.read_csv(
            path,
            sep=r"\s+",
            comment="#",
            names=ZEEK_CONN_COLUMNS,
            na_values=[null_token],
            engine="python",
        )

    # Some CSV exports (e.g. Kaggle's iot23preprocesseddata) were written
    # from a pandas DataFrame with the row index included, producing an
    # unnamed leading column of sequential integers. It carries no
    # information and would otherwise ride along as a spurious feature.
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed: \d+$")]

    # Some CSV exports collapse Zeek's separate `label`/`detailed-label`
    # pair into a single `label` column that actually holds the
    # fine-grained value (e.g. 'C&C-HeartBeat', 'PartOfAHorizontalPortScan').
    # Normalise it to `detailed-label` so the rest of the pipeline — which
    # is written against native conn.log.labeled's two-column convention —
    # doesn't need to special-case this. Only rename when `detailed-label`
    # doesn't already exist, since native Zeek files have both columns.
    if "detailed-label" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "detailed-label"})

    return df


def load_conn_logs(
    paths: list[str | Path],
    required_columns: list[str],
    null_token: str = "-",
) -> pd.DataFrame:
    """FR-1.1, FR-1.5: load one or more capture files, preserving a
    capture_id column so held-out-family / temporal splitting (FR-8.6,
    FR-8.7) can later be keyed off the source capture.
    """
    if not paths:
        raise ValueError(
            "No raw data paths configured. Populate data.raw_paths in "
            "config/default.yaml after downloading IoT-23 (see README.md)."
        )

    frames = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Configured data path does not exist: {path}")
        df = _read_single_file(path, null_token)
        df["capture_id"] = path.stem
        frames.append(df)
        logger.info("Loaded %s: %d rows", path.name, len(df))

    combined = pd.concat(frames, ignore_index=True)
    validate_schema(combined, required_columns)
    combined = coerce_numeric(combined)
    logger.info("Combined dataset: %d rows from %d capture(s)", len(combined), len(paths))
    return combined


def validate_schema(df: pd.DataFrame, required_columns: list[str]) -> None:
    """FR-1.2: abort with a clear error if essential columns are absent."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"Missing required column(s) {missing}. This dataset variant "
            "cannot support timing-feature derivation. Use the full IoT-23 "
            "variant or the original conn.log.labeled files — see PRD §4.1."
        )


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """FR-1.3: coerce numeric columns, treating Zeek '-' as null rather
    than leaving it as a literal string that silently poisons downstream
    arithmetic."""
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_data_profile(df: pd.DataFrame, label_column: str = "detailed-label") -> dict[str, Any]:
    """FR-1.4: row count, null fraction per column, label distribution."""
    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "null_fraction": df.isna().mean().round(4).to_dict(),
    }
    if label_column in df.columns:
        counts = df[label_column].value_counts(dropna=False)
        profile["label_distribution"] = counts.to_dict()
        profile["label_fraction"] = (counts / len(df)).round(6).to_dict()
    logger.info(
        "Data profile: %d rows, %d columns, label distribution: %s",
        profile["row_count"], profile["column_count"],
        profile.get("label_distribution", {}),
    )
    return profile


def stratified_sample(
    df: pd.DataFrame,
    working_size: int | None,
    label_column: str = "detailed-label",
    seed: int = 42,
) -> pd.DataFrame:
    """FR-1.6: subsample to a configurable working size for fast dev
    iteration, preserving each label's relative frequency."""
    if working_size is None or working_size >= len(df):
        return df
    frac = working_size / len(df)
    sampled = (
        df.groupby(label_column, group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=seed))
    )
    logger.info("Stratified sample: %d -> %d rows", len(df), len(sampled))
    return sampled.reset_index(drop=True)


def build_binary_target(
    df: pd.DataFrame,
    positive_labels: list[str],
    negative_labels: list[str],
    excluded_labels: list[str],
    label_column: str = "detailed-label",
) -> pd.DataFrame:
    """PRD §4.4: binary target construction. Flows whose detailed-label is
    neither a positive nor a negative label (e.g. DDoS, PartOfAHorizontalPortScan)
    are dropped entirely — they are malicious-but-not-C2 and including them
    on either side would corrupt the attack-specific scope of this project."""
    df = df.copy()
    labels = df[label_column].astype(str).str.strip()

    is_positive = labels.isin(positive_labels)
    is_negative = labels.isin(negative_labels)
    is_excluded = labels.isin(excluded_labels)

    unrecognised = ~(is_positive | is_negative | is_excluded)
    if unrecognised.any():
        logger.warning(
            "%d rows have unrecognised detailed-label values not listed in "
            "config; treating as excluded: %s",
            unrecognised.sum(),
            labels[unrecognised].unique().tolist(),
        )

    keep_mask = is_positive | is_negative
    result = df.loc[keep_mask].copy()
    result["target"] = is_positive.loc[keep_mask].astype(int)

    logger.info(
        "Binary target built: %d positive, %d negative, %d excluded "
        "(of %d total rows)",
        int(is_positive.sum()), int(is_negative.sum()),
        len(df) - keep_mask.sum(), len(df),
    )
    return result


def fold_rare_classes(
    df: pd.DataFrame,
    label_column: str = "detailed-label",
    min_count: int = 50,
    fold_into: str = "other-C&C",
) -> pd.DataFrame:
    """PRD §4.5 point 3: multi-class analysis aid — categories below
    min_count are folded into a combined bucket so per-class evaluation
    doesn't collapse to zero-sample folds."""
    df = df.copy()
    counts = df[label_column].value_counts()
    rare = counts[counts < min_count].index
    is_ccc = df[label_column].astype(str).str.startswith("C&C")
    df["multiclass_label"] = np.where(
        df[label_column].isin(rare) & is_ccc, fold_into, df[label_column]
    )
    return df
