import pandas as pd
import pytest

from src.ingest import (
    SchemaValidationError,
    build_binary_target,
    coerce_numeric,
    fold_rare_classes,
    validate_schema,
)


def test_validate_schema_passes_with_required_columns(synthetic_flows):
    validate_schema(synthetic_flows, ["ts", "id.orig_h", "id.resp_h"])


def test_validate_schema_raises_when_columns_missing():
    df = pd.DataFrame({"ts": [1, 2]})
    with pytest.raises(SchemaValidationError):
        validate_schema(df, ["ts", "uid", "id.orig_h", "id.resp_h"])


def test_coerce_numeric_handles_zeek_null_token():
    df = pd.DataFrame({"orig_bytes": ["100", "-", "200"]})
    out = coerce_numeric(df)
    assert out["orig_bytes"].isna().sum() == 1
    assert out["orig_bytes"].dropna().tolist() == [100.0, 200.0]


def test_build_binary_target_excludes_non_c2_malicious(synthetic_flows):
    df = synthetic_flows.copy()
    df.loc[0, "detailed-label"] = "PartOfAHorizontalPortScan"
    out = build_binary_target(
        df,
        positive_labels=["C&C-HeartBeat"],
        negative_labels=["Benign"],
        excluded_labels=["PartOfAHorizontalPortScan"],
    )
    assert len(out) == len(df) - 1
    assert set(out["target"].unique()) <= {0, 1}
    assert out.loc[out["detailed-label"] == "C&C-HeartBeat", "target"].eq(1).all()
    assert out.loc[out["detailed-label"] == "Benign", "target"].eq(0).all()


def test_fold_rare_classes_folds_below_threshold(synthetic_flows):
    out = fold_rare_classes(synthetic_flows, min_count=10, fold_into="other-C&C")
    assert (out.loc[out["detailed-label"] == "C&C-HeartBeat", "multiclass_label"] == "other-C&C").all()
