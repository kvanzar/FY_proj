"""FR-7 — Explainability: SHAP global importance and per-verdict local
attribution, rendered as text an analyst can act on without touching a
plot.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from .config import get_logger

logger = get_logger(__name__)

# Human-readable feature name/unit templates for FR-7.3. Falls back to the
# raw column name (with underscores replaced) for anything not listed.
_FEATURE_RENDER = {
    "iat_cv": lambda v: f"interval CV {v:.2f} ({'highly regular' if v < 0.15 else 'irregular'})",
    "resp_bytes_is_zero": lambda v: f"response bytes {'0' if v > 0.5 else 'nonzero'}",
    "dest_fanin": lambda v: f"destination fan-in {int(round(v))}",
    "periodicity_score": lambda v: f"periodicity score {v:.2f}",
    "orig_resp_byte_ratio": lambda v: f"upload/download ratio {v:.1f}",
}


def render_feature(name: str, value: float) -> str:
    """FR-7.3: e.g. 'interval CV 0.03 (highly regular)'."""
    if name in _FEATURE_RENDER:
        return _FEATURE_RENDER[name](value)
    return f"{name.replace('_', ' ')} {value:.3g}"


def _select_positive_class(shap_values):
    """Normalise across SHAP API versions: older releases return a list
    [class0_shap, class1_shap]; newer ones (>=0.5x) return a single
    (n_samples, n_features, n_classes) ndarray instead. Either way,
    downstream code wants one 2D (n_samples, n_features) array of
    positive-class attributions."""
    if isinstance(shap_values, list):
        return shap_values[1]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def compute_global_importance(model: Any, X_background: pd.DataFrame, X_eval: pd.DataFrame):
    """FR-7.1: SHAP global feature importance for the supervised branch.
    Uses TreeExplainer for tree models (fast, exact) and falls back to
    KernelExplainer's model-agnostic path otherwise."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_eval)
        shap_values = _select_positive_class(shap_values)
    except Exception:
        logger.info("TreeExplainer unavailable for this model; falling back to KernelExplainer")
        background = shap.sample(X_background, min(100, len(X_background)))
        explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], background)
        shap_values = explainer.shap_values(X_eval, nsamples=100)
        shap_values = _select_positive_class(shap_values)

    global_importance = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=X_eval.columns)
        .sort_values(ascending=False)
    )
    return shap_values, global_importance


def local_explanation(
    shap_values: np.ndarray, X_eval: pd.DataFrame, row_idx: int, top_k: int = 3
) -> list[tuple[str, float, float]]:
    """FR-7.2: top-k contributing features for one verdict, with signed
    SHAP direction. Returns [(feature_name, feature_value, shap_value), ...]."""
    row_shap = shap_values[row_idx]
    order = np.argsort(-np.abs(row_shap))[:top_k]
    return [
        (X_eval.columns[i], float(X_eval.iloc[row_idx, i]), float(row_shap[i]))
        for i in order
    ]


def render_local_explanation(
    shap_values: np.ndarray, X_eval: pd.DataFrame, row_idx: int, top_k: int = 3
) -> str:
    """FR-7.3: human-readable rendering of the top-k local explanation,
    e.g. 'interval CV 0.03 (highly regular) · response bytes 0 ·
    destination fan-in 1'."""
    parts = local_explanation(shap_values, X_eval, row_idx, top_k)
    return " · ".join(render_feature(name, value) for name, value, _shap in parts)


def compare_to_builtin_importance(model: Any, global_shap_importance: pd.Series) -> pd.DataFrame:
    """FR-7.4: compare SHAP ranking against a tree model's built-in
    feature_importances_ and flag disagreements in rank."""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            "compare_to_builtin_importance requires a tree model with "
            "feature_importances_ (e.g. the uncalibrated RandomForest "
            "before wrapping in CalibratedClassifierCV)"
        )
    builtin = pd.Series(model.feature_importances_, index=global_shap_importance.index)
    comparison = pd.DataFrame({
        "shap_importance": global_shap_importance,
        "shap_rank": global_shap_importance.rank(ascending=False),
        "builtin_importance": builtin,
        "builtin_rank": builtin.rank(ascending=False),
    })
    comparison["rank_disagreement"] = (comparison["shap_rank"] - comparison["builtin_rank"]).abs()
    return comparison.sort_values("rank_disagreement", ascending=False)
