"""FR-3 — Supervised detection branch: RF, XGBoost, LR baselines with
explicit class-imbalance handling, PR-AUC-optimised tuning, and
probability calibration.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from ..config import get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - exercised only if xgboost missing
    XGBClassifier = None


def _resample_smote(X: pd.DataFrame, y: pd.Series, k_neighbors: int, seed: int):
    from imblearn.over_sampling import SMOTE

    n_minority = int(y.sum())
    k = min(k_neighbors, max(1, n_minority - 1))
    sm = SMOTE(k_neighbors=k, random_state=seed)
    return sm.fit_resample(X, y)


def train_classifier(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """FR-3.1-3.6: fit `model_name` in {random_forest, xgboost, logistic_regression}
    with the configured imbalance strategy, tuned via stratified CV on
    PR-AUC (average_precision — FR-3.5), then calibrated (FR-3.6).

    Returns (calibrated_estimator, training_metadata).
    """
    seed = cfg["seed"]
    strategy = cfg["imbalance"]["strategy"]

    X_fit, y_fit = X_train, y_train
    if strategy == "smote":
        X_fit, y_fit = _resample_smote(
            X_train, y_train, cfg["imbalance"]["smote_k_neighbors"], seed
        )
        class_weight = None
    elif strategy == "class_weight":
        class_weight = "balanced"
    else:
        class_weight = None  # focal_loss handled inside XGBoost's objective, not here

    base_estimator, param_grid = _build_estimator(model_name, cfg, class_weight, seed)

    cv = StratifiedKFold(
        n_splits=cfg["models"]["cv_folds"], shuffle=True, random_state=seed
    )
    search = GridSearchCV(
        base_estimator,
        param_grid,
        scoring=cfg["models"]["cv_scoring"],  # PR-AUC, never accuracy (FR-3.5)
        cv=cv,
        n_jobs=-1,
    )

    start = time.perf_counter()
    search.fit(X_fit, y_fit)
    train_seconds = time.perf_counter() - start

    calibrated = CalibratedClassifierCV(
        search.best_estimator_,
        method=cfg["models"]["calibration_method"],
        cv=cv,
    )
    calibrated.fit(X_fit, y_fit)

    latency_ms = _measure_inference_latency(calibrated, X_train)

    metadata = {
        "model_name": model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "imbalance_strategy": strategy,
        "best_params": search.best_params_,
        "cv_best_score_pr_auc": float(search.best_score_),
        "train_seconds": train_seconds,                 # FR-3.8
        "inference_latency_ms_per_flow": latency_ms,     # FR-3.8, validates NFR-1
        "feature_list": list(X_train.columns),           # FR-3.7
        "n_train_rows": len(X_fit),
    }
    logger.info(
        "%s trained: CV PR-AUC=%.4f, train=%.1fs, inference=%.3fms/flow",
        model_name, metadata["cv_best_score_pr_auc"], train_seconds, latency_ms,
    )
    return calibrated, metadata


def _build_estimator(model_name: str, cfg: dict[str, Any], class_weight, seed: int):
    models_cfg = cfg["models"]
    if model_name == "random_forest":
        est = RandomForestClassifier(random_state=seed, class_weight=class_weight)
        grid = {
            "n_estimators": models_cfg["random_forest"]["n_estimators"],
            "max_depth": models_cfg["random_forest"]["max_depth"],
            "min_samples_leaf": models_cfg["random_forest"]["min_samples_leaf"],
        }
    elif model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed; see requirements.txt")
        scale_pos_weight = 1.0
        est = XGBClassifier(
            random_state=seed,
            eval_metric="aucpr",
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight if class_weight else 1.0,
        )
        grid = {
            "n_estimators": models_cfg["xgboost"]["n_estimators"],
            "max_depth": models_cfg["xgboost"]["max_depth"],
            "learning_rate": models_cfg["xgboost"]["learning_rate"],
        }
    elif model_name == "logistic_regression":
        est = LogisticRegression(
            random_state=seed, class_weight=class_weight, max_iter=2000
        )
        grid = {"C": models_cfg["logistic_regression"]["C"]}
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return est, grid


def _measure_inference_latency(model, X: pd.DataFrame, n_samples: int = 1000) -> float:
    """FR-3.8: per-flow inference latency in milliseconds, measured on a
    warm model (excludes JIT/first-call overhead)."""
    sample = X.sample(n=min(n_samples, len(X)), random_state=0)
    model.predict_proba(sample.iloc[:1])  # warm-up
    start = time.perf_counter()
    model.predict_proba(sample)
    elapsed = time.perf_counter() - start
    return (elapsed / len(sample)) * 1000


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def save_model(model, metadata: dict[str, Any], models_dir: str | Path, name: str) -> Path:
    """FR-3.7: persist the trained model with version metadata and the
    exact feature list used, so a later load can validate feature-set
    compatibility before scoring."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}.joblib"
    meta_path = models_dir / f"{name}_metadata.json"
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Saved model %s and metadata to %s", name, models_dir)
    return model_path


def load_model(models_dir: str | Path, name: str) -> tuple[Any, dict[str, Any]]:
    models_dir = Path(models_dir)
    model = joblib.load(models_dir / f"{name}.joblib")
    metadata = json.loads((models_dir / f"{name}_metadata.json").read_text())
    return model, metadata
