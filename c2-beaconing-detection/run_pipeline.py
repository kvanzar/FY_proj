#!/usr/bin/env python
"""FR-8.9 — single-command entry point that reproducibly runs the full
Phase 1 C2 beaconing detection pipeline end to end: ingest -> sessionize ->
features -> supervised + anomaly branches -> fusion -> decision simulation ->
explanation -> adversarial sweep -> evaluation report.

Usage:
    python run_pipeline.py [--config config/default.yaml]

Requires config.data.raw_paths to point at a downloaded IoT-23 variant
that retains ts/uid/id.orig_h/id.resp_h (see README.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from beacon_detection.adversarial import jitter_sweep
from beacon_detection.config import get_logger, load_config
from beacon_detection.decision import build_decision_log, calibrate_thresholds, decide, save_decision_log, save_thresholds
from beacon_detection.evaluate import (
    compute_confusion_matrix,
    compute_metrics,
    false_positives_per_day,
    held_out_family_split,
    random_stratified_split,
    run_ablation,
    save_report,
    temporal_split,
)
from beacon_detection.explain import compute_global_importance, render_local_explanation
from beacon_detection.features import connstate
from beacon_detection.models import anomaly as anomaly_model
from beacon_detection.models import fusion as fusion_model
from beacon_detection.models import supervised as supervised_model
from beacon_detection.pipeline import build_feature_matrix, load_and_prepare, select_tiers
from beacon_detection.sessionize import build_session_id

logger = get_logger("run_pipeline")


def _train_and_evaluate_split(
    split_name: str,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Trains a fresh supervised + anomaly branch on this split's training
    data, fuses, and computes the full FR-8.1-8.5 metric suite. Used for
    all three splitting strategies (§8.1: 'all three must be reported').
    """
    sup_model, sup_meta = supervised_model.train_classifier("random_forest", X_train, y_train, cfg)
    sup_scores = supervised_model.predict_proba(sup_model, X_test)

    X_train_benign = X_train[y_train == 0]
    ae_model, ae_scaler, ae_meta = anomaly_model.train_autoencoder(X_train_benign, cfg)
    raw_errors = anomaly_model.score(ae_model, ae_scaler, X_test)
    anomaly_scores = anomaly_model.normalise_score(raw_errors, ae_meta["threshold_raw_error"])

    risk = fusion_model.fuse(sup_scores, anomaly_scores, cfg)
    tau_low, tau_high = calibrate_thresholds(
        risk, cfg["decision"]["alerts_per_day_budget"], cfg["decision"]["assumed_daily_flow_volume"]
    )
    metrics = compute_metrics(y_test.to_numpy(), risk, tau_high)
    metrics["false_positives_per_day"] = false_positives_per_day(
        y_test.to_numpy(), risk, tau_high, cfg["decision"]["assumed_daily_flow_volume"]
    )
    # FR-4.5/AC-4: anomaly-branch recall in isolation, the branch's whole
    # justification on the held-out-family split, but reported for every
    # split so the supervised/anomaly contributions can be compared directly.
    anomaly_only_metrics = compute_metrics(y_test.to_numpy(), anomaly_scores, 0.5)

    logger.info(
        "[%s split] PR-AUC=%.4f, recall@thresh=%.3f, anomaly-only PR-AUC=%.4f",
        split_name, metrics["pr_auc"], metrics["recall_at_threshold"], anomaly_only_metrics["pr_auc"],
    )
    return {
        "supervised_metadata": sup_meta,
        "anomaly_metadata": ae_meta,
        "thresholds": {"tau_low": tau_low, "tau_high": tau_high},
        "metrics": metrics,
        "anomaly_branch_only_metrics": anomaly_only_metrics,
        "confusion_matrix": compute_confusion_matrix(y_test.to_numpy(), risk, tau_high).tolist(),
        "model": sup_model,
        "anomaly_model": ae_model,
        "supervised_scores": sup_scores,
        "anomaly_scores": anomaly_scores,
        "risk": risk,
    }


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    outputs = Path(cfg["paths"]["outputs_dir"])

    logger.info("=== Stage 1/8: ingest ===")
    flows = load_and_prepare(cfg)

    logger.info("=== Stage 2/8: sessionize + feature matrix ===")
    X_raw, y, meta = build_feature_matrix(flows, cfg)

    logger.info("=== Stage 3/8: categorical encoding + splits ===")
    # FR-2.6.5: fit the categorical vocabulary on the random split's
    # training portion only, then apply it once so all three splitting
    # strategies below share one feature space. (Zeek's conn_state/proto/
    # service vocabulary is a small, closed protocol-level set — reusing
    # one fit across splits does not carry the leakage risk a per-flow
    # numeric scaler would.)
    prelim = random_stratified_split(X_raw, y, cfg["evaluation"]["split_ratios"], cfg["seed"])
    X_train_raw, _ = prelim["train"]
    _, (X,) = connstate.encode_categoricals(X_train_raw, [X_raw])

    splits_random = random_stratified_split(X, y, cfg["evaluation"]["split_ratios"], cfg["seed"])
    splits_temporal = temporal_split(X, y, meta, cfg["evaluation"]["split_ratios"])
    splits_family = held_out_family_split(
        X, y, meta["detailed_label"], cfg["evaluation"]["held_out_family_test_fraction"], cfg["seed"]
    )

    X_train, y_train = splits_random["train"]
    X_test, y_test = splits_random["test"]

    logger.info("=== Stage 4-6/8: supervised + anomaly branches, fusion, decision (random split) ===")
    primary = _train_and_evaluate_split("random", X_train, y_train, X_test, y_test, cfg)
    sup_model = primary["model"]
    supervised_model.save_model(sup_model, primary["supervised_metadata"], cfg["paths"]["models_dir"], "random_forest")

    tau_low, tau_high = primary["thresholds"]["tau_low"], primary["thresholds"]["tau_high"]
    save_thresholds(tau_low, tau_high, outputs / "logs" / "thresholds.json")
    actions = decide(primary["risk"], tau_low, tau_high)

    logger.info("=== Stage 7/8: explanation ===")
    tree_model = (
        sup_model.calibrated_classifiers_[0].estimator
        if hasattr(sup_model, "calibrated_classifiers_") else sup_model
    )
    shap_values, global_importance = compute_global_importance(
        tree_model, X_train.sample(min(200, len(X_train)), random_state=cfg["seed"]), X_test,
    )
    explanations = [render_local_explanation(shap_values, X_test, i) for i in range(len(X_test))]

    decision_log = build_decision_log(
        meta.loc[X_test.index], primary["risk"], primary["supervised_scores"], primary["anomaly_scores"],
        actions, explanations,
    )
    save_decision_log(decision_log, outputs / "logs" / "decision_log.csv")

    logger.info("=== Stage 8/8: temporal + held-out-family evaluation, ablation, adversarial sweep ===")
    X_train_t, y_train_t = splits_temporal["train"]
    X_test_t, y_test_t = splits_temporal["test"]
    temporal_result = _train_and_evaluate_split("temporal", X_train_t, y_train_t, X_test_t, y_test_t, cfg)

    X_train_f, y_train_f = splits_family["train"]
    X_test_f, y_test_f = splits_family["test"]
    family_result = _train_and_evaluate_split("held-out-family", X_train_f, y_train_f, X_test_f, y_test_f, cfg)

    def _fit_and_score(Xtr, ytr, Xte):
        m, _ = supervised_model.train_classifier("random_forest", Xtr, ytr, cfg)
        return supervised_model.predict_proba(m, Xte)

    ablation = run_ablation(
        cfg["evaluation"]["ablation_tiers"], _fit_and_score, select_tiers,
        X_train, y_train, X_test, y_test,
    )

    flows_positive = flows[flows.get("target", 0) == 1] if "target" in flows.columns else flows.iloc[0:0]
    if len(flows_positive) > 0:
        # inject_jitter groups by session_id, which normally isn't assigned
        # until sessionize.py runs inside build_feature_matrix — add it here
        # so jitter can be applied per-conversation before each sweep step
        # rebuilds features from scratch via _rebuild_and_score below.
        flows_positive = build_session_id(flows_positive, cfg["sessionize"]["group_keys"])

    def _rebuild_and_score(jittered_flows):
        Xj_raw, yj, _ = build_feature_matrix(jittered_flows, cfg)
        _, (Xj,) = connstate.encode_categoricals(X_train_raw, [Xj_raw])
        scores = supervised_model.predict_proba(sup_model, Xj)
        return yj.to_numpy(), scores

    if len(flows_positive) > 0:
        # _rebuild_and_score scores with the supervised branch alone (it
        # doesn't have access to a fitted anomaly-branch scaler here), so
        # the fixed threshold for recall_at_threshold must come from that
        # same supervised-only score distribution on real, unperturbed
        # data — not the fused-risk tau_high, which lives on a different
        # scale. See adversarial.recall_at_threshold's docstring for why
        # this threshold can't be derived from the jitter sweep itself.
        _, sup_only_tau_high = calibrate_thresholds(
            primary["supervised_scores"], cfg["decision"]["alerts_per_day_budget"],
            cfg["decision"]["assumed_daily_flow_volume"],
        )
        jitter_curve = jitter_sweep(
            flows_positive, _rebuild_and_score, cfg["adversarial"]["jitter_percentages"], sup_only_tau_high
        )
    else:
        jitter_curve = None

    def _strip_non_serialisable(result: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in result.items() if k not in {"model", "anomaly_model", "supervised_scores", "anomaly_scores", "risk"}}

    report = {
        "random_split": _strip_non_serialisable(primary),
        "temporal_split": _strip_non_serialisable(temporal_result),
        "held_out_family_split": {
            **_strip_non_serialisable(family_result),
            "held_out_families": splits_family.get("held_out_families"),
        },
        "fusion_comparison": fusion_model.compare_fusion_methods(
            primary["supervised_scores"], primary["anomaly_scores"], y_test.to_numpy(), cfg
        ),
        "ablation": ablation.to_dict(orient="records"),
        "global_shap_importance": global_importance.to_dict(),
        "jitter_degradation_curve": jitter_curve.to_dict(orient="records") if jitter_curve is not None else None,
    }
    save_report(report, outputs / "logs" / "evaluation_report.json")
    logger.info("Pipeline complete. See %s for the full report.", outputs / "logs" / "evaluation_report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    main(args.config)
