import numpy as np

from beacon_detection.decision import calibrate_thresholds, decide


def test_calibrate_thresholds_respects_alert_budget():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=100_000)
    tau_low, tau_high = calibrate_thresholds(
        scores, alerts_per_day_budget=100, assumed_daily_flow_volume=100_000
    )
    n_block = (scores >= tau_high).sum()
    # Should land close to the 100-alert budget (within sampling noise).
    assert 50 <= n_block <= 200
    assert tau_low <= tau_high


def test_decide_maps_risk_to_three_actions():
    scores = np.array([0.1, 0.5, 0.9])
    actions = decide(scores, tau_low=0.4, tau_high=0.8)
    assert list(actions) == ["ALLOW", "ALERT", "BLOCK"]
