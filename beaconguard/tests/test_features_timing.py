from src.features import timing
from src.sessionize import sessionize


def test_perfectly_regular_beacon_has_low_iat_cv(synthetic_flows):
    sessionized, _ = sessionize(
        synthetic_flows, group_keys=["id.orig_h", "id.resp_h", "id.resp_p"], min_flows=4
    )
    feats = timing.build(sessionized)
    assert feats.shape[0] == 2

    beacon_row = feats[feats["iat_mean"].round(0) == 60]
    assert len(beacon_row) == 1
    # A perfectly periodic 60s beacon should have ~0 std/CV.
    assert beacon_row["iat_cv"].iloc[0] < 0.01
    assert beacon_row["periodicity_score"].iloc[0] > 0.5


def test_irregular_session_has_higher_cv_than_beacon(synthetic_flows):
    sessionized, _ = sessionize(
        synthetic_flows, group_keys=["id.orig_h", "id.resp_h", "id.resp_p"], min_flows=4
    )
    feats = timing.build(sessionized)
    assert feats["iat_cv"].max() > feats["iat_cv"].min()


def test_session_flow_count_matches_group_size(synthetic_flows):
    sessionized, _ = sessionize(
        synthetic_flows, group_keys=["id.orig_h", "id.resp_h", "id.resp_p"], min_flows=4
    )
    feats = timing.build(sessionized)
    assert (feats["session_flow_count"] == 6).all()
