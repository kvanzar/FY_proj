from beacon_detection.features import volume
from beacon_detection.sessionize import sessionize


def test_heartbeat_session_has_resp_bytes_is_zero_flag(synthetic_flows):
    sessionized, _ = sessionize(
        synthetic_flows, group_keys=["id.orig_h", "id.resp_h", "id.resp_p"], min_flows=4
    )
    feats = volume.build(sessionized)
    beacon_session = sessionized[sessionized["detailed-label"] == "C&C-HeartBeat"]["session_id"].iloc[0]
    assert feats.loc[beacon_session, "resp_bytes_is_zero"] == 1.0


def test_benign_session_has_higher_byte_ratio_variance(synthetic_flows):
    sessionized, _ = sessionize(
        synthetic_flows, group_keys=["id.orig_h", "id.resp_h", "id.resp_p"], min_flows=4
    )
    feats = volume.build(sessionized)
    beacon_session = sessionized[sessionized["detailed-label"] == "C&C-HeartBeat"]["session_id"].iloc[0]
    benign_session = sessionized[sessionized["detailed-label"] == "Benign"]["session_id"].iloc[0]
    # The beacon's flows are near-identical in size; benign flows vary.
    assert feats.loc[beacon_session, "session_byte_variance"] < feats.loc[benign_session, "session_byte_variance"]


def test_log_transform_handles_zero_without_error():
    import pandas as pd

    df = pd.DataFrame({"total_bytes": [0, 10, 1000]})
    out = volume.log_transform(df, ["total_bytes"])
    assert (out["total_bytes"] >= 0).all()
    assert out["total_bytes"].iloc[0] == 0.0
