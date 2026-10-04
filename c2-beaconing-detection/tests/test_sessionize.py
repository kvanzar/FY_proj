from beacon_detection.sessionize import sessionize


def test_sessionize_groups_by_host_pair_and_port(synthetic_flows):
    out, discard_rate = sessionize(
        synthetic_flows, group_keys=["id.orig_h", "id.resp_h", "id.resp_p"], min_flows=4
    )
    assert out["session_id"].nunique() == 2
    assert discard_rate == 0.0


def test_sessionize_discards_short_sessions(synthetic_flows):
    # Drop all but 2 flows of the beacon session, leaving it under min_flows.
    short = synthetic_flows[~synthetic_flows["uid"].isin(["Cbeacon2", "Cbeacon3", "Cbeacon4", "Cbeacon5"])]
    out, discard_rate = sessionize(
        short, group_keys=["id.orig_h", "id.resp_h", "id.resp_p"], min_flows=4
    )
    assert out["session_id"].nunique() == 1
    assert discard_rate > 0


def test_intervals_sorted_ascending_within_session(synthetic_flows):
    out, _ = sessionize(
        synthetic_flows, group_keys=["id.orig_h", "id.resp_h", "id.resp_p"], min_flows=4
    )
    for _, group in out.groupby("session_id"):
        assert group["ts"].is_monotonic_increasing
