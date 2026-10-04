import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_flows() -> pd.DataFrame:
    """8 flows across 2 sessions: a perfectly regular 60s beacon
    (host A -> C2, heartbeat-shaped: near-zero response bytes) and an
    irregular benign session (host B -> server, human-timed browsing).
    """
    rng = np.random.default_rng(0)
    base_ts = 1_600_000_000.0

    beacon_ts = base_ts + np.arange(6) * 60.0  # exactly periodic
    beacon = pd.DataFrame({
        "ts": beacon_ts,
        "uid": [f"Cbeacon{i}" for i in range(6)],
        "id.orig_h": "10.0.0.5",
        "id.orig_p": rng.integers(49152, 65535, size=6),
        "id.resp_h": "203.0.113.9",
        "id.resp_p": 443,
        "proto": "tcp",
        "service": np.nan,
        "duration": 0.05,
        "orig_bytes": 120,
        "resp_bytes": 0,
        "conn_state": "SF",
        "history": "ShAdD",
        "orig_pkts": 3,
        "orig_ip_bytes": 180,
        "resp_pkts": 1,
        "resp_ip_bytes": 40,
        "label": "Malicious",
        "detailed-label": "C&C-HeartBeat",
        "capture_id": "capture-1",
    })

    benign_gaps = rng.uniform(5, 4000, size=5)
    benign_ts = base_ts + np.concatenate([[0], np.cumsum(benign_gaps)])
    benign = pd.DataFrame({
        "ts": benign_ts,
        "uid": [f"Cbenign{i}" for i in range(6)],
        "id.orig_h": "10.0.0.9",
        "id.orig_p": rng.integers(49152, 65535, size=6),
        "id.resp_h": "198.51.100.4",
        "id.resp_p": 443,
        "proto": "tcp",
        "service": "ssl",
        "duration": rng.uniform(0.5, 5, size=6),
        "orig_bytes": rng.integers(500, 5000, size=6),
        "resp_bytes": rng.integers(2000, 50000, size=6),
        "conn_state": "SF",
        "history": "ShADadFf",
        "orig_pkts": rng.integers(5, 30, size=6),
        "orig_ip_bytes": rng.integers(600, 5200, size=6),
        "resp_pkts": rng.integers(5, 40, size=6),
        "resp_ip_bytes": rng.integers(2100, 51000, size=6),
        "label": "Benign",
        "detailed-label": "Benign",
        "capture_id": "capture-1",
    })

    return pd.concat([beacon, benign], ignore_index=True)
