"""FR-2.5 — Tier 4 (partial) context features derivable from the dataset
alone, without decryption or external enrichment (DNS/WHOIS/reputation
are Phase 2, see tls_stub.py and PRD §13.3).

These are aggregate counts, not raw identities, so they do not violate
FR-2.6.1 (no raw IP addresses in the feature set): the model never sees
*which* host is contacting a destination, only *how many* distinct hosts
do, which is exactly the identity-agnostic design PETNet requires.
"""
from __future__ import annotations

import pandas as pd

TIER = 4


def build(df: pd.DataFrame) -> pd.DataFrame:
    """df must be the flow-level, sessionized frame (has session_id,
    id.orig_h, id.resp_h). Fan-in/fan-out/flow-count are computed over
    the full capture, then broadcast to one row per session_id.
    """
    dest_fanin = df.groupby("id.resp_h")["id.orig_h"].nunique().rename("dest_fanin")
    src_fanout = df.groupby("id.orig_h")["id.resp_h"].nunique().rename("src_fanout")
    dest_flow_count = df.groupby("id.resp_h").size().rename("dest_flow_count")

    session_keys = df.groupby("session_id", sort=False)[["id.orig_h", "id.resp_h"]].first()
    out = session_keys.join(dest_fanin, on="id.resp_h")
    out = out.join(src_fanout, on="id.orig_h")
    out = out.join(dest_flow_count, on="id.resp_h")
    return out[["dest_fanin", "src_fanout", "dest_flow_count"]].astype(float)


TIER4_COLUMNS = ["dest_fanin", "src_fanout", "dest_flow_count"]
