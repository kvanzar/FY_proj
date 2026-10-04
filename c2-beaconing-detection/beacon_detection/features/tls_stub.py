"""Tier 3 — TLS handshake metadata. Phase 2 placeholder (PRD §13.1 / FS-1).

IoT-23's Kaggle distribution ships conn.log only; ssl.log/x509.log are not
present, so Tier 3 features cannot be validated against ground truth in
Phase 1 (PRD §4.6). This module defines the intended interface and schema
so the ablation harness (FR-8.8) and downstream code can reason about
Tier 3 without it existing yet, and so Phase 2 has a concrete contract to
implement against rather than starting from a blank page.
"""
from __future__ import annotations

import pandas as pd

TIER = 3

# Columns Phase 2 must populate, joined from ssl.log/x509.log on Zeek `uid`.
# All are *property* features (never certificate/fingerprint *identities*)
# per the PETNet identity-agnostic constraint (FR-2.6.1, PRD §13.1).
TIER3_COLUMNS = [
    "ja3_known_client",       # bool: JA3 hash matches a known-benign client library
    "cipher_suite_count",     # int: number of cipher suites offered
    "tls_version",            # categorical: negotiated TLS version
    "extension_count",        # int: number of TLS extensions offered
    "cert_validity_days",     # int: certificate validity period
    "cert_is_self_signed",    # bool
    "cert_subject_cn_entropy",  # float: Shannon entropy of the subject CN
]


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Not implemented in Phase 1. Raises so a misconfigured ablation run
    (e.g. including 'tier3' in evaluation.ablation_tiers) fails loudly
    rather than silently producing an empty/garbage feature block."""
    raise NotImplementedError(
        "Tier 3 (TLS metadata) requires ssl.log/x509.log, unavailable in "
        "the Phase 1 IoT-23 Kaggle distribution. See PRD §13.1 (FS-1). "
        "Do not add 'tier3' to evaluation.ablation_tiers until Phase 2."
    )
