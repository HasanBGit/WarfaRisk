"""
Phase 2 — leakage-audit assertions, used by phase2_leakage_audit.py and callable
directly from any phase file before it fits a model
"""

from __future__ import annotations

import pandas as pd

from . import data as data_mod


class LeakageError(AssertionError):
    """Raised when a leakage check fails"""


def assert_no_leakage_columns(X: pd.DataFrame, forbidden_cols: list[str], context: str) -> None:
    present = [c for c in forbidden_cols if c in X.columns]
    if present:
        raise LeakageError(
            f"[{context}] Leakage columns present in feature matrix: {present}. "
            "These are downstream of / derived from the prediction target and must be dropped."
        )


def assert_disjoint_ids(train_ids, test_ids, context: str) -> None:
    train_set, test_set = set(train_ids), set(test_ids)
    overlap = train_set & test_set
    if overlap:
        raise LeakageError(
            f"[{context}] {len(overlap)} ID(s) appear in both train and test: "
            f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
        )


def audit_iwpc6256(X: pd.DataFrame) -> None:
    assert_no_leakage_columns(X, data_mod.IWPC6256_LEAKAGE_COLS, context="iwpc_6256")


def audit_iwpc1780(X: pd.DataFrame) -> None:
    assert_no_leakage_columns(X, data_mod.IWPC1780_LEAKAGE_COLS, context="iwpc_1780")
