"""Phase 2 — leakage-audit assertions, used by phase2_leakage_audit.py and
callable directly from any phase file before it fits a model.

Two distinct failure modes this guards against, per TRAINING_PLAN.md Phase 2:
  1. Outcome leakage: an INR/Target-INR column reaching X for a *dose* target
     (INR is the response in some framings, not a legitimate predictor of dose).
  2. Split leakage: the same patient/stay appearing in both train and test
     because a longitudinal dataset was split by row instead of by ID.
"""

from __future__ import annotations

import pandas as pd

from . import data as data_mod


class LeakageError(AssertionError):
    """Raised when a leakage check fails. Subclasses AssertionError so it's
    still catchable by generic test-runner assertion handling."""


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
