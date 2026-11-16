"""Phase 4 — ancestry-stratified / leave-one-ancestry-out validation"""

from __future__ import annotations

import pandas as pd

from warfarisk.common import data as data_mod
from warfarisk.common.metrics import regression_report


def iwpc1780_ancestry_group(df: pd.DataFrame) -> pd.Series:
    """
    1780's ancestry signal is two binary flags (Black, Asian), not one categorical
    column like 6256's `Race (Reported)` — collapse into a single group label so
    both cohorts can share the same holdout loop below
    """

    def _label(row):
        if row["Black"] == 1:
            return "Black"
        if row["Asian"] == 1:
            return "Asian"
        return "Other/unspecified"

    return df.apply(_label, axis=1)


def leave_one_ancestry_out(
    df: pd.DataFrame,
    ancestry_col_or_series,
    feature_set_fn,
    target_col: str,
    model_builder,
    which: str = "combined",
    min_group_n: int = 20,
) -> pd.DataFrame:
    """
    For every ancestry group with at least min_group_n patients: fit on every OTHER
    group, test only on the held-out group, and separately fit a matched in-group
    80/20 internal control — reproduces the review's own Table 4 shape (held-out-
    group MAE next to a matched in-group control)
    """
    if isinstance(ancestry_col_or_series, str):
        ancestry = df[ancestry_col_or_series]
    else:
        ancestry = ancestry_col_or_series

    df = df.copy()
    df["_ancestry_group"] = ancestry.values

    group_counts = df["_ancestry_group"].value_counts()
    eligible_groups = group_counts[group_counts >= min_group_n].index.tolist()

    rows = []
    for group in eligible_groups:
        held_out = df[df["_ancestry_group"] == group]
        rest = df[df["_ancestry_group"] != group]

        X_train = feature_set_fn(rest, which)
        y_train = rest[target_col]
        X_test = feature_set_fn(held_out, which)
        y_test = held_out[target_col]

        pipeline = model_builder()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        report = regression_report(y_test, y_pred)
        report["evaluation"] = f"train_non_{group}__test_{group}"
        report["group"] = group
        rows.append(report)

        in_group = df[df["_ancestry_group"] == group].sample(frac=1.0, random_state=0)
        split_idx = int(len(in_group) * 0.8)
        in_train, in_test = in_group.iloc[:split_idx], in_group.iloc[split_idx:]
        if len(in_test) >= 5:
            X_in_train = feature_set_fn(in_train, which)
            X_in_test = feature_set_fn(in_test, which)
            control_pipeline = model_builder()
            control_pipeline.fit(X_in_train, in_train[target_col])
            y_in_pred = control_pipeline.predict(X_in_test)
            control_report = regression_report(in_test[target_col], y_in_pred)
            control_report["evaluation"] = f"{group}_internal_80_20_control"
            control_report["group"] = group
            rows.append(control_report)

    return pd.DataFrame(rows)


def run_phase4_iwpc6256(df: pd.DataFrame, model_builder, which: str = "combined") -> pd.DataFrame:
    return leave_one_ancestry_out(
        df,
        "Race (Reported)",
        data_mod.iwpc6256_feature_set,
        data_mod.IWPC6256_TARGET,
        model_builder,
        which=which,
    )


def run_phase4_iwpc1780(df: pd.DataFrame, model_builder, which: str = "combined") -> pd.DataFrame:
    return leave_one_ancestry_out(
        df,
        iwpc1780_ancestry_group(df),
        data_mod.iwpc1780_feature_set,
        data_mod.IWPC1780_TARGET,
        model_builder,
        which=which,
    )
