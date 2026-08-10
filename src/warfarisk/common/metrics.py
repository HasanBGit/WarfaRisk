"""Metrics shared across phases, all stratifiable by a group column so Phase 8
("safety/fairness reporting, by design") is a formatting requirement on these
functions' output rather than a separate modeling step, per TRAINING_PLAN.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def pw20(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of predictions within +/-20% of the observed dose — the
    metric the review's own literature uses most, alongside MAE/R2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    within_20pct = np.abs(y_pred - y_true) <= 0.2 * np.abs(y_true)
    return float(np.mean(within_20pct))


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pw20": pw20(y_true, y_pred),
    }


def stratified_regression_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    group: pd.Series,
    min_group_n: int = 5,
) -> pd.DataFrame:
    """One row per subgroup (e.g. ancestry, sex, dose category) plus an
    'overall' row — the shape every Phase 3-7 results table should ship in,
    per Phase 8's per-subgroup reporting requirement. Groups smaller than
    min_group_n are flagged rather than silently reported (matches the
    review's own caution about small ancestry-subgroup sample sizes)."""
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred), "group": np.asarray(group)})
    rows = []
    overall = regression_report(df["y_true"], df["y_pred"])
    overall["group"] = "overall"
    overall["underpowered"] = False
    rows.append(overall)
    for group_name, sub in df.groupby("group"):
        report = regression_report(sub["y_true"], sub["y_pred"])
        report["group"] = group_name
        report["underpowered"] = report["n"] < min_group_n
        rows.append(report)
    return pd.DataFrame(rows)[["group", "n", "mae", "r2", "pw20", "underpowered"]]


def conformal_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage of a prediction interval — compare against the
    nominal level (e.g. 0.9 for a 90% interval) per Phase 5's verify step."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def stratified_conformal_coverage(
    y_true: pd.Series, lower: np.ndarray, upper: np.ndarray, group: pd.Series
) -> pd.DataFrame:
    df = pd.DataFrame(
        {"y_true": np.asarray(y_true), "lower": np.asarray(lower), "upper": np.asarray(upper), "group": np.asarray(group)}
    )
    rows = [{"group": "overall", "n": len(df), "coverage": conformal_coverage(df["y_true"], df["lower"], df["upper"])}]
    for group_name, sub in df.groupby("group"):
        rows.append(
            {"group": group_name, "n": len(sub), "coverage": conformal_coverage(sub["y_true"], sub["lower"], sub["upper"])}
        )
    return pd.DataFrame(rows)


def concordance_index(event_times: np.ndarray, predicted_risk: np.ndarray, event_observed: np.ndarray) -> float:
    """Harrell's C-index for Phase 7b's survival-analysis (time-to-therapeutic-INR)
    models, which don't produce a point dose/INR estimate so regression_report
    doesn't apply to them. Kept dependency-free (no lifelines/scikit-survival
    import) since it's a small O(n^2) pairwise comparison, fine at this
    project's cohort sizes (n up to a few hundred)."""
    event_times = np.asarray(event_times, dtype=float)
    predicted_risk = np.asarray(predicted_risk, dtype=float)
    event_observed = np.asarray(event_observed, dtype=bool)

    n = len(event_times)
    concordant = 0
    permissible = 0
    for i in range(n):
        if not event_observed[i]:
            continue
        for j in range(n):
            if i == j or event_times[j] <= event_times[i]:
                continue
            permissible += 1
            if predicted_risk[i] > predicted_risk[j]:
                concordant += 1
            elif predicted_risk[i] == predicted_risk[j]:
                concordant += 0.5
    return float(concordant / permissible) if permissible > 0 else float("nan")
