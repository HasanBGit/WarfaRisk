"""Exact per-ancestry binomial confidence intervals for MAPIE coverage,
plus a multiplicity-adjusted test against the 90% nominal target.

Reuses the positional-recovery approach from check_mapie_ancestry_width.py:
phase5_iwpc6256_mapie_predictions.csv has no ancestry column, but its row
order matches the fixed IWPC-6256 test split exactly (verified there by
row-for-row y_true comparison), so the ancestry label can be reattached by
position.

For each of the 18 ancestry labels in the test split, this computes the
exact (Clopper-Pearson) 95% CI on MAPIE coverage via scipy's binomtest, and
a two-sided exact binomial test of H0: coverage = 0.90. Labels with n < 10
are excluded from testing (point estimate and CI only) since that is the
same cutoff already used elsewhere in this project for Mondrian pooling;
testing a group with, e.g., n = 1 is not statistically meaningful. The
n >= 10 group's p-values are then adjusted with Benjamini-Hochberg FDR
control, implemented directly (statsmodels is not installed in this
environment) since it is a five-line, well-defined algorithm.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import binomtest

from warfarisk.common import data as data_mod
from warfarisk.common import splits

NOMINAL_COVERAGE = 0.90
MIN_N_FOR_TEST = 10


def load_mapie_predictions_with_ancestry() -> pd.DataFrame:
    split = splits.load_split("iwpc_6256")
    df = data_mod.load_iwpc_6256()
    _, test_df = splits.apply_split(df, split)
    group_test = test_df["Race (Reported)"].to_numpy()

    predictions = pd.read_csv("phase5_iwpc6256_mapie_predictions.csv")
    if not (predictions["y_true"].to_numpy() == test_df[data_mod.IWPC6256_TARGET].to_numpy()).all():
        raise ValueError(
            "Row alignment check failed: this file's y_true does not match a fresh load "
            "of the fixed test split. Do not attach ancestry labels positionally until "
            "this is resolved."
        )
    predictions["group"] = group_test
    return predictions


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adjusted = [0.0] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = n - rank + 1
        val = min(prev, pvals[i] * n / k)
        adjusted[i] = val
        prev = val
    return adjusted


def main() -> None:
    df = load_mapie_predictions_with_ancestry()
    df["covered"] = (df["y_true"] >= df["lower"]) & (df["y_true"] <= df["upper"])

    rows = []
    for g, sub in df.groupby("group"):
        n = len(sub)
        covered = int(sub["covered"].sum())
        result = binomtest(covered, n, p=NOMINAL_COVERAGE, alternative="two-sided")
        ci = result.proportion_ci(confidence_level=0.95, method="exact")
        rows.append(
            {
                "ancestry_group": g,
                "n": n,
                "coverage_pct": round(100 * covered / n, 1),
                "ci95_low_pct": round(100 * ci.low, 1),
                "ci95_high_pct": round(100 * ci.high, 1),
                "p_value": result.pvalue,
                "tested": n >= MIN_N_FOR_TEST,
            }
        )

    result_df = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)

    tested_mask = result_df["tested"]
    adj = benjamini_hochberg(result_df.loc[tested_mask, "p_value"].tolist())
    result_df["p_value_bh"] = pd.NA
    result_df.loc[tested_mask, "p_value_bh"] = adj
    result_df["significant_bh_0.05"] = pd.NA
    result_df.loc[tested_mask, "significant_bh_0.05"] = result_df.loc[tested_mask, "p_value_bh"] < 0.05

    pd.set_option("display.width", 120)
    print(result_df.to_string(index=False))

    result_df.to_csv("check_mapie_ancestry_binomial_ci_results.csv", index=False)
    print("\nWrote check_mapie_ancestry_binomial_ci_results.csv")

    n_tested = int(tested_mask.sum())
    n_sig = int(result_df.loc[tested_mask, "significant_bh_0.05"].sum())
    print(f"\n{n_tested} labels tested (n >= {MIN_N_FOR_TEST}); {n_sig} significant after Benjamini-Hochberg at 0.05.")


if __name__ == "__main__":
    main()
