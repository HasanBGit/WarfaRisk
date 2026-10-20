"""Paired comparison of the IWPC published equation against the stacking
ensemble on IWPC-6256's standard 80/20 test set (n = 1,207).

Background: a reviewer asked for a paired confidence interval, dose-stratified
errors, and the proportion of clinically unacceptable errors for the
manuscript's headline MAE improvement (8.55 mg/week, AutoGluon, vs. 9.18
mg/week, IWPC equation). Per-patient predictions were never retained for the
AutoGluon ablation run, so that specific pair cannot be reconstructed. This
script instead does the same paired analysis against the stacking ensemble,
the only architecture for which per-patient predictions on this exact test
set were retained (from the Phase 5 MAPIE/Mondrian benchmarking run). This is
a genuinely different, weaker comparison (stacking ensemble MAE = 9.127, not
8.55) and must not be read as validating the headline AutoGluon result.

Row alignment: the retained predictions file has no patient identifier, only
positional order. Verified directly before running any comparison: a fresh
load of the same fixed test split (common.splits.load_split("iwpc_6256"))
produces y_true values that match the retained file's y_true column exactly,
position for position, across all 1,207 rows.

Outputs check_iwpc_vs_stacking_paired_results.csv with:
- Paired 95% bootstrap CI (10,000 resamples, seed 20260725) for the mean
  difference in absolute error (IWPC equation minus stacking ensemble).
- A paired Wilcoxon signed-rank test on absolute errors.
- Dose-stratified MAE (tertiles of true weekly dose) for both methods.
- Proportion of "clinically unacceptable" errors (predictions outside the
  same 20% PW20 threshold already used throughout this manuscript) for both
  methods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from warfarisk.common import data as data_mod
from warfarisk.common import splits
from warfarisk.phase1_baselines import iwpc_published_equation_predict

RANDOM_STATE = 20260725
N_BOOTSTRAP = 10_000


def load_aligned_predictions() -> pd.DataFrame:
    split = splits.load_split("iwpc_6256")
    df = data_mod.load_iwpc_6256()
    _, test_df = splits.apply_split(df, split)
    y_test = test_df[data_mod.IWPC6256_TARGET].to_numpy()

    retained = pd.read_csv("phase5_iwpc6256_mondrian_vs_mapie_predictions.csv")
    if not np.array_equal(y_test, retained["y_true"].to_numpy()):
        raise ValueError(
            "Row alignment check failed: a fresh load of the fixed test split does not "
            "match the retained predictions file's y_true column. Do not proceed with a "
            "positional paired comparison until this is resolved."
        )

    iwpc_pred = iwpc_published_equation_predict(test_df)
    return pd.DataFrame(
        {
            "y_true": y_test,
            "iwpc_pred": iwpc_pred,
            "stacking_pred": retained["mapie_pred"].to_numpy(),
        }
    )


def paired_bootstrap_ci(abs_err_a: np.ndarray, abs_err_b: np.ndarray, random_state: int = RANDOM_STATE) -> tuple[float, float, float]:
    """95% bootstrap CI for the mean paired difference (a - b), resampling
    patients (not errors independently) to preserve pairing."""
    diff = abs_err_a - abs_err_b
    point = float(diff.mean())
    rng = np.random.RandomState(random_state)
    n = len(diff)
    boot_means = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        sample_idx = rng.randint(0, n, n)
        boot_means[i] = diff[sample_idx].mean()
    boot_means.sort()
    lo = boot_means[int(0.025 * N_BOOTSTRAP)]
    hi = boot_means[int(0.975 * N_BOOTSTRAP) - 1]
    return point, lo, hi


def dose_tertile_mae(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    tertiles = pd.qcut(y_true, 3, labels=["low", "medium", "high"])
    err = np.abs(y_true - y_pred)
    return pd.DataFrame({"tertile": tertiles, "abs_err": err}).groupby("tertile", observed=True)["abs_err"].agg(["mean", "count"])


def unacceptable_error_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.20) -> float:
    """Proportion of predictions falling outside the same +/-20% PW20
    threshold already used throughout this manuscript (1 - PW20)."""
    within = np.abs(y_true - y_pred) <= threshold * y_true
    return float(1.0 - within.mean())


def main() -> None:
    df = load_aligned_predictions()
    y_true = df["y_true"].to_numpy()
    iwpc_err = np.abs(y_true - df["iwpc_pred"].to_numpy())
    stacking_err = np.abs(y_true - df["stacking_pred"].to_numpy())

    print(f"n = {len(df)}")
    print(f"IWPC equation MAE:     {iwpc_err.mean():.3f}")
    print(f"Stacking ensemble MAE: {stacking_err.mean():.3f}")
    print()

    point, lo, hi = paired_bootstrap_ci(iwpc_err, stacking_err)
    print(f"Paired difference (IWPC - stacking), mean absolute error: {point:.3f} mg/week")
    print(f"95% bootstrap CI: [{lo:.3f}, {hi:.3f}]")

    stat, p = wilcoxon(iwpc_err, stacking_err)
    print(f"Wilcoxon signed-rank test: statistic={stat:.1f}, p={p:.4f}")
    print()

    print("Dose-stratified MAE (tertiles of true weekly dose):")
    print("  IWPC equation:")
    print(dose_tertile_mae(y_true, df["iwpc_pred"].to_numpy()).to_string())
    print("  Stacking ensemble:")
    print(dose_tertile_mae(y_true, df["stacking_pred"].to_numpy()).to_string())
    print()

    iwpc_unacceptable = unacceptable_error_rate(y_true, df["iwpc_pred"].to_numpy())
    stacking_unacceptable = unacceptable_error_rate(y_true, df["stacking_pred"].to_numpy())
    print(f"Proportion of predictions outside +/-20% of true dose:")
    print(f"  IWPC equation:     {iwpc_unacceptable * 100:.1f}%")
    print(f"  Stacking ensemble: {stacking_unacceptable * 100:.1f}%")

    summary = pd.DataFrame(
        [
            {"metric": "mae", "iwpc_equation": round(iwpc_err.mean(), 3), "stacking_ensemble": round(stacking_err.mean(), 3)},
            {"metric": "paired_diff_mean", "iwpc_equation": round(point, 3), "stacking_ensemble": None},
            {"metric": "paired_diff_95ci_lo", "iwpc_equation": round(lo, 3), "stacking_ensemble": None},
            {"metric": "paired_diff_95ci_hi", "iwpc_equation": round(hi, 3), "stacking_ensemble": None},
            {"metric": "wilcoxon_p", "iwpc_equation": round(p, 4), "stacking_ensemble": None},
            {"metric": "outside_pw20_pct", "iwpc_equation": round(iwpc_unacceptable * 100, 1), "stacking_ensemble": round(stacking_unacceptable * 100, 1)},
        ]
    )
    summary.to_csv("check_iwpc_vs_stacking_paired_results.csv", index=False)
    print("\nWrote check_iwpc_vs_stacking_paired_results.csv")


if __name__ == "__main__":
    main()
