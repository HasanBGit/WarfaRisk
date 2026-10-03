"""Per-ancestry SHAP feature importance, IWPC-1780, stacking ensemble.

Why this file exists: `phase6_explainability.shap_by_ancestry_subgroup`
was written as part of this project's original Phase 6 design (see that
module's docstring) but the driver script that produced the manuscript's
published SHAP table (`run_phase456_iwpc1780_local.py`) only ever called
the unstratified `shap_feature_importance`. The per-ancestry function was
built but never run. This script runs it, using the identical split and
model fit as the published result, to check whether SHAP attribution
(not just point accuracy or calibration) is uniform across ancestry
groups on IWPC-1780.

IWPC-1780's only ancestry signal is two binary flags, collapsed by
`iwpc1780_ancestry_group` into Black / Asian / Other-unspecified (coarser
than IWPC-6256's 18-label field; see that function's own docstring).

Raw IWPC-1780 data is not bundled in this repo (per the Data Availability
Statement, raw patient-level data is excluded from the public release) and
is read from the private local working copy below.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from warfarisk.common import data as data_mod
from warfarisk.common import splits
from warfarisk.phase3_genetics_ablation import build_stacking_ensemble
from warfarisk.phase4_ancestry_holdout import iwpc1780_ancestry_group
from warfarisk.phase6_explainability import shap_by_ancestry_subgroup, shap_feature_importance

RAW_IWPC_1780 = Path(
    "/Users/hassan/Documents/Work/SURE-AIC-Research/Warfin/SURE-Program/Models-Development"
    "/data/eda_notebooks/cleaned/iwpc_1780_clean.csv"
)
OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "phase6_iwpc1780_shap_by_ancestry.csv"


def main() -> None:
    df = pd.read_csv(RAW_IWPC_1780)
    split = splits.load_split("iwpc_1780")
    train_df, test_df = splits.apply_split(df, split)

    X_train = data_mod.iwpc1780_feature_set(train_df, "combined")
    y_train = train_df[data_mod.IWPC1780_TARGET]
    X_test = data_mod.iwpc1780_feature_set(test_df, "combined")

    print(f"Fitting stacking_ensemble[combined] on {X_train.shape} train rows...", flush=True)
    pipeline = build_stacking_ensemble()
    pipeline.fit(X_train, y_train)

    # Sanity check: unstratified overall importance should match the already-published
    # phase6_iwpc1780_stacking_ensemble_shap.csv (same pipeline, same split, same seed).
    overall = shap_feature_importance(pipeline, X_test, sample_size=200)
    top_feature, top_value = overall.iloc[0]["feature"], overall.iloc[0]["mean_abs_shap"]
    print(f"Sanity check, overall top feature: {top_feature} = {top_value:.3f} (published: VKORC1.AA = 8.110)", flush=True)

    group = iwpc1780_ancestry_group(test_df)
    print("Ancestry group sizes in test set:", flush=True)
    print(group.value_counts().to_string(), flush=True)

    print("\nComputing SHAP per ancestry subgroup...", flush=True)
    by_group = shap_by_ancestry_subgroup(pipeline, X_test, group, sample_size=200)
    print(by_group.to_string(), flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    by_group.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
