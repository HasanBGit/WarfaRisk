"""Per-ancestry SHAP feature importance, IWPC-6256, stacking ensemble.

Extends run_phase6_shap_by_ancestry_1780.py to the primary cohort's full
18-label Race (Reported) field, now that the per-ancestry SHAP capability
has been verified correct on IWPC-1780 (sanity check there matched the
published overall result exactly). This checks whether the explainability
shift found on IWPC-1780's three coarse groups also holds at IWPC-6256's
finer label resolution.

Raw IWPC-6256 data is not bundled in this repo (per the Data Availability
Statement) and is read from the private local working copy below.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from warfarisk.common import data as data_mod
from warfarisk.common import splits
from warfarisk.phase3_genetics_ablation import build_stacking_ensemble
from warfarisk.phase6_explainability import shap_by_ancestry_subgroup, shap_feature_importance

RAW_IWPC_6256 = Path(
    "/Users/hassan/Documents/Work/SURE-AIC-Research/Warfin/SURE-Program/Models-Development"
    "/data/eda_notebooks/cleaned/iwpc_full_6256_clean.csv"
)
OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "phase6_iwpc6256_shap_by_ancestry.csv"


def main() -> None:
    df = pd.read_csv(RAW_IWPC_6256)
    split = splits.load_split("iwpc_6256")
    train_df, test_df = splits.apply_split(df, split)

    X_train = data_mod.iwpc6256_feature_set(train_df, "combined")
    y_train = train_df[data_mod.IWPC6256_TARGET]
    X_test = data_mod.iwpc6256_feature_set(test_df, "combined")

    print(f"Fitting stacking_ensemble[combined] on {X_train.shape} train rows...", flush=True)
    pipeline = build_stacking_ensemble()
    pipeline.fit(X_train, y_train)

    # Sanity check: unstratified overall importance should match the already-published
    # phase6_iwpc6256_stacking_ensemble_shap.csv (same pipeline, same split, same seed).
    overall = shap_feature_importance(pipeline, X_test, sample_size=200)
    top_feature, top_value = overall.iloc[0]["feature"], overall.iloc[0]["mean_abs_shap"]
    print(f"Sanity check, overall top feature: {top_feature} = {top_value:.3f} (expected: VKORC1 -1639 consensus = 3.257)", flush=True)

    group = test_df["Race (Reported)"]
    print("Ancestry group sizes in test set:", flush=True)
    print(group.value_counts().to_string(), flush=True)

    print("\nComputing SHAP per ancestry subgroup (18 labels)...", flush=True)
    by_group = shap_by_ancestry_subgroup(pipeline, X_test, group, sample_size=200)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    by_group.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print("PHASE6_6256_BY_ANCESTRY_DONE", flush=True)


if __name__ == "__main__":
    main()
