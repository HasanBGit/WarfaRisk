import datetime

import pandas as pd

from warfarisk.common import data as data_mod
from warfarisk.common import splits
from warfarisk.common.metrics import stratified_conformal_coverage
from warfarisk.phase3_genetics_ablation import build_stacking_ensemble
from warfarisk.phase4_ancestry_holdout import iwpc1780_ancestry_group, run_phase4_iwpc1780
from warfarisk.phase5_calibration import deep_ensemble_interval
from warfarisk.phase6_explainability import shap_feature_importance

df = data_mod.load_iwpc_1780()
print(f"Loaded iwpc_1780: {df.shape}", flush=True)

# --- Phase 4 ---
print("\n=== Phase 4: ancestry holdout, iwpc_1780 ===", flush=True)
result4 = run_phase4_iwpc1780(df, build_stacking_ensemble, which="combined")
print(result4.to_string(), flush=True)
result4.to_csv("phase4_iwpc1780_stacking_ensemble_results.csv", index=False)

# --- Phase 5 & 6 need a train/test split ---
split = splits.load_split("iwpc_1780")
train_df, test_df = splits.apply_split(df, split)
X_train = data_mod.iwpc1780_feature_set(train_df, "combined")
y_train = train_df[data_mod.IWPC1780_TARGET]
X_test = data_mod.iwpc1780_feature_set(test_df, "combined")
y_test = test_df[data_mod.IWPC1780_TARGET]

# --- Phase 5 ---
print("\n=== Phase 5: deep-ensemble calibration, iwpc_1780 ===", flush=True)
y_pred, lower, upper = deep_ensemble_interval(X_train, y_train, X_test, n_members=5, alpha=0.1)
overall_coverage = ((y_test.to_numpy() >= lower) & (y_test.to_numpy() <= upper)).mean()
print(f"Overall 90% interval empirical coverage: {overall_coverage:.4f} (n_test={len(y_test)})", flush=True)
ancestry_test = iwpc1780_ancestry_group(test_df)
coverage_report = stratified_conformal_coverage(y_test, lower, upper, ancestry_test)
print(coverage_report.to_string(), flush=True)
coverage_report.to_csv("phase5_iwpc1780_deep_ensemble_coverage.csv", index=False)
pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": y_pred, "lower": lower, "upper": upper}).to_csv(
    "phase5_iwpc1780_deep_ensemble_predictions.csv", index=False
)

# --- Phase 6 ---
print("\n=== Phase 6: SHAP, iwpc_1780 ===", flush=True)
pipeline = build_stacking_ensemble()
pipeline.fit(X_train, y_train)
shap_report = shap_feature_importance(pipeline, X_test, sample_size=200)
print(shap_report.to_string(), flush=True)
shap_report.to_csv("phase6_iwpc1780_stacking_ensemble_shap.csv", index=False)

print(f"\nAll done at {datetime.datetime.now().isoformat()}", flush=True)
print("PHASE456_1780_DONE", flush=True)
