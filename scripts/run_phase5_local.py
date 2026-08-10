import datetime

from warfarisk.common import data as data_mod
from warfarisk.common import splits
from warfarisk.common.metrics import stratified_conformal_coverage
from warfarisk.phase5_calibration import deep_ensemble_interval

split = splits.load_split("iwpc_6256")
df = data_mod.load_iwpc_6256()
train_df, test_df = splits.apply_split(df, split)

X_train = data_mod.iwpc6256_feature_set(train_df, "combined")
y_train = train_df[data_mod.IWPC6256_TARGET]
X_test = data_mod.iwpc6256_feature_set(test_df, "combined")
y_test = test_df[data_mod.IWPC6256_TARGET]

print(f"Fitting 5-member deep ensemble on {X_train.shape} train rows...", flush=True)
y_pred, lower, upper = deep_ensemble_interval(X_train, y_train, X_test, n_members=5, alpha=0.1)
print("Ensemble fit done.", flush=True)

overall_coverage = ((y_test.to_numpy() >= lower) & (y_test.to_numpy() <= upper)).mean()
print(f"\nOverall 90% interval empirical coverage: {overall_coverage:.4f} (n_test={len(y_test)})", flush=True)

# Also stratify by ancestry group, matching Phase 4's own subgroup discipline
ancestry_test = test_df["Race (Reported)"]
coverage_report = stratified_conformal_coverage(y_test, lower, upper, ancestry_test)
print("\n=== Coverage by ancestry group ===", flush=True)
print(coverage_report.to_string(), flush=True)
coverage_report.to_csv("phase5_iwpc6256_deep_ensemble_coverage.csv", index=False)

import pandas as pd

pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": y_pred, "lower": lower, "upper": upper}).to_csv(
    "phase5_iwpc6256_deep_ensemble_predictions.csv", index=False
)
print(f"\nSaved at {datetime.datetime.now().isoformat()}", flush=True)
print("PHASE5_DONE", flush=True)
