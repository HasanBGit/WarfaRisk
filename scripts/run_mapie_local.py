import datetime

import pandas as pd

from warfarisk.common import data as data_mod
from warfarisk.common import splits
from warfarisk.common.metrics import stratified_conformal_coverage
from warfarisk.phase3_genetics_ablation import build_stacking_ensemble
from warfarisk.phase4_ancestry_holdout import iwpc1780_ancestry_group
from warfarisk.phase5_calibration import mapie_conformal_interval

# --- iwpc_6256 ---
print("=== Phase 5: MAPIE conformal calibration, iwpc_6256 ===", flush=True)
split = splits.load_split("iwpc_6256")
df = data_mod.load_iwpc_6256()
train_df, test_df = splits.apply_split(df, split)
X_train = data_mod.iwpc6256_feature_set(train_df, "combined")
y_train = train_df[data_mod.IWPC6256_TARGET]
X_test = data_mod.iwpc6256_feature_set(test_df, "combined")
y_test = test_df[data_mod.IWPC6256_TARGET]

base_estimator = build_stacking_ensemble()
y_pred, lower, upper = mapie_conformal_interval(base_estimator, X_train, y_train, X_test, alpha=0.1)
overall_coverage = ((y_test.to_numpy() >= lower) & (y_test.to_numpy() <= upper)).mean()
print(f"Overall 90% interval empirical coverage: {overall_coverage:.4f} (n_test={len(y_test)})", flush=True)
ancestry_test = test_df["Race (Reported)"]
coverage_report = stratified_conformal_coverage(y_test, lower, upper, ancestry_test)
print(coverage_report.to_string(), flush=True)
coverage_report.to_csv("phase5_iwpc6256_mapie_coverage.csv", index=False)
pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": y_pred, "lower": lower, "upper": upper}).to_csv(
    "phase5_iwpc6256_mapie_predictions.csv", index=False
)

# --- iwpc_1780 ---
print("\n=== Phase 5: MAPIE conformal calibration, iwpc_1780 ===", flush=True)
split1780 = splits.load_split("iwpc_1780")
df1780 = data_mod.load_iwpc_1780()
train1780, test1780 = splits.apply_split(df1780, split1780)
X_train1780 = data_mod.iwpc1780_feature_set(train1780, "combined")
y_train1780 = train1780[data_mod.IWPC1780_TARGET]
X_test1780 = data_mod.iwpc1780_feature_set(test1780, "combined")
y_test1780 = test1780[data_mod.IWPC1780_TARGET]

base_estimator2 = build_stacking_ensemble()
y_pred2, lower2, upper2 = mapie_conformal_interval(base_estimator2, X_train1780, y_train1780, X_test1780, alpha=0.1)
overall_coverage2 = ((y_test1780.to_numpy() >= lower2) & (y_test1780.to_numpy() <= upper2)).mean()
print(f"Overall 90% interval empirical coverage: {overall_coverage2:.4f} (n_test={len(y_test1780)})", flush=True)
ancestry_test1780 = iwpc1780_ancestry_group(test1780)
coverage_report2 = stratified_conformal_coverage(y_test1780, lower2, upper2, ancestry_test1780)
print(coverage_report2.to_string(), flush=True)
coverage_report2.to_csv("phase5_iwpc1780_mapie_coverage.csv", index=False)
pd.DataFrame({"y_true": y_test1780.to_numpy(), "y_pred": y_pred2, "lower": lower2, "upper": upper2}).to_csv(
    "phase5_iwpc1780_mapie_predictions.csv", index=False
)

print(f"\nAll done at {datetime.datetime.now().isoformat()}", flush=True)
print("MAPIE_DONE", flush=True)
