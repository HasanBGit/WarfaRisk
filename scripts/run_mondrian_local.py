import datetime

import pandas as pd

from warfarisk.common import data as data_mod
from warfarisk.common import splits
from warfarisk.common.metrics import stratified_conformal_coverage
from warfarisk.phase3_genetics_ablation import build_stacking_ensemble
from warfarisk.phase4_ancestry_holdout import iwpc1780_ancestry_group
from warfarisk.phase5_calibration import mapie_conformal_interval, mondrian_conformal_interval


def run_cohort(name, X_train, y_train, X_test, y_test, group_train, group_test):
    print(f"\n=== Phase 5: MAPIE vs Mondrian conformal calibration, {name} ===", flush=True)

    mapie_est = build_stacking_ensemble()
    mapie_pred, mapie_lower, mapie_upper = mapie_conformal_interval(mapie_est, X_train, y_train, X_test, alpha=0.1)
    mapie_report = stratified_conformal_coverage(y_test, mapie_lower, mapie_upper, group_test)
    mapie_report["method"] = "mapie"

    mondrian_est = build_stacking_ensemble()
    mondrian_pred, mondrian_lower, mondrian_upper, quantiles = mondrian_conformal_interval(
        mondrian_est, X_train, y_train, X_test, group_train, group_test, alpha=0.1
    )
    mondrian_report = stratified_conformal_coverage(y_test, mondrian_lower, mondrian_upper, group_test)
    mondrian_report["method"] = "mondrian"
    mondrian_width = float((mondrian_upper - mondrian_lower).mean())
    mapie_width = float((mapie_upper - mapie_lower).mean())

    print("--- MAPIE (marginal) per-group coverage ---", flush=True)
    print(mapie_report.to_string(), flush=True)
    print("--- Mondrian (group-conditional) per-group coverage ---", flush=True)
    print(mondrian_report.to_string(), flush=True)
    print("--- Mondrian calibration quantiles by group (pooling audit) ---", flush=True)
    print(quantiles.to_string(), flush=True)
    print(f"Mean interval width: MAPIE={mapie_width:.3f} mg/week, Mondrian={mondrian_width:.3f} mg/week", flush=True)

    comparison = pd.concat([mapie_report, mondrian_report], ignore_index=True)
    comparison.to_csv(f"phase5_{name}_mondrian_vs_mapie_coverage.csv", index=False)
    quantiles.to_csv(f"phase5_{name}_mondrian_quantiles.csv", index=False)
    pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "group": group_test.to_numpy(),
            "mapie_pred": mapie_pred,
            "mapie_lower": mapie_lower,
            "mapie_upper": mapie_upper,
            "mondrian_pred": mondrian_pred,
            "mondrian_lower": mondrian_lower,
            "mondrian_upper": mondrian_upper,
        }
    ).to_csv(f"phase5_{name}_mondrian_vs_mapie_predictions.csv", index=False)
    return comparison, quantiles, mapie_width, mondrian_width


# --- iwpc_6256 ---
split = splits.load_split("iwpc_6256")
df = data_mod.load_iwpc_6256()
train_df, test_df = splits.apply_split(df, split)
X_train = data_mod.iwpc6256_feature_set(train_df, "combined")
y_train = train_df[data_mod.IWPC6256_TARGET]
X_test = data_mod.iwpc6256_feature_set(test_df, "combined")
y_test = test_df[data_mod.IWPC6256_TARGET]
group_train_6256 = train_df["Race (Reported)"]
group_test_6256 = test_df["Race (Reported)"]

run_cohort("iwpc6256", X_train, y_train, X_test, y_test, group_train_6256, group_test_6256)

# --- iwpc_1780 ---
split1780 = splits.load_split("iwpc_1780")
df1780 = data_mod.load_iwpc_1780()
train1780, test1780 = splits.apply_split(df1780, split1780)
X_train1780 = data_mod.iwpc1780_feature_set(train1780, "combined")
y_train1780 = train1780[data_mod.IWPC1780_TARGET]
X_test1780 = data_mod.iwpc1780_feature_set(test1780, "combined")
y_test1780 = test1780[data_mod.IWPC1780_TARGET]
group_train_1780 = iwpc1780_ancestry_group(train1780)
group_test_1780 = iwpc1780_ancestry_group(test1780)

run_cohort("iwpc1780", X_train1780, y_train1780, X_test1780, y_test1780, group_train_1780, group_test_1780)

print(f"\nAll done at {datetime.datetime.now().isoformat()}", flush=True)
print("MONDRIAN_DONE", flush=True)
