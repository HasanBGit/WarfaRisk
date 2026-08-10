import datetime

import pandas as pd

from warfarisk.common import data as data_mod
from warfarisk.common import splits
from warfarisk.phase3_genetics_ablation import build_stacking_ensemble
from warfarisk.phase6_explainability import compare_shap_vs_ebm, ebm_feature_importance, shap_feature_importance

split = splits.load_split("iwpc_1780")
df = data_mod.load_iwpc_1780()
train_df, test_df = splits.apply_split(df, split)

X_train = data_mod.iwpc1780_feature_set(train_df, "combined")
y_train = train_df[data_mod.IWPC1780_TARGET]
X_test = data_mod.iwpc1780_feature_set(test_df, "combined")

print(f"Fitting EBM on {X_train.shape} train rows...", flush=True)
ebm_report = ebm_feature_importance(X_train, y_train)
print("\n=== EBM native feature importance ===", flush=True)
print(ebm_report.to_string(), flush=True)
ebm_report.to_csv("phase6_iwpc1780_ebm_importance.csv", index=False)

shap_report = pd.read_csv("phase6_iwpc1780_stacking_ensemble_shap.csv")
comparison = compare_shap_vs_ebm(shap_report, ebm_report, top_n=10)
print("\n=== SHAP vs EBM top-10 rank comparison ===", flush=True)
print(comparison.to_string(), flush=True)
comparison.to_csv("phase6_iwpc1780_shap_vs_ebm_comparison.csv", index=False)

n_agree = comparison["rank_agreement"].sum()
n_total = len(comparison)
print(f"\n{n_agree}/{n_total} features appear in both top-10 lists", flush=True)

print(f"\nDone at {datetime.datetime.now().isoformat()}", flush=True)
print("EBM_1780_DONE", flush=True)
