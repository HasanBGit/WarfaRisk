import datetime

import pandas as pd

from warfarisk.common import data as data_mod
from warfarisk.common import splits
from warfarisk.phase6_explainability import compare_shap_vs_ebm, ebm_feature_importance

split = splits.load_split("iwpc_6256")
df = data_mod.load_iwpc_6256()
train_df, test_df = splits.apply_split(df, split)

X_train = data_mod.iwpc6256_feature_set(train_df, "combined")
y_train = train_df[data_mod.IWPC6256_TARGET]

print(f"Fitting EBM on {X_train.shape} train rows...", flush=True)
ebm_report = ebm_feature_importance(X_train, y_train)
print("\n=== EBM native feature importance (all features) ===", flush=True)
print(ebm_report.to_string(), flush=True)
ebm_report.to_csv("phase6_iwpc6256_ebm_importance.csv", index=False)

shap_report = pd.read_csv("phase6_iwpc6256_stacking_ensemble_shap.csv")
comparison = compare_shap_vs_ebm(shap_report, ebm_report, top_n=10)
print("\n=== SHAP vs EBM top-10 rank comparison ===", flush=True)
print(comparison.to_string(), flush=True)
comparison.to_csv("phase6_iwpc6256_shap_vs_ebm_comparison.csv", index=False)

n_agree = comparison["rank_agreement"].sum()
n_total = len(comparison)
print(f"\n{n_agree}/{n_total} features appear in both top-10 lists", flush=True)

print(f"\nDone at {datetime.datetime.now().isoformat()}", flush=True)
print("EBM_DONE", flush=True)
