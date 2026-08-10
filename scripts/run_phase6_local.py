import datetime

from warfarisk.common import data as data_mod
from warfarisk.common import splits
from warfarisk.phase3_genetics_ablation import build_stacking_ensemble
from warfarisk.phase6_explainability import shap_feature_importance

split = splits.load_split("iwpc_6256")
df = data_mod.load_iwpc_6256()
train_df, test_df = splits.apply_split(df, split)

X_train = data_mod.iwpc6256_feature_set(train_df, "combined")
y_train = train_df[data_mod.IWPC6256_TARGET]
X_test = data_mod.iwpc6256_feature_set(test_df, "combined")

print(f"Fitting stacking_ensemble[combined] on {X_train.shape} train rows...", flush=True)
pipeline = build_stacking_ensemble()
pipeline.fit(X_train, y_train)
print("Fit done, computing SHAP...", flush=True)

shap_report = shap_feature_importance(pipeline, X_test, sample_size=200)
print("\n=== Top 15 features by mean |SHAP| ===", flush=True)
print(shap_report.head(15).to_string(), flush=True)
shap_report.to_csv("phase6_iwpc6256_stacking_ensemble_shap.csv", index=False)
print(f"\nSaved to phase6_iwpc6256_stacking_ensemble_shap.csv at {datetime.datetime.now().isoformat()}", flush=True)
print("PHASE6_DONE", flush=True)
