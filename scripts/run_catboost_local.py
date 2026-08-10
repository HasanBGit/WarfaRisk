import datetime

from warfarisk.common import data as data_mod
from warfarisk.common import splits
from warfarisk.phase3_genetics_ablation import build_catboost, run_catboost_ablation
from warfarisk.phase4_ancestry_holdout import run_phase4_iwpc1780, run_phase4_iwpc6256

print("=== Phase 3: CatBoost ablation, iwpc_6256 ===", flush=True)
split6256 = splits.load_split("iwpc_6256")
df6256 = data_mod.load_iwpc_6256()
train6256, test6256 = splits.apply_split(df6256, split6256)
result3_6256 = run_catboost_ablation(train6256, test6256, data_mod.iwpc6256_feature_set, data_mod.IWPC6256_TARGET, dataset_name="iwpc_6256")
print(result3_6256.to_string(), flush=True)
result3_6256.to_csv("phase3_catboost_iwpc6256_results.csv", index=False)

print("\n=== Phase 3: CatBoost ablation, iwpc_1780 ===", flush=True)
split1780 = splits.load_split("iwpc_1780")
df1780 = data_mod.load_iwpc_1780()
train1780, test1780 = splits.apply_split(df1780, split1780)
result3_1780 = run_catboost_ablation(train1780, test1780, data_mod.iwpc1780_feature_set, data_mod.IWPC1780_TARGET, dataset_name="iwpc_1780")
print(result3_1780.to_string(), flush=True)
result3_1780.to_csv("phase3_catboost_iwpc1780_results.csv", index=False)

print("\n=== Phase 4: ancestry holdout with real CatBoost, iwpc_6256 ===", flush=True)
result4_6256 = run_phase4_iwpc6256(df6256, build_catboost, which="combined")
print(result4_6256.to_string(), flush=True)
result4_6256.to_csv("phase4_iwpc6256_catboost_results.csv", index=False)

print("\n=== Phase 4: ancestry holdout with real CatBoost, iwpc_1780 ===", flush=True)
result4_1780 = run_phase4_iwpc1780(df1780, build_catboost, which="combined")
print(result4_1780.to_string(), flush=True)
result4_1780.to_csv("phase4_iwpc1780_catboost_results.csv", index=False)

print(f"\nAll done at {datetime.datetime.now().isoformat()}", flush=True)
print("CATBOOST_RERUN_DONE", flush=True)
