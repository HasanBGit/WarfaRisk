import datetime

from warfarisk.common import data as data_mod
from warfarisk.phase3_genetics_ablation import build_catboost_autocat
from warfarisk.phase4_ancestry_holdout import run_phase4_iwpc1780, run_phase4_iwpc6256

df6256 = data_mod.load_iwpc_6256()
df1780 = data_mod.load_iwpc_1780()

print("=== Phase 4: ancestry holdout with real CatBoost (autocat wrapper), iwpc_6256 ===", flush=True)
result4_6256 = run_phase4_iwpc6256(df6256, build_catboost_autocat, which="combined")
print(result4_6256.to_string(), flush=True)
result4_6256.to_csv("phase4_iwpc6256_catboost_results.csv", index=False)

print("\n=== Phase 4: ancestry holdout with real CatBoost (autocat wrapper), iwpc_1780 ===", flush=True)
result4_1780 = run_phase4_iwpc1780(df1780, build_catboost_autocat, which="combined")
print(result4_1780.to_string(), flush=True)
result4_1780.to_csv("phase4_iwpc1780_catboost_results.csv", index=False)

print(f"\nAll done at {datetime.datetime.now().isoformat()}", flush=True)
print("CATBOOST_PHASE4_RERUN_DONE", flush=True)
