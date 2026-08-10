import datetime

from warfarisk.common import data as data_mod
from warfarisk.phase3_genetics_ablation import build_stacking_ensemble
from warfarisk.phase4_ancestry_holdout import run_phase4_iwpc6256

df = data_mod.load_iwpc_6256()
print(f"Loaded iwpc_6256: {df.shape}", flush=True)
print(f"Ancestry groups (Race (Reported)) value counts:\n{df['Race (Reported)'].value_counts()}", flush=True)

result = run_phase4_iwpc6256(df, build_stacking_ensemble, which="combined")
print("\n=== Phase 4 results (stacking_ensemble[combined], iwpc_6256) ===", flush=True)
print(result.to_string(), flush=True)
result.to_csv("phase4_iwpc6256_stacking_ensemble_results.csv", index=False)
print(f"\nSaved to phase4_iwpc6256_stacking_ensemble_results.csv at {datetime.datetime.now().isoformat()}", flush=True)
print("PHASE4_DONE", flush=True)
