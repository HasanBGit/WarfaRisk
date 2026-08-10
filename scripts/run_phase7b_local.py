import datetime

import numpy as np

from warfarisk.common import data as data_mod
from warfarisk.phase7_longitudinal_rl import build_eicu_sequences
from warfarisk.phase7b_survival_ttr import compute_time_to_therapeutic_range, evaluate_concordance, fit_cox_or_rsf

print("Building eICU longitudinal sequences...", flush=True)
sequences = build_eicu_sequences()
print(f"Built sequences for {len(sequences)} stays", flush=True)

ttr_df = compute_time_to_therapeutic_range(sequences, time_unit="offset_minutes")
print(f"\nTTR dataframe: {ttr_df.shape}", flush=True)
print(f"Event rate (reached therapeutic range): {ttr_df['event_observed'].mean():.3f}", flush=True)
print(ttr_df.describe(), flush=True)
ttr_df.to_csv("phase7b_eicu_ttr.csv", index=False)

demographics = data_mod.load_eicu_demographics()
merged = ttr_df.merge(demographics, left_on="id", right_on=data_mod.EICU_STAY_ID_COL, how="left")
print(f"\nMerged with demographics: {merged.shape}, non-null age_years: {merged['age_years'].notna().sum()}", flush=True)

covariates = merged[["age_years", "admissionweight", "admissionheight"]]
ttr_aligned = merged[["duration_minutes", "event_observed"]]

rng = np.random.default_rng(20260725)
n = len(merged)
shuffled_idx = rng.permutation(n)
n_test = max(1, int(round(n * 0.2)))
test_idx = shuffled_idx[:n_test]
train_idx = shuffled_idx[n_test:]
print(f"\nHeld-out split: {len(train_idx)} train / {len(test_idx)} test", flush=True)

cov_train, cov_test = covariates.iloc[train_idx], covariates.iloc[test_idx]
ttr_train, ttr_test = ttr_aligned.iloc[train_idx], ttr_aligned.iloc[test_idx]

print("\n=== In-sample (fit==eval, same data) — original numbers, kept for comparison ===", flush=True)
cox_model_full = fit_cox_or_rsf(ttr_aligned, covariates, model="cox")
print(f"Cox PH in-sample concordance: {evaluate_concordance(cox_model_full, ttr_aligned, covariates):.4f}", flush=True)
rsf_model_full = fit_cox_or_rsf(ttr_aligned, covariates, model="rsf")
print(f"RSF in-sample concordance: {evaluate_concordance(rsf_model_full, ttr_aligned, covariates):.4f}", flush=True)

print("\n=== Held-out (fit on train, evaluate on test) — the real result ===", flush=True)
cox_model = fit_cox_or_rsf(ttr_train, cov_train, model="cox")
cox_c_index = evaluate_concordance(cox_model, ttr_test, cov_test)
print(f"Cox PH held-out concordance index: {cox_c_index:.4f}", flush=True)

rsf_model = fit_cox_or_rsf(ttr_train, cov_train, model="rsf")
rsf_c_index = evaluate_concordance(rsf_model, ttr_test, cov_test)
print(f"RSF held-out concordance index: {rsf_c_index:.4f}", flush=True)

print(f"\nDone at {datetime.datetime.now().isoformat()}", flush=True)
print("PHASE7B_DONE", flush=True)
