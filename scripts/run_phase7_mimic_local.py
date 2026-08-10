import datetime

import numpy as np
import torch

from warfarisk.common import data as data_mod
from warfarisk.phase7_longitudinal_rl import build_mimic3_sequences, build_mimic4_sequences, fit_lstm_dose_classifier
from warfarisk.phase7b_survival_ttr import compute_time_to_therapeutic_range, evaluate_concordance, fit_cox_or_rsf

DOSE_BINS = np.array([1.0, 2.5, 5.0, 7.5])


def lstm_eval(seqs, model, dose_bins):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _sid, seq in seqs.items():
            inr_values = seq.loc[seq["kind"] == "inr", "value"].to_numpy(dtype=float)
            dose_values = seq.loc[seq["kind"] == "dose", "value"].to_numpy(dtype=float)
            n_steps = min(len(inr_values) - 1, len(dose_values))
            for t in range(1, n_steps):
                history = inr_values[:t]
                if len(history) < 2:
                    continue
                x = torch.tensor(history, dtype=torch.float32).unsqueeze(-1).unsqueeze(0)
                logits = model(x)
                pred = int(torch.argmax(logits, dim=-1).item())
                true_bin = int(np.digitize(dose_values[t], dose_bins))
                correct += int(pred == true_bin)
                total += 1
    return correct, total


def run_cohort(name, build_fn, admissions_loader, time_unit):
    print(f"\n{'='*20} {name} {'='*20}", flush=True)
    sequences = build_fn()
    usable_ids = [sid for sid, seq in sequences.items() if (seq["kind"] == "inr").sum() >= 2 and (seq["kind"] == "dose").sum() >= 1]
    print(f"{name}: {len(sequences)} total sequences, {len(usable_ids)} usable (>=2 INR, >=1 dose)", flush=True)

    if len(usable_ids) < 10:
        print(f"{name}: too few usable sequences ({len(usable_ids)}) for a meaningful held-out split — skipping LSTM/BCQ here, reporting as such.", flush=True)
        return

    rng = np.random.default_rng(20260725)
    # usable_ids may be a list of (subject_id, hadm_id) tuples for MIMIC —
    # np.random.permutation on a list of same-length tuples silently
    # converts them into a 2D array of lists, breaking hashability as dict
    # keys. Shuffle indices instead and index back into the original list.
    shuffled_idx = rng.permutation(len(usable_ids))
    n_test = max(1, int(round(len(shuffled_idx) * 0.2)))
    test_ids = [usable_ids[i] for i in shuffled_idx[:n_test]]
    train_ids = [usable_ids[i] for i in shuffled_idx[n_test:]]
    print(f"{name}: held-out split {len(train_ids)} train / {len(test_ids)} test (small — read accordingly)", flush=True)

    train_sequences = {sid: sequences[sid] for sid in train_ids}
    test_sequences = {sid: sequences[sid] for sid in test_ids}

    # --- LSTM ---
    print(f"\n{name} — LSTM dose classifier:", flush=True)
    try:
        model = fit_lstm_dose_classifier(train_sequences, DOSE_BINS, hidden_size=16, epochs=50)
        train_correct, train_total = lstm_eval(train_sequences, model, DOSE_BINS)
        test_correct, test_total = lstm_eval(test_sequences, model, DOSE_BINS)
        print(f"{name} LSTM train accuracy: {train_correct}/{train_total} = {train_correct/max(train_total,1):.4f}", flush=True)
        print(f"{name} LSTM held-out accuracy: {test_correct}/{test_total} = {test_correct/max(test_total,1):.4f}", flush=True)
    except Exception as e:
        print(f"{name} LSTM failed: {type(e).__name__}: {e}", flush=True)

    # --- Survival / TTR ---
    print(f"\n{name} — survival (TTR):", flush=True)
    ttr_df = compute_time_to_therapeutic_range(sequences, time_unit=time_unit)
    print(f"{name} TTR n={len(ttr_df)}, event rate={ttr_df['event_observed'].mean():.3f}", flush=True)

    admissions = admissions_loader()
    # build_mimic3/4_sequences() keys its dict by (subject_id, hadm_id) tuples
    # — merging on subject_id alone caused a many-to-many join blowup here
    # (confirmed: MIMIC-III's 34-row TTR table exploded to 250 rows, MIMIC-IV's
    # 62 to 339) because a subject can have multiple admissions in the
    # admissions table. Fixed by merging on the exact (subject_id, hadm_id)
    # pair, matching build_mimic3/4_sequences()'s own grouping key, for a
    # correct 1:1 join.
    ttr_df = ttr_df.copy()
    ttr_df["subject_id"] = ttr_df["id"].apply(lambda x: x[0])
    ttr_df["hadm_id"] = ttr_df["id"].apply(lambda x: x[1])
    merged = ttr_df.merge(admissions[["subject_id", "hadm_id", "age_years", "gender"]], on=["subject_id", "hadm_id"], how="left")
    merged = merged.dropna(subset=["age_years"])
    assert len(merged) <= len(ttr_df), f"merge should never grow the row count (had {len(ttr_df)}, got {len(merged)})"
    print(f"{name} merged with demographics: {len(merged)} rows with non-null age", flush=True)

    if len(merged) < 15:
        print(f"{name}: too few rows with demographics ({len(merged)}) for a meaningful Cox/RSF fit — skipping.", flush=True)
        return

    covariates = merged[["age_years"]].copy()
    covariates["gender_male"] = (merged["gender"] == "M").astype(float)
    ttr_aligned = merged[["duration_minutes", "event_observed"]]

    try:
        cox_model = fit_cox_or_rsf(ttr_aligned, covariates, model="cox")
        cox_c = evaluate_concordance(cox_model, ttr_aligned, covariates)
        print(f"{name} Cox PH in-sample concordance (n={len(merged)}, too small to split further): {cox_c:.4f}", flush=True)
    except Exception as e:
        print(f"{name} Cox failed: {type(e).__name__}: {e}", flush=True)


run_cohort("MIMIC-III", build_mimic3_sequences, data_mod.load_mimic3_admissions, "time")
run_cohort("MIMIC-IV", build_mimic4_sequences, data_mod.load_mimic4_admissions, "time")

print(f"\nDone at {datetime.datetime.now().isoformat()}", flush=True)
print("PHASE7_MIMIC_DONE", flush=True)
