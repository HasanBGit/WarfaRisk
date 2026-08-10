import datetime

import numpy as np

from warfarisk.phase7_longitudinal_rl import build_bcq_dataset, build_eicu_sequences, fit_bcq

sequences = build_eicu_sequences()
usable_ids = [sid for sid, seq in sequences.items() if (seq["kind"] == "inr").sum() >= 2 and (seq["kind"] == "dose").sum() >= 1]
print(f"Usable stays: {len(usable_ids)}", flush=True)

rng = np.random.default_rng(20260725)
shuffled = rng.permutation(usable_ids)
n_test = max(1, int(round(len(shuffled) * 0.2)))
test_ids = shuffled[:n_test].tolist()
train_ids = shuffled[n_test:].tolist()
print(f"Held-out split: {len(train_ids)} train / {len(test_ids)} test stays (same split as the LSTM comparator)", flush=True)

train_sequences = {sid: sequences[sid] for sid in train_ids}
test_sequences = {sid: sequences[sid] for sid in test_ids}

dose_bins = np.array([1.0, 2.5, 5.0, 7.5])  # same bins as the LSTM comparator, for direct comparability

print("\nBuilding train MDPDataset...", flush=True)
train_dataset = build_bcq_dataset(train_sequences, dose_bins)
print(f"Train transitions: {train_dataset.transition_count}", flush=True)

print("\nFitting DiscreteBCQ (500 gradient steps — small dataset, no need for the 10000-step default)...", flush=True)
model = fit_bcq(train_dataset, n_steps=500)
print("Fit done.", flush=True)


def action_agreement(model, sequences, dose_bins):
    """Simple, transparent sanity metric: does the policy's chosen action
    match what was actually prescribed on held-out transitions? This is NOT
    a value-based off-policy evaluation (e.g. Fitted-Q Evaluation) — it's a
    much weaker, honestly-labeled check that the pipeline produces a
    non-degenerate policy, not a claim about whether the policy would
    improve real outcomes. Real off-policy evaluation is a separate,
    substantial piece of work out of scope for this pass.
    """
    agree = 0
    total = 0
    action_counts = {}
    for _sid, seq in sequences.items():
        inr_values = seq.loc[seq["kind"] == "inr", "value"].to_numpy(dtype=float)
        dose_values = seq.loc[seq["kind"] == "dose", "value"].to_numpy(dtype=float)
        n_steps = min(len(inr_values) - 1, len(dose_values))
        for t in range(n_steps):
            state = np.array([[inr_values[t]]], dtype=np.float32)
            predicted_action = int(model.predict(state)[0])
            true_action = int(np.digitize(dose_values[t], dose_bins))
            agree += int(predicted_action == true_action)
            total += 1
            action_counts[predicted_action] = action_counts.get(predicted_action, 0) + 1
    return agree, total, action_counts


train_agree, train_total, train_action_counts = action_agreement(model, train_sequences, dose_bins)
test_agree, test_total, test_action_counts = action_agreement(model, test_sequences, dose_bins)

print(f"\nTrain action-agreement: {train_agree}/{train_total} = {train_agree/max(train_total,1):.4f}", flush=True)
print(f"Held-out action-agreement: {test_agree}/{test_total} = {test_agree/max(test_total,1):.4f}", flush=True)
print(f"Train predicted-action distribution: {train_action_counts}", flush=True)
print(f"Held-out predicted-action distribution: {test_action_counts}", flush=True)

n_distinct_actions = len(set(list(train_action_counts.keys()) + list(test_action_counts.keys())))
print(f"\nDistinct actions the policy actually chooses: {n_distinct_actions} (out of 5 possible bins)", flush=True)
if n_distinct_actions == 1:
    print("WARNING: policy collapsed to a single constant action — degenerate, not a real policy.", flush=True)

print(f"\nDone at {datetime.datetime.now().isoformat()}", flush=True)
print("PHASE7_BCQ_DONE", flush=True)
