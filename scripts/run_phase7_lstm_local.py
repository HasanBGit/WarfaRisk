import datetime

import numpy as np
import torch

from warfarisk.phase7_longitudinal_rl import build_eicu_sequences, fit_lstm_dose_classifier

sequences = build_eicu_sequences()
print(f"Built sequences for {len(sequences)} stays", flush=True)

# Only keep stays with >=2 INR readings and >=1 dose event (fit_lstm_dose_classifier's
# own requirement) so the held-out split doesn't get diluted by unusable stays.
usable_ids = []
for stay_id, seq in sequences.items():
    n_inr = (seq["kind"] == "inr").sum()
    n_dose = (seq["kind"] == "dose").sum()
    if n_inr >= 2 and n_dose >= 1:
        usable_ids.append(stay_id)
print(f"Usable stays (>=2 INR, >=1 dose): {len(usable_ids)}", flush=True)

rng = np.random.default_rng(20260725)
shuffled = rng.permutation(usable_ids)
n_test = max(1, int(round(len(shuffled) * 0.2)))
test_ids = set(shuffled[:n_test].tolist())
train_ids = set(shuffled[n_test:].tolist())
print(f"Held-out split: {len(train_ids)} train / {len(test_ids)} test stays", flush=True)

train_sequences = {k: v for k, v in sequences.items() if k in train_ids}
test_sequences = {k: v for k, v in sequences.items() if k in test_ids}

dose_bins = np.array([1.0, 2.5, 5.0, 7.5])  # -> 5 classes: <1, 1-2.5, 2.5-5, 5-7.5, >7.5 mg

print(f"\nTraining LSTM dose classifier on {len(train_sequences)} stays...", flush=True)
model = fit_lstm_dose_classifier(train_sequences, dose_bins, hidden_size=32, epochs=50)
print("Training done.", flush=True)


def evaluate(seqs, model, dose_bins):
    model.eval()
    correct = 0
    total = 0
    baseline_correct = 0  # naive baseline: always predict the most common training bin
    with torch.no_grad():
        for _seq_id, seq in seqs.items():
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


train_correct, train_total = evaluate(train_sequences, model, dose_bins)
test_correct, test_total = evaluate(test_sequences, model, dose_bins)

print(f"\nTrain accuracy: {train_correct}/{train_total} = {train_correct/max(train_total,1):.4f}", flush=True)
print(f"Held-out test accuracy: {test_correct}/{test_total} = {test_correct/max(test_total,1):.4f}", flush=True)

print(f"\nDone at {datetime.datetime.now().isoformat()}", flush=True)
print("PHASE7_LSTM_DONE", flush=True)
