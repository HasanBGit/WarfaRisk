"""
Phase 7 — longitudinal / dynamic-dosing track (clinical-only, no genotype), plus
comparator-sweep LSTM
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warfarisk.common import data as data_mod


def build_eicu_sequences() -> dict[int, pd.DataFrame]:
    """
    One DataFrame per patientunitstayid: columns [offset_minutes, kind, value] where
    kind in {'inr', 'dose'}, offset_minutes relative to unit admission (eICU's
    native offset convention — real, irregular spacing, not resampled)
    """
    inr = data_mod.load_eicu_inr_labs()
    inr = inr[inr["labname"] == data_mod.EICU_INR_LABNAME_TARGET]
    inr_events = inr[[data_mod.EICU_STAY_ID_COL, "labresultoffset", "labresult"]].rename(
        columns={"labresultoffset": "offset_minutes", "labresult": "value"}
    )
    inr_events["kind"] = "inr"

    meds = data_mod.load_eicu_warfarin_medication()
    dose_events = meds[[data_mod.EICU_STAY_ID_COL, "drugstartoffset", "dosage"]].rename(
        columns={"drugstartoffset": "offset_minutes", "dosage": "value"}
    )
    dose_events["value"] = (
        dose_events["value"].astype(str).str.extract(r"^\s*([\d.]+)")[0].astype(float)
    )
    dose_events["kind"] = "dose"

    combined = pd.concat([inr_events, dose_events], ignore_index=True).sort_values(
        [data_mod.EICU_STAY_ID_COL, "offset_minutes"]
    )
    return {
        stay_id: group.drop(columns=data_mod.EICU_STAY_ID_COL)
        for stay_id, group in combined.groupby(data_mod.EICU_STAY_ID_COL)
    }


def build_mimic3_sequences() -> dict[int, pd.DataFrame]:
    """One DataFrame per (subject_id, hadm_id): columns [charttime, kind, value]"""
    inr = data_mod.load_mimic3_inr_trajectory()
    inr_events = inr[
        [data_mod.MIMIC3_PATIENT_ID_COL, data_mod.MIMIC3_STAY_ID_COL, "charttime", "valuenum"]
    ].rename(columns={"charttime": "time", "valuenum": "value"})
    inr_events["time"] = pd.to_datetime(inr_events["time"])
    inr_events["kind"] = "inr"

    rx = data_mod.load_mimic3_warfarin_prescriptions()
    dose_events = rx[
        [data_mod.MIMIC3_PATIENT_ID_COL, data_mod.MIMIC3_STAY_ID_COL, "startdate", "dose_val_rx"]
    ].rename(columns={"startdate": "time", "dose_val_rx": "value"})
    dose_events["time"] = pd.to_datetime(dose_events["time"])
    dose_events["kind"] = "dose"

    combined = pd.concat([inr_events, dose_events], ignore_index=True).sort_values(
        [data_mod.MIMIC3_PATIENT_ID_COL, data_mod.MIMIC3_STAY_ID_COL, "time"]
    )
    key_cols = [data_mod.MIMIC3_PATIENT_ID_COL, data_mod.MIMIC3_STAY_ID_COL]
    return {key: group.drop(columns=key_cols) for key, group in combined.groupby(key_cols)}


def build_mimic4_sequences() -> dict[int, pd.DataFrame]:
    """Same shape as build_mimic3_sequences(), for MIMIC-IV's own charttime/starttime columns"""
    inr = data_mod.load_mimic4_inr_labs()
    inr_events = inr[
        [data_mod.MIMIC4_PATIENT_ID_COL, data_mod.MIMIC4_STAY_ID_COL, "charttime", "valuenum"]
    ].rename(columns={"charttime": "time", "valuenum": "value"})
    inr_events["time"] = pd.to_datetime(inr_events["time"])
    inr_events["kind"] = "inr"

    rx = data_mod.load_mimic4_warfarin_prescriptions()
    dose_events = rx[
        [data_mod.MIMIC4_PATIENT_ID_COL, data_mod.MIMIC4_STAY_ID_COL, "starttime", "dose_val_rx"]
    ].rename(columns={"starttime": "time", "dose_val_rx": "value"})
    dose_events["time"] = pd.to_datetime(dose_events["time"])
    dose_events["kind"] = "dose"

    combined = pd.concat([inr_events, dose_events], ignore_index=True).sort_values(
        [data_mod.MIMIC4_PATIENT_ID_COL, data_mod.MIMIC4_STAY_ID_COL, "time"]
    )
    key_cols = [data_mod.MIMIC4_PATIENT_ID_COL, data_mod.MIMIC4_STAY_ID_COL]
    return {key: group.drop(columns=key_cols) for key, group in combined.groupby(key_cols)}


def temporal_split(
    admissions_df: pd.DataFrame, time_col: str, test_fraction: float = 0.2
) -> tuple[pd.Index, pd.Index]:
    """
    Earlier admissions train, later admissions test — an honest holdout for sequence
    data, per the resampling-risk paper's warning against random shuffles on
    temporally-structured clinical data
    """
    ordered = admissions_df.sort_values(time_col)
    n_test = max(1, int(round(len(ordered) * test_fraction)))
    train_idx = ordered.index[:-n_test]
    test_idx = ordered.index[-n_test:]
    return train_idx, test_idx


def build_bcq_dataset(sequences: dict, dose_bins: np.ndarray):
    """
    Turns the {id: DataFrame[time/offset, kind, value]} sequences above into a
    d3rlpy MDPDataset: state = recent INR history, action = discretized dose (via
    dose_bins), reward = closeness of resulting INR to the therapeutic range [2.0,
    3.0]
    """
    try:
        import d3rlpy
    except ImportError as e:
        raise ImportError("d3rlpy is required. Install with: uv add d3rlpy torch") from e

    observations, actions, rewards, terminals = [], [], [], []
    for _seq_id, seq in sequences.items():
        inr_values = seq.loc[seq["kind"] == "inr", "value"].to_numpy(dtype=float)
        dose_values = seq.loc[seq["kind"] == "dose", "value"].to_numpy(dtype=float)
        n_steps = min(len(inr_values) - 1, len(dose_values))
        if n_steps < 1:
            continue
        for t in range(n_steps):
            state = np.array([inr_values[t]], dtype=np.float32)
            action = int(np.digitize(dose_values[t], dose_bins))
            next_inr = inr_values[t + 1]
            reward = -abs(next_inr - 2.5)
            observations.append(state)
            actions.append(action)
            rewards.append(reward)
            terminals.append(1.0 if t == n_steps - 1 else 0.0)

    return d3rlpy.dataset.MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.float32),
    )


def fit_bcq(dataset, n_steps: int = 10000):
    try:
        import d3rlpy
    except ImportError as e:
        raise ImportError("d3rlpy is required. Install with: uv add d3rlpy torch") from e

    bcq = d3rlpy.algos.DiscreteBCQConfig().create()
    bcq.fit(dataset, n_steps=n_steps)
    return bcq


def fit_lstm_dose_classifier(
    sequences: dict, dose_bins: np.ndarray, hidden_size: int = 32, epochs: int = 50
):
    """
    A minimal LSTM over each stay's INR history, classifying the next discretized
    dose bin — the direct sequence-model comparator to build_bcq_dataset()/fit_bcq()
    above, testing whether Kuang et al.'s temporal-data benefit replicates at this
    project's much smaller cohort sizes (17-160 vs
    """
    try:
        import torch
        from torch import nn
    except ImportError as e:
        raise ImportError("PyTorch is required. Install with: uv add torch") from e

    class DoseLSTM(nn.Module):
        def __init__(self, hidden_size: int, n_classes: int):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, n_classes)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.head(h_n[-1])

    n_classes = len(dose_bins) + 1
    model = DoseLSTM(hidden_size=hidden_size, n_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    sequences_x, sequences_y = [], []
    for _seq_id, seq in sequences.items():
        inr_values = seq.loc[seq["kind"] == "inr", "value"].to_numpy(dtype=float)
        dose_values = seq.loc[seq["kind"] == "dose", "value"].to_numpy(dtype=float)
        n_steps = min(len(inr_values) - 1, len(dose_values))
        for t in range(1, n_steps):
            history = inr_values[:t]
            if len(history) < 2:
                continue
            sequences_x.append(torch.tensor(history, dtype=torch.float32).unsqueeze(-1))
            sequences_y.append(int(np.digitize(dose_values[t], dose_bins)))

    if not sequences_x:
        raise ValueError(
            "No sequences with >= 2 INR readings found — cannot train an LSTM on this cohort."
        )

    model.train()
    for _epoch in range(epochs):
        for x, y in zip(sequences_x, sequences_y):
            optimizer.zero_grad()
            logits = model(x.unsqueeze(0))
            loss = loss_fn(logits, torch.tensor([y]))
            loss.backward()
            optimizer.step()

    return model
