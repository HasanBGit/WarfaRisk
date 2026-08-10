"""Phase 7b — survival analysis for time-in-therapeutic-range (TTR).

Implements the "comparator sweep 2" survival-analysis addition to
TRAINING_PLAN.md Phase 7: frames "time until INR first reaches the
therapeutic range [2.0, 3.0]" as a time-to-event problem with censoring
(patients discharged/sequence-truncated before ever reaching range are real,
right-censored observations here, not missing data to drop) instead of only
predicting a dose or INR point value. This is a genuinely different target
from every other phase in the plan and reports its own metric (concordance
index), not MAE/R2/PW20 — TTR is the clinical outcome the review's own
literature treats as more meaningful than a raw dose error.

scikit-survival (Cox regression, Random Survival Forest) is the recommended
starting point at this project's scale; pycox's deep survival models
(DeepSurv, DeepHit) are real but likely to overfit below the eICU cohort's
n=160 — only worth trying on that slice, not MIMIC's 17-23 patients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warfarisk.common.metrics import concordance_index

THERAPEUTIC_LOWER = 2.0
THERAPEUTIC_UPPER = 3.0


def compute_time_to_therapeutic_range(sequences: dict, time_unit: str = "offset_minutes") -> pd.DataFrame:
    """From the {id: DataFrame} sequences built in phase7_longitudinal_rl (eICU
    uses 'offset_minutes', MIMIC-III/IV use 'time' as a datetime column):
    one row per patient/stay with (duration, event_observed). event_observed
    = True if INR reached [2.0, 3.0] within the observed sequence, False
    (right-censored) if the sequence ends first.
    """
    rows = []
    for seq_id, seq in sequences.items():
        inr_rows = seq[seq["kind"] == "inr"].copy()
        if inr_rows.empty:
            continue
        inr_rows = inr_rows.sort_values(time_unit)
        first_time = inr_rows[time_unit].iloc[0]
        in_range = inr_rows[(inr_rows["value"] >= THERAPEUTIC_LOWER) & (inr_rows["value"] <= THERAPEUTIC_UPPER)]

        if not in_range.empty:
            event_time = in_range[time_unit].iloc[0]
            event_observed = True
        else:
            event_time = inr_rows[time_unit].iloc[-1]  # last observed time, sequence ends before reaching range
            event_observed = False

        if time_unit == "offset_minutes":
            duration = float(event_time - first_time)
        else:
            duration = (pd.to_datetime(event_time) - pd.to_datetime(first_time)).total_seconds() / 60.0

        rows.append({"id": seq_id, "duration_minutes": duration, "event_observed": event_observed})

    return pd.DataFrame(rows)


def fit_cox_or_rsf(
    ttr_df: pd.DataFrame,
    covariates: pd.DataFrame,
    model: str = "cox",
):
    """ttr_df must have 'duration_minutes' and 'event_observed' columns (from
    compute_time_to_therapeutic_range), aligned by index with `covariates`.
    model in {'cox', 'rsf'}. Requires `uv add 'scikit-survival>=0.28'` —
    confirmed 2026-07-26 that scikit-survival==0.24.1 is incompatible with
    scikit-learn>=1.9 (already installed): `sksurv.ensemble` imports the
    private Cython symbol `DTYPE` from `sklearn.tree._tree`, which no longer
    exists in current scikit-learn, raising `ImportError: cannot import name
    'DTYPE'` on any import from `sksurv.ensemble` (including
    RandomSurvivalForest) even though CoxPHSurvivalAnalysis alone would be
    unaffected. scikit-survival 0.28.0 declares explicit
    `scikit-learn<1.10,>=1.9.0` support and fixes this."""
    try:
        from sksurv.util import Surv
    except ImportError as e:
        raise ImportError("scikit-survival is required. Install with: uv add scikit-survival") from e

    y = Surv.from_arrays(event=ttr_df["event_observed"].to_numpy(), time=ttr_df["duration_minutes"].to_numpy())
    X = covariates.select_dtypes(include="number").fillna(covariates.select_dtypes(include="number").median())

    if model == "cox":
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        estimator = CoxPHSurvivalAnalysis()
    elif model == "rsf":
        from sksurv.ensemble import RandomSurvivalForest

        estimator = RandomSurvivalForest(random_state=0, n_estimators=200)
    else:
        raise ValueError(f"model must be 'cox' or 'rsf', got {model!r}")

    estimator.fit(X, y)
    return estimator


def evaluate_concordance(estimator, ttr_df: pd.DataFrame, covariates: pd.DataFrame) -> float:
    """Wraps common.metrics.concordance_index for a fitted Cox/RSF model's
    risk predictions — the metric to report instead of MAE/R2/PW20 for this
    phase, kept in its own clearly-labeled result table per TRAINING_PLAN.md's
    verify step for this comparator."""
    X = covariates.select_dtypes(include="number").fillna(covariates.select_dtypes(include="number").median())
    predicted_risk = estimator.predict(X)
    return concordance_index(
        ttr_df["duration_minutes"].to_numpy(),
        np.asarray(predicted_risk),
        ttr_df["event_observed"].to_numpy(),
    )


def fit_pycox_deepsurv(ttr_df: pd.DataFrame, covariates: pd.DataFrame):
    """Deep survival model (pycox's DeepSurv) — only recommended on the eICU
    slice (n=160), likely to overfit on MIMIC-III/IV's 17-23 patients per
    TRAINING_PLAN.md's own caution. Requires `uv add pycox torch`."""
    try:
        import torch
        import torchtuples as tt
        from pycox.models import CoxPH
    except ImportError as e:
        raise ImportError("pycox and torchtuples are required. Install with: uv add pycox torch torchtuples") from e

    X = covariates.select_dtypes(include="number").fillna(covariates.select_dtypes(include="number").median()).to_numpy(
        dtype="float32"
    )
    duration = ttr_df["duration_minutes"].to_numpy(dtype="float32")
    event = ttr_df["event_observed"].to_numpy(dtype="float32")

    net = tt.practical.MLPVanilla(in_features=X.shape[1], num_nodes=[32, 32], out_features=1, batch_norm=True, dropout=0.1)
    model = CoxPH(net, tt.optim.Adam)
    model.fit(X, (duration, event), batch_size=32, epochs=100, verbose=False)
    return model
