"""Sensitivity analysis: pool the four site-specific Black/African-American
labels in IWPC-6256 into one group (reviewer round 2, Comment 3).

Step 1 (reproduction check): rerun the Mondrian comparator with the ORIGINAL
labels and compare against the saved phase5_iwpc6256_mondrian_vs_mapie_predictions.csv.
If it does not reproduce, stop: a pooled run would not be comparable.

Step 2: rerun Mondrian with the four labels merged into one group, same
seed, same calibration split, same fitted model; report pooled coverage,
width and calibration-set size.

Step 3: pooled MAPIE coverage from the headline MAPIE predictions file (no
refit), with an exact binomial CI and a Benjamini-Hochberg test against the
90% target across the pooled label set.

Writes check_pooled_black_sensitivity_results.csv. Touches no existing file.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from check_mapie_ancestry_binomial_ci import (
    MIN_N_FOR_TEST,
    NOMINAL_COVERAGE,
    benjamini_hochberg,
    load_mapie_predictions_with_ancestry,
)
from warfarisk.common import data as data_mod
from warfarisk.common import splits
from warfarisk.phase3_genetics_ablation import build_stacking_ensemble
from warfarisk.phase5_calibration import mondrian_conformal_interval

FOUR = ["Black", "Black or African American", "African-American", "African American"]
POOLED = "Black/African-American (pooled)"


def covered(y, lo, up):
    return (y >= lo) & (y <= up)


split = splits.load_split("iwpc_6256")
df = data_mod.load_iwpc_6256()
train_df, test_df = splits.apply_split(df, split)
X_train = data_mod.iwpc6256_feature_set(train_df, "combined")
y_train = train_df[data_mod.IWPC6256_TARGET]
X_test = data_mod.iwpc6256_feature_set(test_df, "combined")
y_test = test_df[data_mod.IWPC6256_TARGET]
g_train = train_df["Race (Reported)"]
g_test = test_df["Race (Reported)"]


def run_mondrian(gtr, gte):
    est = build_stacking_ensemble()
    _, lo, up, quant = mondrian_conformal_interval(est, X_train, y_train, X_test, gtr, gte, alpha=0.1)
    return lo, up, quant


# --- Step 1: reproduction check -------------------------------------------
print("Step 1: Mondrian with original labels (reproduction check)", flush=True)
lo0, up0, _ = run_mondrian(g_train, g_test)
saved = pd.read_csv("phase5_iwpc6256_mondrian_vs_mapie_predictions.csv")
if not (saved["y_true"].to_numpy() == y_test.to_numpy()).all():
    sys.exit("ABORT: saved Mondrian file is not row-aligned with the test split.")
max_lo = float(np.abs(saved["mondrian_lower"].to_numpy() - lo0).max())
max_up = float(np.abs(saved["mondrian_upper"].to_numpy() - up0).max())
cov_new = covered(y_test.to_numpy(), lo0, up0)
cov_old = covered(saved["y_true"].to_numpy(), saved["mondrian_lower"].to_numpy(), saved["mondrian_upper"].to_numpy())
print(f"  max |lower diff| = {max_lo:.6f}, max |upper diff| = {max_up:.6f} mg/week", flush=True)
print(f"  overall coverage: new {cov_new.mean():.4f} vs saved {cov_old.mean():.4f}; identical per-patient coverage: {(cov_new == cov_old).all()}", flush=True)
if not (cov_new == cov_old).all():
    sys.exit("ABORT: Mondrian does not reproduce its saved result; do not use a pooled rerun.")

# --- Step 2: pooled-label Mondrian ------------------------------------------
print("Step 2: Mondrian with the four labels pooled", flush=True)
pool = lambda s: s.where(~s.isin(FOUR), POOLED)
gtr_p, gte_p = pool(g_train), pool(g_test)
lo1, up1, quant = run_mondrian(gtr_p, gte_p)
m = (gte_p == POOLED).to_numpy()
cov_mond = covered(y_test.to_numpy(), lo1, up1)
print(quant[quant["group"] == POOLED].to_string(), flush=True)

# --- Step 3: pooled MAPIE (headline file) + exact tests ----------------------
mp = load_mapie_predictions_with_ancestry()
cov_mapie = covered(mp["y_true"].to_numpy(), mp["lower"].to_numpy(), mp["upper"].to_numpy())
mm = mp["group"].isin(FOUR).to_numpy()
assert mm.sum() == m.sum(), (mm.sum(), m.sum())

rows = []
for name, cov, lo, up, mask in (
    ("MAPIE (headline file)", cov_mapie, mp["lower"].to_numpy(), mp["upper"].to_numpy(), mm),
    ("Mondrian, labels pooled", cov_mond, lo1, up1, m),
    ("Mondrian, original labels (same patients)", cov_new, lo0, up0, m),
):
    k, n = int(cov[mask].sum()), int(mask.sum())
    ci = binomtest(k, n, NOMINAL_COVERAGE).proportion_ci(method="exact")
    rows.append({
        "method": name, "n": n, "covered": k, "coverage": k / n,
        "ci_low": ci.low, "ci_high": ci.high,
        "mean_width": float((up - lo)[mask].mean()),
        "p_vs_90": binomtest(k, n, NOMINAL_COVERAGE).pvalue,
    })
out = pd.DataFrame(rows)

# BH across MAPIE per-label tests with the four labels replaced by the pooled one
mp["group_p"] = pool(mp["group"])
mp["cov"] = cov_mapie
tested = []
for lab, g in mp.groupby("group_p"):
    if len(g) >= MIN_N_FOR_TEST:
        tested.append((lab, binomtest(int(g["cov"].sum()), len(g), NOMINAL_COVERAGE).pvalue))
adj = benjamini_hochberg([p for _, p in tested])
bh = {lab: a for (lab, _), a in zip(tested, adj)}
out["bh_adj_p_mapie_pooled_labelset"] = [bh.get(POOLED) if r.startswith("MAPIE") else np.nan for r in out["method"]]
out["n_labels_tested"] = len(tested)
print(out.to_string(), flush=True)
out.to_csv("check_pooled_black_sensitivity_results.csv", index=False)
print("POOLED_DONE", flush=True)
