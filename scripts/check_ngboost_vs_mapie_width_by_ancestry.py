"""Per-ancestry NGBoost vs. MAPIE interval width, recovered by positional
alignment against the fixed test split.

Why this file exists: the manuscript's Discussion states that NGBoost
tracks per-ancestry coverage better than MAPIE for the under-covered labels.
That is a coverage-level observation, not an explanation. This script checks
the actual mechanism: whether NGBoost's predicted interval width (its
log-scale output is a boosted function of patient features, unlike MAPIE's
single global conformal margin) varies by ancestry group in a way that lines
up with which groups MAPIE under- or over-covers.

Inputs this repo does not bundle (per the Data Availability Statement, raw
patient-level data is excluded from the public release): the cleaned
IWPC-6256 CSV and the two phases' saved per-patient prediction files. Both
are read from the private local working copy (Models-Development) via the
paths below; swap them for this repo's own `results/` outputs if those
predictions files are ever added here.

Row alignment is verified against each prediction file's own y_true column
before trusting the positional ancestry join, same safeguard already used
for the equivalent MAPIE-only check.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from warfarisk.common import data as data_mod
from warfarisk.common import splits

_DEV_ROOT = Path("/Users/hassan/Documents/Work/SURE-AIC-Research/Warfin/SURE-Program/Models-Development")
RAW_IWPC_6256 = _DEV_ROOT / "data/eda_notebooks/cleaned/iwpc_full_6256_clean.csv"
MAPIE_PREDICTIONS = _DEV_ROOT / "model_dev/phase5_iwpc6256_mapie_predictions.csv"
NGBOOST_PREDICTIONS = _DEV_ROOT / "model_dev/phase5_iwpc6256_ngboost_predictions.csv"

OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "check_ngboost_vs_mapie_width_by_ancestry_results.csv"

GROUPS = [
    "African American", "African-American", "Black or African American", "Intermediate",
    "Black", "Caucasian", "Chinese", "White", "Japanese", "Korean",
]


def _load_with_ancestry(predictions_path: Path, test_df: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path)
    if not (predictions["y_true"].to_numpy() == test_df[data_mod.IWPC6256_TARGET].to_numpy()).all():
        raise ValueError(
            f"Row alignment check failed for {predictions_path}: its y_true does not match "
            "a fresh load of the fixed test split. Do not attach ancestry labels positionally."
        )
    predictions["group"] = test_df["Race (Reported)"].to_numpy()
    predictions["width"] = predictions["upper"] - predictions["lower"]
    return predictions


def main() -> None:
    split = splits.load_split("iwpc_6256")
    df = pd.read_csv(RAW_IWPC_6256)
    _, test_df = splits.apply_split(df, split)

    mapie = _load_with_ancestry(MAPIE_PREDICTIONS, test_df)
    ngboost = _load_with_ancestry(NGBOOST_PREDICTIONS, test_df)

    rows = []
    print(f"{'group':30s} {'n':>4s} {'MAPIE width':>12s} {'NGBoost width':>14s}")
    for g in GROUPS:
        m = mapie[mapie["group"] == g]
        n_ = ngboost[ngboost["group"] == g]
        assert len(m) == len(n_)
        rows.append({
            "ancestry_group": g,
            "n": len(m),
            "mapie_width_mg_per_week": round(m["width"].mean(), 2),
            "ngboost_width_mg_per_week": round(n_["width"].mean(), 2),
        })
        print(f"{g:30s} {len(m):4d} {m['width'].mean():12.2f} {n_['width'].mean():14.2f}")

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print("\nMAPIE width std across groups:  ", round(out["mapie_width_mg_per_week"].std(), 2))
    print("NGBoost width std across groups:", round(out["ngboost_width_mg_per_week"].std(), 2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
