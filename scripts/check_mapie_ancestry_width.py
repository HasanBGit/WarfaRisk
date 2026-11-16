"""Per-ancestry MAPIE interval width, recovered by positional alignment"""

from __future__ import annotations

import pandas as pd

from warfarisk.common import data as data_mod
from warfarisk.common import splits

TARGET_GROUPS = ["African American", "African-American", "Black or African American", "Black"]


def load_mapie_predictions_with_ancestry() -> pd.DataFrame:
    split = splits.load_split("iwpc_6256")
    df = data_mod.load_iwpc_6256()
    _, test_df = splits.apply_split(df, split)
    group_test = test_df["Race (Reported)"].to_numpy()

    predictions = pd.read_csv("phase5_iwpc6256_mapie_predictions.csv")
    if not (predictions["y_true"].to_numpy() == test_df[data_mod.IWPC6256_TARGET].to_numpy()).all():
        raise ValueError(
            "Row alignment check failed: this file's y_true does not match a fresh load "
            "of the fixed test split. Do not attach ancestry labels positionally until "
            "this is resolved."
        )
    predictions["group"] = group_test
    return predictions


def main() -> None:
    df = load_mapie_predictions_with_ancestry()

    rows = []
    for g in TARGET_GROUPS:
        sub = df[df["group"] == g]
        n = len(sub)
        coverage = ((sub["y_true"] >= sub["lower"]) & (sub["y_true"] <= sub["upper"])).mean()
        width = (sub["upper"] - sub["lower"]).mean()
        rows.append(
            {
                "ancestry_group": g,
                "n": n,
                "mapie_coverage_pct": round(coverage * 100, 1),
                "mapie_width_mg_per_week": round(width, 2),
            }
        )
        print(f"{g:30s} n={n:3d} coverage={coverage * 100:.1f}% width={width:.2f} mg/week")

    pd.DataFrame(rows).to_csv("check_mapie_ancestry_width_results.csv", index=False)
    print("\nWrote check_mapie_ancestry_width_results.csv")


if __name__ == "__main__":
    main()
