"""
Mean and median prediction-interval width for the Mondrian (group-conditional) conformal
predictor on IWPC-6256's standard test set (n = 1,207)
"""

from __future__ import annotations

import pandas as pd


def main() -> None:
    df = pd.read_csv("phase5_iwpc6256_mondrian_vs_mapie_predictions.csv")
    width = df["mondrian_upper"] - df["mondrian_lower"]

    print(f"n = {len(df)}")
    print(
        f"Mondrian interval width: mean = {width.mean():.2f} mg/week, median = {width.median():.2f} mg/week"
    )
    print(f"Mondrian interval width: min = {width.min():.2f}, max = {width.max():.2f} mg/week")

    by_group = (
        df.assign(width=width)
        .groupby("group", observed=True)["width"]
        .agg(["mean", "count"])
        .sort_values("count", ascending=False)
    )
    print("\nPer-ancestry-label Mondrian interval width:")
    print(by_group.to_string())

    pd.DataFrame(
        [
            {"metric": "mondrian_width_mean", "value": round(width.mean(), 2)},
            {"metric": "mondrian_width_median", "value": round(width.median(), 2)},
        ]
    ).to_csv("check_mondrian_interval_width_results.csv", index=False)
    print("\nWrote check_mondrian_interval_width_results.csv")


if __name__ == "__main__":
    main()
