"""Mean and median prediction-interval width for the Mondrian (group-conditional)
conformal predictor on IWPC-6256's standard test set (n = 1,207).

Background: a reviewer asked for prediction-interval width alongside coverage,
since a nominally calibrated interval can still be clinically unhelpful if it
is excessively wide. Per-patient lower/upper bounds for both MAPIE and
Mondrian were already retained from the Phase 5 MAPIE/Mondrian benchmarking
run (phase5_iwpc6256_mondrian_vs_mapie_predictions.csv).

Only the Mondrian width is reported from this file, not MAPIE's: this
particular file's MAPIE run is a fresh, non-canonical run used only to
benchmark Mondrian against a same-run MAPIE baseline (Section 6.2), not the
same run behind this manuscript's published overall-coverage figures.
Reporting a MAPIE-specific interval width computed from this run risks
diverging from whatever number the companion evaluation manuscript's own
MAPIE run would produce, given known run-to-run variation in MAPIE's
CrossConformalRegressor coverage between unseeded runs. Mondrian's calibration is fully
seeded (random_state=20260725) and is new work specific to this manuscript,
with no pre-existing published counterpart, so its width carries no such
risk.
"""

from __future__ import annotations

import pandas as pd


def main() -> None:
    df = pd.read_csv("phase5_iwpc6256_mondrian_vs_mapie_predictions.csv")
    width = df["mondrian_upper"] - df["mondrian_lower"]

    print(f"n = {len(df)}")
    print(f"Mondrian interval width: mean = {width.mean():.2f} mg/week, median = {width.median():.2f} mg/week")
    print(f"Mondrian interval width: min = {width.min():.2f}, max = {width.max():.2f} mg/week")

    by_group = df.assign(width=width).groupby("group", observed=True)["width"].agg(["mean", "count"]).sort_values("count", ascending=False)
    print("\nPer-ancestry-label Mondrian interval width:")
    print(by_group.to_string())

    pd.DataFrame(
        [{"metric": "mondrian_width_mean", "value": round(width.mean(), 2)},
         {"metric": "mondrian_width_median", "value": round(width.median(), 2)}]
    ).to_csv("check_mondrian_interval_width_results.csv", index=False)
    print("\nWrote check_mondrian_interval_width_results.csv")


if __name__ == "__main__":
    main()
