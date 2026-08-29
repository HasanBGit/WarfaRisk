# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
#     "pandas",
# ]
# ///
"""Plot per-ancestry MAPIE calibration coverage on IWPC-6256, against the 90%
nominal target -- the System paper's central finding: aggregate coverage looks
fine (89.8%) but several ancestry subgroups fall well short once unpacked.

Reads directly from results/phase5_iwpc6256_mapie_coverage.csv (the real,
already-computed coverage-by-group table this project produced) -- no
hardcoded numbers.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    """Plot per-ancestry MAPIE coverage vs. the 90% nominal target.

    Saves
    -----
    ./figures/fig_ancestry_calibration_gap.png
    ./figures/fig_ancestry_calibration_gap.pdf
    """
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "results" / "phase5_iwpc6256_mapie_coverage.csv"
    df = pd.read_csv(csv_path).dropna()

    # Keep groups with a meaningful sample size (n >= 10) so single-digit
    # counts don't dominate the chart with unstable percentages; report the
    # overall aggregate separately as a labeled reference line instead of a bar.
    overall = df[df["group"] == "overall"].iloc[0]
    df = df[(df["group"] != "overall") & (df["n"] >= 10)].copy()
    df["coverage_pct"] = df["coverage"] * 100
    df = df.sort_values("coverage_pct", ascending=False)

    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")

    n_bars = df.shape[0]
    fig_w = max(12, 1.15 * n_bars)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5), dpi=150)

    traffic_light = {"green": "#33a02c", "amber": "#fdbf6f", "red": "#e31a1c"}
    colors = [
        traffic_light["green"] if c >= 90 else (traffic_light["amber"] if c >= 70 else traffic_light["red"])
        for c in df["coverage_pct"]
    ]

    x_pos = range(n_bars)
    bars = ax.bar(x_pos, df["coverage_pct"], color=colors, width=0.65, zorder=3)

    ax.axhline(y=90, color="dimgrey", linewidth=1.4, linestyle="--", zorder=2)
    ax.text(n_bars - 0.4, 92, f"90% nominal target (aggregate: {overall['coverage'] * 100:.1f}%)",
            ha="right", va="bottom", fontsize=10, color="dimgrey", style="italic")

    for bar, val, n in zip(bars, df["coverage_pct"], df["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2, f"{val:.0f}%",
                ha="center", va="bottom", fontsize=10, color="dimgrey", weight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, -4, f"n={int(n)}",
                ha="center", va="top", fontsize=8, color="dimgrey")

    ax.set_title("MAPIE Conformal Prediction Coverage by Ancestry Subgroup (IWPC-6256)",
                 loc="left", fontsize=13.5, pad=12)
    ax.set_ylim(-8, max(105, df["coverage_pct"].max() + 10))
    ax.set_ylabel("Empirical 90% Interval Coverage (%)", fontsize=12, labelpad=10, color="dimgrey")
    ax.set_xlabel("")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(df["group"], rotation=35, ha="right")

    sns.despine(left=True, bottom=True)
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")

    fig.text(0.01, -0.04,
              "Green >=90% (nominal) | Amber 70-89% | Red <70%. Groups with n<10 excluded for stability.",
              fontsize=8, color="dimgrey", ha="left")

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "fig_ancestry_calibration_gap.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "fig_ancestry_calibration_gap.pdf", dpi=150, bbox_inches="tight")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
