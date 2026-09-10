# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
#     "pandas",
# ]
# ///
"""Plot overall empirical coverage of the 90%-nominal interval, all three
calibration methods (deep ensemble, NGBoost, MAPIE), both cohorts.

Reads the "overall" row directly from the six real, already-computed
results/phase5_*_coverage.csv files -- no hardcoded coverage numbers. The
95% bootstrap confidence intervals are not derivable from those files (they
come from a separate 10,000-resample bootstrap over the retained per-patient
test predictions, fixed seed 20260725, not shipped in this repo); they are
carried here as documented constants matching the System paper's reported
values (Sec. Results -- Calibration) and docs/RESULTS.md.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

# 95% bootstrap CIs (10,000 resamples, seed 20260725) -- see module docstring.
BOOTSTRAP_CI = {
    ("iwpc6256", "deep_ensemble"): (22.5, 27.4),
    ("iwpc6256", "ngboost"): (87.2, 90.7),
    ("iwpc6256", "mapie"): (88.1, 91.5),
    ("iwpc1780", "deep_ensemble"): (17.4, 26.1),
    ("iwpc1780", "ngboost"): (87.9, 93.8),
    ("iwpc1780", "mapie"): (90.4, 95.8),
}


def main():
    """Plot overall calibration coverage vs. the 90% nominal target, both cohorts.

    Saves
    -----
    ./figures/fig_calibration_coverage_ci.pdf
    ./figures/fig_calibration_coverage_ci.png
    """
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")

    cohorts = [
        {"key": "iwpc6256", "csv_tag": "iwpc6256", "title": None},
        {"key": "iwpc1780", "csv_tag": "iwpc1780", "title": None},
    ]
    methods = [
        {"tag": "deep_ensemble", "name": "Deep ensemble"},
        {"tag": "ngboost", "name": "NGBoost"},
        {"tag": "mapie", "name": "MAPIE"},
    ]

    # --- Data: real "overall" coverage + n from results/, CI from constants ---
    for cohort in cohorts:
        rows = []
        n = None
        for method in methods:
            df = pd.read_csv(f"results/phase5_{cohort['csv_tag']}_{method['tag']}_coverage.csv")
            overall = df.loc[df["group"] == "overall"].iloc[0]
            n = int(overall["n"])
            obs = overall["coverage"] * 100
            lo, hi = BOOTSTRAP_CI[(cohort["key"], method["tag"])]
            rows.append({"name": method["name"], "obs": obs, "lo": lo, "hi": hi,
                        "miss": method["tag"] == "deep_ensemble"})
        cohort["rows"] = rows
        cohort["title"] = f"IWPC-{cohort['key'][4:]} (n = {n:,})"

    # --- Palette ---
    pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)
    ON_TARGET = pal[4]       # NGBoost / MAPIE
    ON_TARGET_LINE = pal[3]
    MISS = "#bd0c0c"         # deep ensemble (ACCENT_RED)
    MISS_LINE = "#d9695f"

    NOMINAL = 90.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), dpi=150, sharey=True)

    n_rows = len(methods)
    y_positions = list(range(n_rows - 1, -1, -1))  # top row = highest y

    for ax, cohort in zip(axes, cohorts):
        for y, row in zip(y_positions, cohort["rows"]):
            color = MISS if row["miss"] else ON_TARGET
            line_color = MISS_LINE if row["miss"] else ON_TARGET_LINE

            # 95% CI line
            ax.hlines(y=y, xmin=row["lo"], xmax=row["hi"],
                      color=line_color, lw=4, alpha=0.75, zorder=2)
            # end caps
            for x_end in (row["lo"], row["hi"]):
                ax.plot([x_end, x_end], [y - 0.11, y + 0.11],
                        color=line_color, lw=2.2, alpha=0.9, zorder=3)
            # observed-coverage marker
            ax.scatter(row["obs"], y, s=280, color=color,
                       edgecolors="white", linewidth=1.8, zorder=5)

            # value label above
            ax.text(row["obs"], y + 0.30, f'{row["obs"]:.1f}%',
                    ha="center", va="bottom", fontsize=12.5,
                    weight="semibold", color="dimgrey")
            # CI bracket below
            ax.text((row["lo"] + row["hi"]) / 2, y - 0.30,
                    f'[{row["lo"]:.1f}, {row["hi"]:.1f}]',
                    ha="center", va="top", fontsize=9.5, color="dimgrey")

        # nominal-target reference line
        ax.axvline(NOMINAL, color="grey", linestyle="--", linewidth=1.1,
                   alpha=0.55, zorder=0)
        ax.text(NOMINAL, n_rows - 0.42, "90% target", ha="center",
                va="bottom", fontsize=9, color="dimgrey", style="italic")

        ax.set_xlim(5, 100)
        ax.set_ylim(-0.65, n_rows - 0.1)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([r["name"] for r in cohort["rows"]], fontsize=12.5)
        ax.set_title(cohort["title"], loc="left", fontsize=13.5, pad=10, color="dimgrey")
        ax.set_xlabel("Coverage (%)", fontsize=11.5, labelpad=8, color="dimgrey")
        ax.grid(False)
        ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")

    sns.despine(left=True, bottom=True)
    for ax in axes:
        ax.patch.set_edgecolor("lightgrey")
        ax.patch.set_linewidth(0.8)

    fig.suptitle("MAPIE, NGBoost hit the 90% target; the deep ensemble doesn't",
                fontsize=15.5, x=0.015, ha="left", color="dimgrey", y=1.03)

    # --- Legend (shared, encodes both shape and color meaning) ---
    legend_handles = [
        Line2D([0], [0], color=ON_TARGET_LINE, lw=4, marker="o", markersize=11,
              markerfacecolor=ON_TARGET, markeredgecolor="white", markeredgewidth=1.5,
              label="On target"),
        Line2D([0], [0], color=MISS_LINE, lw=4, marker="o", markersize=11,
              markerfacecolor=MISS, markeredgecolor="white", markeredgewidth=1.5,
              label="Off target"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, frameon=True, facecolor="white", framealpha=0.9,
              edgecolor="lightgrey", labelcolor="dimgrey", fontsize=10.5)

    fig.subplots_adjust(wspace=0.12, top=0.82, bottom=0.1)

    # --- Save ---
    Path("./scripts/figures").mkdir(exist_ok=True, parents=True)
    fig.savefig("./scripts/figures/fig_calibration_coverage_ci.pdf", dpi=150, bbox_inches="tight")
    fig.savefig("./scripts/figures/fig_calibration_coverage_ci.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
