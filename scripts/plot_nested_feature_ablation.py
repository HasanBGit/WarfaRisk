# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
#     "pandas",
# ]
# ///
"""Plot the nested clinical/genetic/combined feature-set ablation across all
9 model architectures on IWPC-6256 -- the System paper's other central
result: genetics adds consistent value on top of clinical-only features
across every architecture, and AutoGluon (combined) is the strongest single
result.

Reads from results/phase3_full_iwpc6256_summary.csv, which was transcribed
directly from the already-computed, evidence-linked table in
docs/RESULTS.md's Phase 3 section (log paths cited there) -- not
re-derived or hardcoded inline here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    """Plot MAE by model architecture, grouped by feature set.

    Saves
    -----
    ./figures/fig_nested_feature_ablation.png
    ./figures/fig_nested_feature_ablation.pdf
    """
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "results" / "phase3_full_iwpc6256_summary.csv"
    df = pd.read_csv(csv_path).dropna()

    # Order models by their combined-feature-set MAE (best to worst) so the
    # strongest architecture reads left-to-right, matching how the paper's
    # own prose describes the ranking (AutoGluon best, XGBoost weakest).
    order = (
        df[df["feature_set"] == "combined"]
        .sort_values("mae")["model"]
        .tolist()
    )
    # linear[clinical] etc. all present for every model except AutoGluon,
    # which was only run on the combined set (the field's accuracy ceiling
    # check, not part of the ablation) -- keep it last, separated visually.
    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)

    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")

    feature_sets = ["clinical", "genetic", "combined"]
    colors = {"clinical": "#fee090", "genetic": "#fc8d59", "combined": "#4575b4"}

    models = order
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.26

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)

    for i, fs in enumerate(feature_sets):
        sub = df[df["feature_set"] == fs].set_index("model").reindex(models)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, sub["mae"], width=width, color=colors[fs],
                       label=fs.capitalize(), zorder=3)
        for bar, val in zip(bars, sub["mae"]):
            if pd.notna(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.12, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7.5, color="dimgrey", rotation=90)

    # Highlight the field's best result (AutoGluon, combined-only). AutoGluon
    # sorts to the leftmost column (lowest combined MAE), so the callout runs
    # horizontally into the open space above the neighbouring bars, with a
    # short arrow down to the AutoGluon combined bar. Anchoring it under the
    # bar would put the text on top of the y-axis.
    best_idx = models.index("autogluon_extreme_quality")
    ax.annotate("Field ceiling: AutoGluon\ncombined only, MAE 8.55, R² 0.475",
                xy=(best_idx + width, 8.7), xytext=(best_idx + 0.35, 13.1),
                fontsize=9.5, color="dimgrey", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="dimgrey", linewidth=0.9,
                                 connectionstyle="arc3,rad=0.0"))

    ax.set_title("Nested Clinical / Genetic / Combined Feature Ablation, 9 Architectures (IWPC-6256)",
                 loc="left", fontsize=13.5, pad=12)
    ax.set_ylabel("MAE (mg/week, lower is better)", fontsize=12, labelpad=10, color="dimgrey")
    ax.set_xlabel("")
    ax.set_xticks(x)
    display_names = [m.replace("_", " ").replace("stacking ensemble", "stacking").title() for m in models]
    # Blank the AutoGluon tick label: at the leftmost position its rotated
    # text collides with the y-axis "0" tick, and it's already named in the
    # "Field ceiling" annotation directly above its bar.
    display_names[best_idx] = ""
    ax.set_xticklabels(display_names, rotation=30, ha="right")
    ax.set_ylim(0, df["mae"].max() + 2.5)

    ax.legend(title="Feature set", loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=True, facecolor="white", framealpha=0.9, edgecolor="lightgrey",
              labelcolor="dimgrey", borderaxespad=0.0)

    sns.despine(left=True, bottom=True)
    ax.xaxis.grid(False)
    ax.yaxis.grid(alpha=0.5, linewidth=0.8)
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")

    fig.text(0.01, -0.05,
              "AutoGluon was run on the combined feature set only (accuracy-ceiling check, not part of the "
              "clinical/genetic ablation).",
              fontsize=8, color="dimgrey", ha="left")

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "fig_nested_feature_ablation.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "fig_nested_feature_ablation.pdf", dpi=150, bbox_inches="tight")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
