# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "pandas",
# ]
# ///
"""Plot NGBoost's natural-gradient training loss curve on both cohorts, from
the actual logged fit (results/phase5_ngboost_training_loss.csv, transcribed
from results/logs/logs_ngboost_local.log -- not synthetic)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    """Plot NGBoost training loss by boosting iteration, both cohorts.

    Saves
    -----
    ./figures/fig_ngboost_training_loss.png
    ./figures/fig_ngboost_training_loss.pdf
    """
    repo_root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(repo_root / "results" / "phase5_ngboost_training_loss.csv").dropna()

    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    colors = {"iwpc_6256": "#4575b4", "iwpc_1780": "#fc8d59"}
    labels = {"iwpc_6256": "IWPC-6256", "iwpc_1780": "IWPC-1780"}

    for cohort in ["iwpc_6256", "iwpc_1780"]:
        sub = df[df["cohort"] == cohort].sort_values("iter")
        ax.plot(sub["iter"], sub["loss"], marker="o", markersize=5, linewidth=2,
                color=colors[cohort], label=labels[cohort])
        last = sub.iloc[-1]
        ax.annotate(f"{last['loss']:.3f}", xy=(last["iter"], last["loss"]),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=colors[cohort], weight="bold")

    ax.set_title("NGBoost Natural-Gradient Training Loss", loc="left", fontsize=13.5, pad=12)
    ax.set_xlabel("Boosting iteration", fontsize=11, labelpad=8, color="dimgrey")
    ax.set_ylabel("Training loss (negative log-likelihood)", fontsize=11, labelpad=8, color="dimgrey")
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.8,
              edgecolor="lightgrey", labelcolor="dimgrey")

    sns.despine(left=True, bottom=True)
    ax.grid(alpha=0.5, linewidth=0.8, axis="y")
    ax.xaxis.grid(False)
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "fig_ngboost_training_loss.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "fig_ngboost_training_loss.pdf", dpi=150, bbox_inches="tight")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
