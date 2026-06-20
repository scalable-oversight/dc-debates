"""Visualize per-transcript tier-membership uncertainty for the exploratory
3-tier classification (parallels plot_tier_classification.py).

Reads tier3-classification-{primary,secondary}.csv and produces:

  output/tier3-uncertainty-stacked.png
    Two-panel figure (primary | secondary). For each outcome, a stacked
    horizontal bar per transcript showing the three posterior tier-membership
    probabilities (P_tier1..P_tier3). Transcripts sorted by delta_mean
    descending (hardest at top).
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(os.environ.get("ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent / "output")))

# 3-color RdYlGn slice: tier 1 (easy) -> tier 3 (hard). Drops the second
# light-orange step from the 4-tier palette to keep visual progression clean.
TIER_COLORS = ["#1a9850", "#fdae61", "#a50026"]
N_TIERS = 3


def load_classification(outcome):
    df = pd.read_csv(
        OUT_DIR / f"tier3-classification-{outcome}.csv", encoding="utf-8"
    )
    return df.sort_values("delta_mean", ascending=False).reset_index(drop=True)


def row_label(row):
    return f"{int(row['transcript']):>2}  {row['question']}"


def plot_stacked_bars(prim, sec, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=False)
    for ax, df, title in zip(axes, [prim, sec], ["primary", "secondary"]):
        n = len(df)
        y = np.arange(n)
        left = np.zeros(n)
        for t in range(N_TIERS):
            probs = df[f"P_tier{t + 1}"].values
            ax.barh(
                y, probs, left=left, color=TIER_COLORS[t],
                edgecolor="white", linewidth=0.5,
                label=f"Tier {t + 1}" if ax is axes[0] else None,
            )
            for i, p in enumerate(probs):
                if p >= 0.08:
                    ax.text(
                        left[i] + p / 2, i, f"{int(round(p * 100))}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if t in (0, N_TIERS - 1) else "black",
                    )
            left += probs
        ax.set_yticks(y)
        ax.set_yticklabels([row_label(r) for _, r in df.iterrows()], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Posterior tier-membership probability")
        ax.set_title(
            f"{title}: sorted by mean delta_v (hardest at top)", fontsize=10
        )
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=8)
    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.suptitle(
        "Per-transcript tier-membership uncertainty (exploratory 3-tier)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path.name}")


prim = load_classification("primary")
sec = load_classification("secondary")

plot_stacked_bars(prim, sec, OUT_DIR / "tier3-uncertainty-stacked.png")
