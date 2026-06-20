"""Visualize per-transcript tier uncertainty and delta_v HDIs.

Reads tier-classification-{primary,secondary}.csv and produces:

  output/tier-uncertainty-stacked.png
    Two-panel figure (primary | secondary). For each outcome, a stacked
    horizontal bar per transcript showing the four posterior tier-membership
    probabilities (P_tier1..P_tier4). Transcripts sorted by delta_mean
    descending (hardest at top).

  output/tier-hdi-forest.png
    Two-panel forest plot. For each outcome, a horizontal line from
    delta_hdi_lo to delta_hdi_hi per transcript, with a dot at delta_mean.
    Coloured by domain. Sorted by delta_mean descending within each panel
    (hardest at top). A vertical dashed line at delta_v = 0 marks the
    boundary between transcripts that on average mislead vs guide judges.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

OUT_DIR = Path(os.environ.get("ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent / "output")))

# Sequential green->red palette: tier 1 (easy) -> tier 4 (hard).
TIER_COLORS = ["#1a9850", "#fdae61", "#f46d43", "#a50026"]
DOMAIN_COLORS = {
    "brainteasers": "#d62728",
    "astrophysics": "#1f77b4",
    "coding": "#2ca02c",
}


def load_classification(outcome):
    df = pd.read_csv(
        OUT_DIR / f"tier-classification-{outcome}.csv", encoding="utf-8"
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
        for t in range(4):
            probs = df[f"P_tier{t + 1}"].values
            ax.barh(
                y, probs, left=left, color=TIER_COLORS[t],
                edgecolor="white", linewidth=0.5,
                label=f"Tier {t + 1}" if ax is axes[0] else None,
            )
            # Annotate per-segment percentage when there's room.
            for i, p in enumerate(probs):
                if p >= 0.08:
                    ax.text(
                        left[i] + p / 2, i, f"{int(round(p * 100))}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if t in (0, 3) else "black",
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
    fig.suptitle("Per-transcript tier-membership uncertainty", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path.name}")


def plot_forest(prim, sec, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharex=False)
    for ax, df, title in zip(axes, [prim, sec], ["primary", "secondary"]):
        n = len(df)
        y = np.arange(n)
        colors = [DOMAIN_COLORS[d] for d in df["domain"]]
        for i in range(n):
            ax.plot(
                [df["delta_hdi_lo"].iloc[i], df["delta_hdi_hi"].iloc[i]],
                [y[i], y[i]],
                color=colors[i], lw=2.2, solid_capstyle="round", alpha=0.85,
            )
        ax.scatter(
            df["delta_mean"], y, c=colors, s=36, zorder=3,
            edgecolor="white", linewidth=0.6,
        )
        ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([row_label(r) for _, r in df.iterrows()], fontsize=8)
        ax.set_xlabel("delta_v  (higher = harder)")
        ax.set_title(
            f"{title}: posterior mean + 95% HDI per transcript", fontsize=10
        )
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=8)
    legend_handles = [
        Line2D([0], [0], color=c, lw=2.6, label=d)
        for d, c in DOMAIN_COLORS.items()
    ]
    axes[0].legend(handles=legend_handles, loc="lower right",
                   fontsize=9, framealpha=0.95)
    fig.suptitle("Per-transcript delta_v posterior", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path.name}")


prim = load_classification("primary")
sec = load_classification("secondary")

plot_stacked_bars(prim, sec, OUT_DIR / "tier-uncertainty-stacked.png")
plot_forest(prim, sec, OUT_DIR / "tier-hdi-forest.png")
