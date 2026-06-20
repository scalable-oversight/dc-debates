"""Single-panel tier-uncertainty + delta_v forest plots for the exploratory
Bernoulli sign model (parallels plot_tier_classification.py).

Reads tier-classification-secondary-sign.csv and produces:

  output/tier-uncertainty-stacked-sign.png
    For each transcript, a stacked horizontal bar showing the four
    posterior tier-membership probabilities (P_tier1..P_tier4). Sorted
    by delta_mean descending (hardest at top).

  output/tier-hdi-forest-sign.png
    Horizontal forest line from delta_hdi_lo to delta_hdi_hi per
    transcript, with a dot at delta_mean. Coloured by domain. Sorted by
    delta_mean descending. Vertical dashed line at delta_v = 0.

Note: delta_v here is on the log-odds-of-toward-correct scale (because the
sign model is fit with a logit link). Higher = harder = less likely to push
the judge toward the correct answer.
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

TIER_COLORS = ["#1a9850", "#fdae61", "#f46d43", "#a50026"]
DOMAIN_COLORS = {
    "brainteasers": "#d62728",
    "astrophysics": "#1f77b4",
    "coding": "#2ca02c",
}
N_TIERS = 4


def load_classification():
    df = pd.read_csv(
        OUT_DIR / "tier-classification-secondary-sign.csv", encoding="utf-8"
    )
    return df.sort_values("delta_mean", ascending=False).reset_index(drop=True)


def row_label(row):
    return f"{int(row['transcript']):>2}  {row['question']}"


def plot_stacked(df, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(9, 10))
    n = len(df)
    y = np.arange(n)
    left = np.zeros(n)
    for t in range(N_TIERS):
        probs = df[f"P_tier{t + 1}"].values
        ax.barh(
            y, probs, left=left, color=TIER_COLORS[t],
            edgecolor="white", linewidth=0.5,
            label=f"Tier {t + 1}",
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
        "secondary-sign: sorted by mean delta_v (hardest at top)", fontsize=10
    )
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.suptitle(
        "Per-transcript tier-membership uncertainty (exploratory sign model)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path.name}")


def plot_forest(df, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(9, 10))
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
    ax.set_xlabel("delta_v (log-odds scale; higher = harder)")
    ax.set_title(
        "secondary-sign: posterior mean + 95% HDI per transcript",
        fontsize=10,
    )
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=8)
    legend_handles = [
        Line2D([0], [0], color=c, lw=2.6, label=d)
        for d, c in DOMAIN_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=9, framealpha=0.95)
    fig.suptitle(
        "Per-transcript delta_v posterior (exploratory sign model)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path.name}")


df = load_classification()
plot_stacked(df, OUT_DIR / "tier-uncertainty-stacked-sign.png")
plot_forest(df, OUT_DIR / "tier-hdi-forest-sign.png")
