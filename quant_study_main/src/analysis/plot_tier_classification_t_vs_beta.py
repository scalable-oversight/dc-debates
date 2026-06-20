"""Per-transcript tier-membership uncertainty: Student-t vs Beta (side by side).

Analogue of plot_tier_classification.py and plot_tier_classification_3.py
but with the secondary-t fit on the left and the secondary-beta fit on the
right (instead of primary vs secondary). Useful for comparing the two
alternative-likelihood fits' tier-classification stories directly.

Outputs:
  output/tier-uncertainty-stacked-t-vs-beta.png    (4-tier stacked bars)
  output/tier3-uncertainty-stacked-t-vs-beta.png   (3-tier stacked bars)
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(os.environ.get(
    "ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent / "output")
))

# Same palettes used by the existing tier plots, kept consistent so the
# new figure can be skimmed alongside tier-uncertainty-stacked.png and
# tier3-uncertainty-stacked.png without recolouring.
TIER_COLORS_4 = ["#1a9850", "#fdae61", "#f46d43", "#a50026"]
TIER_COLORS_3 = ["#1a9850", "#fdae61", "#a50026"]


def load_classification(in_csv):
    df = pd.read_csv(in_csv, encoding="utf-8")
    return df.sort_values("delta_mean", ascending=False).reset_index(drop=True)


def row_label(row):
    return f"{int(row['transcript']):>2}  {row['question']}"


def plot_stacked_bars(left_df, right_df, n_tiers, palette,
                      titles, suptitle, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=False)
    for ax, df, title in zip(axes, [left_df, right_df], titles):
        n = len(df)
        y = np.arange(n)
        left = np.zeros(n)
        for t in range(n_tiers):
            probs = df[f"P_tier{t + 1}"].values
            ax.barh(
                y, probs, left=left, color=palette[t],
                edgecolor="white", linewidth=0.5,
                label=f"Tier {t + 1}" if ax is axes[0] else None,
            )
            for i, p in enumerate(probs):
                if p >= 0.08:
                    ax.text(
                        left[i] + p / 2, i, f"{int(round(p * 100))}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if t in (0, n_tiers - 1) else "black",
                    )
            left += probs
        ax.set_yticks(y)
        ax.set_yticklabels([row_label(r) for _, r in df.iterrows()], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Posterior tier-membership probability")
        ax.set_title(
            f"{title}: sorted by mean delta_v (hardest at top)", fontsize=10,
        )
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=8)
    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path.name}")


# 4-tier comparison.
t_4 = load_classification(OUT_DIR / "tier-classification-secondary-t.csv")
beta_4 = load_classification(OUT_DIR / "tier-classification-secondary-beta.csv")
plot_stacked_bars(
    t_4, beta_4, n_tiers=4, palette=TIER_COLORS_4,
    titles=("secondary-t (Student-t)", "secondary-beta (Beta)"),
    suptitle=("Per-transcript tier-membership uncertainty "
              "(Student-t vs Beta alternative fits)"),
    out_path=OUT_DIR / "tier-uncertainty-stacked-t-vs-beta.png",
)

# 3-tier comparison.
t_3 = load_classification(OUT_DIR / "tier3-classification-secondary-t.csv")
beta_3 = load_classification(OUT_DIR / "tier3-classification-secondary-beta.csv")
plot_stacked_bars(
    t_3, beta_3, n_tiers=3, palette=TIER_COLORS_3,
    titles=("secondary-t (Student-t)", "secondary-beta (Beta)"),
    suptitle=("Per-transcript tier-membership uncertainty "
              "(Student-t vs Beta, exploratory 3-tier)"),
    out_path=OUT_DIR / "tier3-uncertainty-stacked-t-vs-beta.png",
)
