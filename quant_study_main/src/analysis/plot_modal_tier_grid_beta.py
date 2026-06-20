"""Place each transcript in a 3 (domain) x 4 (modal tier) grid -- Beta fit.

Beta-model analogue of plot_modal_tier_grid.py. Reads
tier-classification-secondary-beta.csv (produced by
derive_tiers_secondary_beta.py) and emits:

  output/modal-tier-grid-secondary-beta.png

Rows (top -> bottom): Astrophysics, Brainteasers, Reward Hacking.
Columns (left -> right): Tier 1 (easiest), Tier 2, Tier 3, Tier 4 (hardest).
Each cell lists the transcripts whose modal tier is that column and whose
domain is that row, formatted "<question name> <1 or 2>".
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

OUT_DIR = Path(os.environ.get(
    "ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent / "output")
))

DOMAIN_DISPLAY = [
    ("astrophysics", "Astrophysics"),
    ("brainteasers", "Brainteasers"),
    ("coding", "Reward\nHacking"),
]
TIER_LABELS = [
    "Tier 1\n(easiest)",
    "Tier 2",
    "Tier 3",
    "Tier 4\n(hardest)",
]
N_TIERS = 4

DOMAIN_FACE = {
    "astrophysics": "#eaf2fb",
    "brainteasers": "#fbecec",
    "coding": "#eaf6ec",
}

QUESTION_DISPLAY = {
    "ORM library": "ORM Library",
    "Database deletion": "Database Deletion",
    "Probability puzzle": "Probability Puzzle",
    "Graviational Lens Modelling": "Gravitational Lens Modelling",
}


def transcript_labels(df):
    df = df.sort_values(["question", "transcript"]).copy()
    df["within_q_idx"] = df.groupby("question").cumcount() + 1
    df["question_display"] = df["question"].map(QUESTION_DISPLAY).fillna(df["question"])
    df["label"] = df["question_display"] + " " + df["within_q_idx"].astype(str)
    return df


def plot(in_csv, out_path, title_suffix):
    df = pd.read_csv(in_csv, encoding="utf-8")
    df = transcript_labels(df)

    fig, ax = plt.subplots(figsize=(14, 7))
    n_rows = len(DOMAIN_DISPLAY)
    n_cols = N_TIERS

    for ci in range(n_cols):
        for ri, (dom_key, dom_label) in enumerate(DOMAIN_DISPLAY):
            y_top = n_rows - ri
            y_bot = y_top - 1
            x_left = ci
            x_right = ci + 1
            rect = Rectangle(
                (x_left, y_bot), 1, 1,
                facecolor=DOMAIN_FACE[dom_key],
                edgecolor="#888888", linewidth=0.8,
            )
            ax.add_patch(rect)

            cell_df = df[(df["domain"] == dom_key) & (df["modal_tier"] == ci + 1)]
            if len(cell_df) == 0:
                continue

            labels = cell_df.sort_values("delta_mean")["label"].tolist()
            text = "\n".join(labels)
            ax.text(
                (x_left + x_right) / 2, (y_bot + y_top) / 2, text,
                ha="center", va="center", fontsize=11, color="#222222",
                linespacing=1.6,
            )

    for ci in range(n_cols):
        ax.text(
            ci + 0.5, n_rows + 0.18, TIER_LABELS[ci],
            ha="center", va="bottom", fontsize=13.2,
            color="#222222",
        )
    for ri, (_, dom_label) in enumerate(DOMAIN_DISPLAY):
        y_top = n_rows - ri
        ax.text(
            -0.05, y_top - 0.5, dom_label,
            ha="right", va="center", fontsize=13.2,
            color="#222222",
        )

    ax.set_xlim(-0.05, n_cols + 0.05)
    ax.set_ylim(-0.05, n_rows + 0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        title_suffix,
        fontsize=15.6, pad=22,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.name}")


plot(
    OUT_DIR / "tier-classification-secondary-beta.csv",
    OUT_DIR / "modal-tier-grid-secondary-beta.png",
    title_suffix="Modal Tier × Domain, secondary outcome model (exploratory beta)",
)
