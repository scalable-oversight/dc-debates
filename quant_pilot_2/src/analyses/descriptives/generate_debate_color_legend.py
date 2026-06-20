"""Generate a legend image showing which debate names correspond to which
color (first vs. second debate within each question) in the by-debate
swarmplots.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DEBATE_ORDER_PALETTE = {"First": "#1f77b4", "Second": "#ff7f0e"}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=False, default=None,
                   help="(Unused — accepted for pipeline uniformity)")
    p.add_argument("--mapping", type=Path, required=True,
                   help="Path to debate-to-name-and-question-mapping.csv")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory for output files")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    OUT = args.out_dir

    mapping = pd.read_csv(args.mapping, encoding="utf-8").sort_values("debate_id").reset_index(drop=True)

    rows = []
    for q in mapping["debate_question"].drop_duplicates():
        g = mapping[mapping["debate_question"] == q].sort_values("debate_id")
        first = g.iloc[0]["debate_name"]
        second = g.iloc[1]["debate_name"] if len(g) > 1 else ""
        rows.append((q, first, second))

    n = len(rows)
    fig, ax = plt.subplots(figsize=(14, max(4, 0.45 * n + 2)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 2)
    ax.set_axis_off()

    ax.text(0.5, n + 1.4, "Debate Color Legend", fontsize=16, fontweight="bold", ha="center")

    ax.text(0.02, n + 0.4, "Question", fontsize=11, fontweight="bold")
    ax.text(0.30, n + 0.4, "First debate", fontsize=11, fontweight="bold",
            color=DEBATE_ORDER_PALETTE["First"])
    ax.text(0.63, n + 0.4, "Second debate", fontsize=11, fontweight="bold",
            color=DEBATE_ORDER_PALETTE["Second"])

    for i, (q, first, second) in enumerate(rows):
        y = n - 1 - i + 0.5
        ax.text(0.02, y, q, fontsize=10, va="center")
        ax.scatter([0.295], [y], s=120, color=DEBATE_ORDER_PALETTE["First"],
                   edgecolor="black", linewidth=0.5)
        ax.text(0.31, y, first, fontsize=10, va="center", family="monospace")
        ax.scatter([0.625], [y], s=120, color=DEBATE_ORDER_PALETTE["Second"],
                   edgecolor="black", linewidth=0.5)
        ax.text(0.64, y, second, fontsize=10, va="center", family="monospace")

    plt.tight_layout()
    out_path = OUT / "debate_color_legend.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path.name}")
    plt.close()


if __name__ == "__main__":
    main()
