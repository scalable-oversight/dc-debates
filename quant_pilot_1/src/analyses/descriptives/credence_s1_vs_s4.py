"""Plot each participant's credence at section 1 (blue dot) and section 4
(purple dot) for every debate question, connected by a thin line coloured
green (moved toward 1.0) or burgundy (moved toward 0.0).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True,
                   help="Path to cleaned quant-1-pilot-no-ids.xlsx")
    p.add_argument("--mapping", type=Path, required=True,
                   help="Path to debate-to-name-and-question-mapping.csv")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory for output files")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    OUT = args.out_dir

    df = pd.read_excel(args.input, engine="openpyxl")
    mapping = pd.read_csv(args.mapping, encoding="utf-8")
    id_to_name = dict(zip(mapping["debate_id"], mapping["debate_question"]))
    df["question_name"] = df["debate_id"].map(id_to_name)

    S1 = "section_1_credence_in_correct_answer"
    S4 = "section_4_credence_in_correct_answer"

    sub = df[[S1, S4, "question_name"]].dropna()

    ordered_names = [id_to_name[i] for i in sorted(id_to_name)]
    seen = set()
    questions = []
    for n in ordered_names:
        if n not in seen and n in sub["question_name"].values:
            seen.add(n)
            questions.append(n)

    q_to_x = {q: i for i, q in enumerate(questions)}

    fig, ax = plt.subplots(figsize=(14, 6))

    JITTER_HALF = 0.08
    OFFSET = 0.06

    rng = np.random.default_rng(42)

    for _, row in sub.iterrows():
        x_base = q_to_x[row["question_name"]] + rng.uniform(-JITTER_HALF, JITTER_HALF)
        y1 = row[S1]
        y4 = row[S4]

        colour = "#228B22" if y4 >= y1 else "#800020"

        ax.plot([x_base, x_base + OFFSET], [y1, y4],
                color=colour, linewidth=0.7, zorder=1)
        ax.scatter(x_base, y1, color="#4169E1", s=12, zorder=2)
        ax.scatter(x_base + OFFSET, y4, color="#7B2D8E", s=12, zorder=2)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xticks(range(len(questions)))
    ax.set_xticklabels(questions, rotation=45, ha="right")
    ax.set_ylabel("Credence in Correct Answer")
    ax.set_title("Credence at Section 1 vs Section 4 by Debate Question")
    ax.set_xlim(-0.5, len(questions) - 0.3)
    ax.set_ylim(-0.02, 1.02)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4169E1',
               markersize=6, label='Section 1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#7B2D8E',
               markersize=6, label='Section 4'),
        Line2D([0], [0], color='#228B22', linewidth=1, label='Moved toward 1.0'),
        Line2D([0], [0], color='#800020', linewidth=1, label='Moved toward 0.0'),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    plt.tight_layout()
    out_path = OUT / "credence_s1_vs_s4_by_debate.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
