"""Histogram, descriptives (with 95% CI), and swarmplot by debate question
of section_1_credence_in_correct_answer.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True,
                   help="Path to cleaned quant-1-pilot-2-no-ids.xlsx")
    p.add_argument("--mapping", type=Path, required=True,
                   help="Path to debate-to-name-and-question-mapping.csv")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory for output files")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    OUT = args.out_dir

    df = pd.read_excel(args.input, engine="openpyxl")

    COL = "section_1_credence_in_correct_answer"
    s = df[COL].dropna()

    n = len(s)
    m = s.mean()
    se = s.std() / np.sqrt(n)
    ci_lo, ci_hi = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)

    report = (
        f"{COL}\n"
        f"  n       = {n}\n"
        f"  mean    = {m:.4f}\n"
        f"  95% CI  = [{ci_lo:.4f}, {ci_hi:.4f}]\n"
        f"  median  = {s.median():.4f}\n"
        f"  sd      = {s.std():.4f}\n"
        f"  min     = {s.min():.4f}\n"
        f"  max     = {s.max():.4f}\n"
    )

    txt = OUT / "section1_credence_descriptives.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote {txt}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(s, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(m, color="red", linestyle="--", label=f"mean = {m:.3f}")
    ax.axvline(0.5, color="grey", linestyle=":", label="chance (0.5)")
    ax.set_xlabel("Section 1 Credence in Correct Answer")
    ax.set_ylabel("Count")
    ax.set_title("Section 1 Credence in Correct Answer")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "section1_credence_histogram.png", dpi=150)
    print(f"Saved section1_credence_histogram.png")
    plt.close()

    mapping = pd.read_csv(args.mapping, encoding="utf-8")
    id_to_name = dict(zip(mapping["debate_id"], mapping["debate_question"]))
    df["question_name"] = df["debate_id"].map(id_to_name)

    fig, ax = plt.subplots(figsize=(14, 6))
    order = [id_to_name[i] for i in sorted(id_to_name) if id_to_name[i] in df["question_name"].values]
    order = list(dict.fromkeys(order))
    sns.swarmplot(
        data=df.dropna(subset=[COL]),
        x="question_name", y=COL, ax=ax,
        order=order, alpha=0.6, size=4,
    )
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Debate Question")
    ax.set_ylabel("Section 1 Credence in Correct Answer")
    ax.set_title("Section 1 Credence in Correct Answer by Debate Question")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUT / "section1_credence_by_debate_swarmplot.png", dpi=150)
    print(f"Saved section1_credence_by_debate_swarmplot.png")
    plt.close()


if __name__ == "__main__":
    main()
