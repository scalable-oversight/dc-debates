"""Histogram, descriptives (with 95% CI), and swarmplot by debate question
of credence_change (section 4 - section 1).
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

    COL = "credence_change"
    s = df[COL].dropna()

    n = len(s)
    m = s.mean()
    se = s.std() / np.sqrt(n)
    ci_lo, ci_hi = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)

    report = (
        f"{COL} (section 4 - section 1 credence in correct answer)\n"
        f"  n       = {n}\n"
        f"  mean    = {m:.4f}\n"
        f"  95% CI  = [{ci_lo:.4f}, {ci_hi:.4f}]\n"
        f"  median  = {s.median():.4f}\n"
        f"  sd      = {s.std():.4f}\n"
        f"  min     = {s.min():.4f}\n"
        f"  max     = {s.max():.4f}\n"
    )

    txt = OUT / "credence_change_descriptives.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote {txt}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(s, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(m, color="red", linestyle="--", label=f"mean = {m:.4f}")
    ax.axvline(0, color="grey", linestyle=":", label="no change")
    ax.set_xlabel("Credence Change (Section 4 - Section 1)")
    ax.set_ylabel("Count")
    ax.set_title("Credence Change")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "credence_change_histogram.png", dpi=150)
    print(f"Saved credence_change_histogram.png")
    plt.close()

    mapping = pd.read_csv(args.mapping, encoding="utf-8")
    id_to_name = dict(zip(mapping["debate_id"], mapping["debate_question"]))
    df["question_name"] = df["debate_id"].map(id_to_name)
    df["debate_order"] = np.where(df["debate_id"] % 2 == 1, "First", "Second")

    DEBATE_ORDER_PALETTE = {"First": "#1f77b4", "Second": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=(14, 6))
    order = [id_to_name[i] for i in sorted(id_to_name) if id_to_name[i] in df["question_name"].values]
    order = list(dict.fromkeys(order))
    sns.swarmplot(
        data=df.dropna(subset=[COL]),
        x="question_name", y=COL, hue="debate_order", ax=ax,
        order=order, hue_order=["First", "Second"], palette=DEBATE_ORDER_PALETTE,
        dodge=False, alpha=0.6, size=4,
    )
    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
    ax.set_xlabel("Debate Question")
    ax.set_ylabel("Credence Change")
    ax.set_title("Credence Change by Debate Question")
    ax.legend(title="Debate (within question)", loc="best")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUT / "credence_change_by_debate_swarmplot.png", dpi=150)
    print(f"Saved credence_change_by_debate_swarmplot.png")
    plt.close()


if __name__ == "__main__":
    main()
