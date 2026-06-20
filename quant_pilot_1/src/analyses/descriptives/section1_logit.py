"""Histogram, descriptives (with 95% CI, logit-to-probability conversions),
swarmplot by debate question, and a one-sample t-test testing whether
logit_section_1_credence_in_correct_answer differs from 0.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def logit_to_prob(x):
    return 1.0 / (1.0 + np.exp(-x))


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

    COL = "logit_section_1_credence_in_correct_answer"
    s = df[COL].dropna()

    n = len(s)
    m = s.mean()
    se = s.std() / np.sqrt(n)
    ci_lo, ci_hi = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)

    report_lines = [
        f"{COL}",
        f"  n       = {n}",
        f"  mean    = {m:.4f}  (probability = {logit_to_prob(m):.4f})",
        f"  95% CI  = [{ci_lo:.4f}, {ci_hi:.4f}]"
        f"  (prob = [{logit_to_prob(ci_lo):.4f}, {logit_to_prob(ci_hi):.4f}])",
        f"  median  = {s.median():.4f}  (probability = {logit_to_prob(s.median()):.4f})",
        f"  sd      = {s.std():.4f}",
        f"  min     = {s.min():.4f}  (probability = {logit_to_prob(s.min()):.4f})",
        f"  max     = {s.max():.4f}  (probability = {logit_to_prob(s.max()):.4f})",
    ]

    t_stat, p_val = stats.ttest_1samp(s, 0)
    t_ci_lo, t_ci_hi = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)

    report_lines += [
        "",
        "One-sample t-test: H0: mean logit = 0 (i.e. 50% credence)",
        f"  t({n - 1})        = {t_stat:.4f}",
        f"  p               = {p_val:.6f}",
        f"  mean            = {m:.4f}  (probability = {logit_to_prob(m):.4f})",
        f"  95% CI          = [{t_ci_lo:.4f}, {t_ci_hi:.4f}]"
        f"  (prob = [{logit_to_prob(t_ci_lo):.4f}, {logit_to_prob(t_ci_hi):.4f}])",
        f"  Interpretation  : mean logit {'differs' if p_val < 0.05 else 'does not differ'}"
        f" significantly from 0 at alpha=0.05.",
    ]

    txt = OUT / "section1_logit_descriptives.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Wrote {txt}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(s, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(m, color="red", linestyle="--", label=f"mean = {m:.3f}")
    ax.axvline(0, color="grey", linestyle=":", label="logit = 0 (50%)")
    ax.set_xlabel("Logit Section 1 Credence in Correct Answer")
    ax.set_ylabel("Count")
    ax.set_title("Logit Section 1 Credence in Correct Answer")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "section1_logit_histogram.png", dpi=150)
    print(f"Saved section1_logit_histogram.png")
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
    ax.axhline(0, color="grey", linestyle="--", alpha=0.5, label="logit = 0 (50%)")
    ax.set_xlabel("Debate Question")
    ax.set_ylabel("Logit Section 1 Credence in Correct Answer")
    ax.set_title("Logit Section 1 Credence by Debate Question")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUT / "section1_logit_by_debate_swarmplot.png", dpi=150)
    print(f"Saved section1_logit_by_debate_swarmplot.png")
    plt.close()


if __name__ == "__main__":
    main()
