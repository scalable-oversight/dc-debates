"""Histogram, descriptives (with 95% CI, logit-to-probability conversions),
swarmplot by debate question, and one-sample t-test testing whether
credence_logits_change differs from 0.
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

    COL = "credence_logits_change"
    s = df[COL].dropna()

    n = len(s)
    m = s.mean()
    se = s.std() / np.sqrt(n)
    ci_lo, ci_hi = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)

    report_lines = [
        f"{COL} (logit section 4 - logit section 1)",
        f"  n       = {n}",
        f"  mean    = {m:.4f}  (probability shift from 50%: {logit_to_prob(m):.4f})",
        f"  95% CI  = [{ci_lo:.4f}, {ci_hi:.4f}]"
        f"  (prob = [{logit_to_prob(ci_lo):.4f}, {logit_to_prob(ci_hi):.4f}])",
        f"  median  = {s.median():.4f}  (probability shift from 50%: {logit_to_prob(s.median()):.4f})",
        f"  sd      = {s.std():.4f}",
        f"  min     = {s.min():.4f}  (probability shift from 50%: {logit_to_prob(s.min()):.4f})",
        f"  max     = {s.max():.4f}  (probability shift from 50%: {logit_to_prob(s.max()):.4f})",
        "",
        "  NOTE: logit-to-probability conversions show the probability",
        "  corresponding to each logit value when applied to a 50% baseline.",
        "  A positive logit change means credence shifted toward the correct answer.",
    ]

    t_stat, p_val = stats.ttest_1samp(s, 0)
    t_ci_lo, t_ci_hi = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)

    report_lines += [
        "",
        "One-sample t-test: H0: mean logit change = 0 (no shift in credence)",
        f"  t({n - 1})        = {t_stat:.4f}",
        f"  p               = {p_val:.6f}",
        f"  mean            = {m:.4f}  (prob from 50%: {logit_to_prob(m):.4f})",
        f"  95% CI          = [{t_ci_lo:.4f}, {t_ci_hi:.4f}]"
        f"  (prob = [{logit_to_prob(t_ci_lo):.4f}, {logit_to_prob(t_ci_hi):.4f}])",
        f"  Interpretation  : mean logit change {'differs' if p_val < 0.05 else 'does not differ'}"
        f" significantly from 0 at alpha=0.05.",
    ]

    txt = OUT / "credence_logits_change_descriptives.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Wrote {txt}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(s, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(m, color="red", linestyle="--", label=f"mean = {m:.3f}")
    ax.axvline(0, color="grey", linestyle=":", label="no change (logit=0)")
    ax.set_xlabel("Credence Logits Change")
    ax.set_ylabel("Count")
    ax.set_title("Credence Logits Change (Section 4 - Section 1)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "credence_logits_change_histogram.png", dpi=150)
    print(f"Saved credence_logits_change_histogram.png")
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
    ax.axhline(0, color="grey", linestyle="--", alpha=0.5, label="no change")
    ax.set_xlabel("Debate Question")
    ax.set_ylabel("Credence Logits Change")
    ax.set_title("Credence Logits Change by Debate Question")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUT / "credence_logits_change_by_debate_swarmplot.png", dpi=150)
    print(f"Saved credence_logits_change_by_debate_swarmplot.png")
    plt.close()


if __name__ == "__main__":
    main()
