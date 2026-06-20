"""Per-question comparison of section 1 vs section 4 credence in the correct
answer, on the LOGIT scale. For each of the 12 debate questions, reports n,
mean, and 95% CI for both sections' logit credence, and flags whether the two
95% CIs overlap.

The logit columns (logit_section_1/4_credence_in_correct_answer) are already
in the dataset; credence_logits_change is logit(section 4) - logit(section 1)
per participant. Working on the logit scale makes the additive within-
participant shift the natural unit. Logit means are annotated with their
probability equivalents (logistic of the logit) for readability.

Non-overlapping 95% CIs imply a significant difference, but overlapping CIs
do not imply the absence of one (the CI-overlap test is conservative). A
paired test per question is reported alongside for that reason.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def logit_to_prob(x):
    return 1.0 / (1.0 + np.exp(-x))


def describe(s):
    """n, mean, 95% CI (t-interval) for a series."""
    s = s.dropna()
    n = len(s)
    m = s.mean()
    se = s.std() / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
    return n, m, lo, hi


def cis_overlap(a_lo, a_hi, b_lo, b_hi):
    return bool((a_lo <= b_hi) and (b_lo <= a_hi))


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

    S1 = "logit_section_1_credence_in_correct_answer"
    S4 = "logit_section_4_credence_in_correct_answer"

    order = [id_to_name[i] for i in sorted(id_to_name) if id_to_name[i] in df["question_name"].values]
    order = list(dict.fromkeys(order))

    lines = [
        "Section 1 vs Section 4 credence in correct answer (LOGIT scale), by debate question",
        "=" * 82,
        "Each block: n, mean, 95% CI for section 1 and section 4 logit credence,",
        "whether the two 95% CIs overlap, and a paired t-test (section 4 - section 1)",
        "for that question. Non-overlapping CIs => significant difference; overlapping",
        "CIs do NOT rule one out (see the paired test).",
        "",
        "Logit values are annotated with their probability equivalents in parentheses",
        "(prob = logistic(logit)). A positive logit difference means credence shifted",
        "toward the correct answer over the course of the debate.",
        "",
    ]

    n_overlap = 0
    n_no_overlap = 0
    summary_rows = []

    for q in order:
        sub = df[df["question_name"] == q]
        paired = sub.dropna(subset=[S1, S4])

        n1, m1, lo1, hi1 = describe(sub[S1])
        n4, m4, lo4, hi4 = describe(sub[S4])

        overlap = cis_overlap(lo1, hi1, lo4, hi4)
        if overlap:
            n_overlap += 1
        else:
            n_no_overlap += 1

        diff = paired[S4] - paired[S1]
        if len(diff) > 1:
            t_stat, p_val = stats.ttest_rel(paired[S4], paired[S1])
            paired_str = (f"  paired t-test (s4 - s1): n={len(diff)}, "
                          f"mean logit diff={diff.mean():+.4f}, "
                          f"t({len(diff) - 1})={t_stat:.3f}, p={p_val:.4f}")
        else:
            paired_str = "  paired t-test (s4 - s1): n/a (insufficient paired data)"

        lines += [
            f"{q}",
            "-" * 82,
            f"  Section 1: n={n1}, mean logit={m1:+.4f} (prob={logit_to_prob(m1):.4f}), "
            f"95% CI=[{lo1:+.4f}, {hi1:+.4f}]",
            f"  Section 4: n={n4}, mean logit={m4:+.4f} (prob={logit_to_prob(m4):.4f}), "
            f"95% CI=[{lo4:+.4f}, {hi4:+.4f}]",
            f"  95% CIs overlap: {'YES' if overlap else 'NO'}",
            paired_str,
            "",
        ]

        summary_rows.append((q, overlap, m1, m4, m4 - m1))

    lines += [
        "Summary",
        "=" * 82,
        f"  Questions with overlapping section 1 / section 4 logit 95% CIs:   {n_overlap} / {len(order)}",
        f"  Questions with non-overlapping (separated) logit 95% CIs:         {n_no_overlap} / {len(order)}",
        "",
        "  Quick view (question : CI overlap : mean logit s1 -> mean logit s4 : difference):",
    ]
    for q, overlap, m1, m4, d in summary_rows:
        lines.append(
            f"    {q:<32} {'overlap ' if overlap else 'SEPARATE'}  "
            f"{m1:+.3f} -> {m4:+.3f}  (diff {d:+.3f})"
        )

    txt = OUT / "section1_vs_section4_credence_logits_by_question.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {txt}")


if __name__ == "__main__":
    main()
