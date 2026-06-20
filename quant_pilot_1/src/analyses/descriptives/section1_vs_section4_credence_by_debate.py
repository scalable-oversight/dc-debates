"""Per-debate comparison of section 1 vs section 4 credence in the correct
answer. For each of the 24 debates, reports n, mean, and 95% CI for both
sections, and flags whether the two 95% CIs overlap.

This is the by-debate counterpart of
section1_vs_section4_credence_by_question.py: the 24 debates pair up into
12 questions, and that script aggregates each pair into one question. Here
each debate is kept separate, so per-debate n is smaller and some debates may
have too few participants to compute a CI or a paired test.

Non-overlapping 95% CIs imply a significant difference, but overlapping CIs
do not imply the absence of one (the CI-overlap test is conservative). A
paired test per debate is reported alongside for that reason.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def describe(s):
    """n, mean, 95% CI (t-interval) for a series. CI is nan when n < 2."""
    s = s.dropna()
    n = len(s)
    if n == 0:
        return 0, np.nan, np.nan, np.nan
    m = s.mean()
    if n < 2:
        return n, m, np.nan, np.nan
    se = s.std() / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
    return n, m, lo, hi


def cis_overlap(a_lo, a_hi, b_lo, b_hi):
    """True/False if both CIs are defined, None if either is undefined (nan)."""
    if any(np.isnan(x) for x in (a_lo, a_hi, b_lo, b_hi)):
        return None
    return bool((a_lo <= b_hi) and (b_lo <= a_hi))


def fmt_ci(lo, hi):
    if np.isnan(lo) or np.isnan(hi):
        return "n/a (n < 2)"
    return f"[{lo:.4f}, {hi:.4f}]"


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
    id_to_name = dict(zip(mapping["debate_id"], mapping["debate_name"]))
    id_to_question = dict(zip(mapping["debate_id"], mapping["debate_question"]))
    df["debate_name"] = df["debate_id"].map(id_to_name)

    S1 = "section_1_credence_in_correct_answer"
    S4 = "section_4_credence_in_correct_answer"

    order = [i for i in sorted(id_to_name) if id_to_name[i] in df["debate_name"].values]

    lines = [
        "Section 1 vs Section 4 credence in correct answer, by debate",
        "=" * 72,
        "Each block: n, mean, 95% CI for section 1 and section 4, whether the two",
        "95% CIs overlap, and a paired t-test (section 4 - section 1) for that",
        "debate. Non-overlapping CIs => significant difference; overlapping CIs",
        "do NOT rule one out (see the paired test). Debates with too few",
        "participants to compute a CI or paired test are reported as n/a.",
        "",
    ]

    n_overlap = 0
    n_no_overlap = 0
    n_undetermined = 0
    summary_rows = []

    for debate_id in order:
        name = id_to_name[debate_id]
        question = id_to_question[debate_id]
        sub = df[df["debate_id"] == debate_id]
        paired = sub.dropna(subset=[S1, S4])

        n1, m1, lo1, hi1 = describe(sub[S1])
        n4, m4, lo4, hi4 = describe(sub[S4])

        overlap = cis_overlap(lo1, hi1, lo4, hi4)
        if overlap is True:
            n_overlap += 1
            overlap_str = "YES"
        elif overlap is False:
            n_no_overlap += 1
            overlap_str = "NO"
        else:
            n_undetermined += 1
            overlap_str = "n/a (insufficient data)"

        diff = paired[S4] - paired[S1]
        if len(diff) > 1:
            t_stat, p_val = stats.ttest_rel(paired[S4], paired[S1])
            paired_str = (f"  paired t-test (s4 - s1): n={len(diff)}, "
                          f"mean diff={diff.mean():+.4f}, t({len(diff) - 1})={t_stat:.3f}, p={p_val:.4f}")
        else:
            paired_str = "  paired t-test (s4 - s1): n/a (insufficient paired data)"

        lines += [
            f"[{debate_id}] {name}  (question: {question})",
            "-" * 72,
            f"  Section 1: n={n1}, mean={m1:.4f}, 95% CI={fmt_ci(lo1, hi1)}",
            f"  Section 4: n={n4}, mean={m4:.4f}, 95% CI={fmt_ci(lo4, hi4)}",
            f"  95% CIs overlap: {overlap_str}",
            paired_str,
            "",
        ]

        summary_rows.append((debate_id, name, question, overlap, m1, m4))

    lines += [
        "Summary",
        "=" * 72,
        f"  Debates with overlapping section 1 / section 4 95% CIs:       {n_overlap} / {len(order)}",
        f"  Debates with non-overlapping (separated) 95% CIs:             {n_no_overlap} / {len(order)}",
        f"  Debates with insufficient data to determine CI overlap:       {n_undetermined} / {len(order)}",
        "",
        "  Quick view (debate : CI overlap : mean s1 -> mean s4):",
    ]
    for debate_id, name, question, overlap, m1, m4 in summary_rows:
        if overlap is True:
            tag = "overlap "
        elif overlap is False:
            tag = "SEPARATE"
        else:
            tag = "n/a     "
        label = f"[{debate_id}] {name}"
        lines.append(f"    {label:<42} {tag}  {m1:.3f} -> {m4:.3f}")

    txt = OUT / "section1_vs_section4_credence_by_debate.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {txt}")


if __name__ == "__main__":
    main()
