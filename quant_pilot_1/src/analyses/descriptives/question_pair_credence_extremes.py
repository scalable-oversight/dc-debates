"""For every unordered pair of the 12 debate questions, aggregate all
section-4 credence-in-correct-answer values across both debates of both
questions, then compute a 95% t-CI for the aggregated mean. Report the pair
with the highest lower bound and the pair with the lowest upper bound, plus
top-10 extremes and the full ranking.
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


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

    CRED = "section_4_credence_in_correct_answer"

    df = pd.read_excel(args.input, engine="openpyxl")
    mapping = pd.read_csv(args.mapping, encoding="utf-8")
    id_to_question = dict(zip(mapping["debate_id"], mapping["debate_question"]))
    df["question_name"] = df["debate_id"].map(id_to_question)

    questions = list(dict.fromkeys(
        [id_to_question[i] for i in sorted(id_to_question)]
    ))
    assert len(questions) == 12, f"expected 12 questions, got {len(questions)}"

    rows = []
    for q1, q2 in combinations(questions, 2):
        sub = df[df["question_name"].isin([q1, q2])][CRED].dropna()
        n = len(sub)
        m = sub.mean()
        se = sub.std(ddof=1) / np.sqrt(n)
        lo, hi = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
        rows.append((q1, q2, n, m, lo, hi))

    rows_by_low = sorted(rows, key=lambda r: r[4], reverse=True)
    rows_by_high = sorted(rows, key=lambda r: r[5])

    best_low = rows_by_low[0]
    best_high = rows_by_high[0]

    lines = [
        "Aggregated section-4 credence by question PAIR (pilot 1 only)",
        "=" * 76,
        f"Number of unordered pairs                = {len(rows)}  (C(12,2) = 66)",
        f"Variable                                 = {CRED}",
        "Method                                   = 95% t-CI on the aggregated sample",
        "",
        "ANSWER — pair with the HIGHEST lower CI bound",
        "-" * 76,
        f"  {best_low[0]}  +  {best_low[1]}",
        f"  n = {best_low[2]}  mean = {best_low[3]:.4f}"
        f"  95% CI = [{best_low[4]:.4f}, {best_low[5]:.4f}]",
        "",
        "ANSWER — pair with the LOWEST upper CI bound",
        "-" * 76,
        f"  {best_high[0]}  +  {best_high[1]}",
        f"  n = {best_high[2]}  mean = {best_high[3]:.4f}"
        f"  95% CI = [{best_high[4]:.4f}, {best_high[5]:.4f}]",
        "",
        "Top 10 pairs by lower bound (highest first)",
        "-" * 76,
        f"  {'pair':<60} {'n':>4} {'mean':>7}  {'CI':>22}",
    ]
    for q1, q2, n, m, lo, hi in rows_by_low[:10]:
        lines.append(f"  {q1 + ' + ' + q2:<60} {n:>4} {m:>7.4f}  [{lo:.4f}, {hi:.4f}]")

    lines += [
        "",
        "Top 10 pairs by upper bound (lowest first)",
        "-" * 76,
        f"  {'pair':<60} {'n':>4} {'mean':>7}  {'CI':>22}",
    ]
    for q1, q2, n, m, lo, hi in rows_by_high[:10]:
        lines.append(f"  {q1 + ' + ' + q2:<60} {n:>4} {m:>7.4f}  [{lo:.4f}, {hi:.4f}]")

    lines += [
        "",
        "Full ranking (sorted alphabetically by first question, then second)",
        "-" * 76,
        f"  {'pair':<60} {'n':>4} {'mean':>7}  {'CI':>22}",
    ]
    for q1, q2, n, m, lo, hi in sorted(rows, key=lambda r: (r[0], r[1])):
        lines.append(f"  {q1 + ' + ' + q2:<60} {n:>4} {m:>7.4f}  [{lo:.4f}, {hi:.4f}]")

    report = "\n".join(lines) + "\n"
    print(report)

    txt = OUT / "question_pair_credence_extremes.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote {txt}")


if __name__ == "__main__":
    main()
