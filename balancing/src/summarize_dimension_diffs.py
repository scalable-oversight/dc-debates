#!/usr/bin/env python3
"""Summarise the per-dimension rating CSVs produced by rate_debate_dimensions.py.

This script does NOT call the API — it only reads existing CSVs under
``data/ratings/<dim>/`` and writes a plain-text summary to
``data/ratings/diff_significance_report_by_section.txt``.

For each dimension, the seven per-turn observations per file are collapsed
into four "section" observations, matching the section-level granularity used
in the paper:

  * Opening                                       (1 turn)
  * Section 1 = mean(Argument 1, Response 1)      (2 turns)
  * Section 2 = mean(Argument 2, Response 2)      (2 turns)
  * Section 3 = mean(Argument 3, Response 3)      (2 turns)

Each section gets one A score and one B score per file by averaging the
relevant per-turn medians. The report lists, per dimension: (a) section-level
rows whose A-minus-B mean-of-medians diff is < -1 or > 1, and (b) per-debate
paired t-tests (four paired observations per file, df=3) flagged significant
at α=0.05.

Run with: ``python3 src/summarize_dimension_diffs.py``

Optional flags:
  --ratings-dir PATH    directory containing per-dimension subfolders
                        (default: ``data/ratings/``)
  --out PATH            report output path
                        (default: ``<ratings-dir>/diff_significance_report_by_section.txt``)
  --corpus-label LABEL  override the corpus name shown in the report header
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats as _scipy_stats


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
DEFAULT_RATINGS_DIR = PACKAGE_DIR / "data" / "ratings"
DEFAULT_DEBATES_DIR = PACKAGE_DIR / "data" / "debates"

# Same 11 dimensions as rate_debate_dimensions.py, in the same order.
DIMS = [
    "authoritative_tone",
    "clarity",
    "confidence",
    "defensiveness",
    "eloquence",
    "grammatical_quality",
    "jargon",
    "professionalism",
    "rhetorical_tactics",
    "rudeness",
    "tentativeness",
]

# Section name → list of per-turn labels that make up that section. The
# Opening is its own one-turn "section"; each round combines its Argument and
# Response.
SECTION_DEFS: list[tuple[str, list[str]]] = [
    ("Opening", ["Opening"]),
    ("Section 1", ["Argument 1", "Response 1"]),
    ("Section 2", ["Argument 2", "Response 2"]),
    ("Section 3", ["Argument 3", "Response 3"]),
]


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_float(raw: str) -> float | None:
    """Return raw as a float, or None if blank/unparseable."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# ─────────────────────────── section collapsing ─────────────────────────── #


def _collapse_to_sections(
    pair_rows: list[dict],
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Collapse the 7-turn pair rows of each file into 4 section-level pairs.

    Returns ``{filename: {section_name: (a_score, b_score, diff)}}`` using the
    MEAN of the constituent per-turn medians. If any constituent turn for a
    section has a blank/missing A or B median, that section is omitted for
    the file."""
    by_file: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for r in pair_rows:
        a = _parse_float(r.get("debater_a_median", ""))
        b = _parse_float(r.get("debater_b_median", ""))
        if a is None or b is None:
            continue
        by_file[r["filename"]][r["turn"]] = (a, b)

    sectioned: dict[str, dict[str, tuple[float, float, float]]] = {}
    for fname, turn_map in by_file.items():
        sections: dict[str, tuple[float, float, float]] = {}
        for sec_name, turn_labels in SECTION_DEFS:
            entries = [turn_map[t] for t in turn_labels if t in turn_map]
            if len(entries) != len(turn_labels):
                continue
            a_mean = sum(e[0] for e in entries) / len(entries)
            b_mean = sum(e[1] for e in entries) / len(entries)
            sections[sec_name] = (a_mean, b_mean, a_mean - b_mean)
        sectioned[fname] = sections
    return sectioned


def _paired_t(diffs: list[float]) -> tuple[float, float, float, float] | None:
    """Return (mean_diff, sd_diff, t_statistic, p_value) for a paired t-test on
    ``diffs`` (treated as A-B). Returns None if undefined (n<2 or zero variance)."""
    n = len(diffs)
    if n < 2:
        return None
    mean_diff = sum(diffs) / n
    if len(set(diffs)) == 1:
        return None  # zero variance — t undefined
    var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    sd = var ** 0.5
    # Equivalent to scipy.stats.ttest_1samp(diffs, 0) — i.e. the paired t-test
    # we want, but expressed on the difference vector directly.
    res = _scipy_stats.ttest_1samp(diffs, 0.0)
    return mean_diff, sd, float(res.statistic), float(res.pvalue)


def analyse_dimension_by_section(
    dim: str, ratings_dir: Path, alpha: float = 0.05,
) -> tuple[list[str], dict]:
    """4-pair (Opening + 3 round sections) analysis. Recomputes the t-test."""
    dim_dir = ratings_dir / dim
    pairs_path = dim_dir / f"turn_{dim}_pairs.csv"

    lines: list[str] = []
    lines.append(f"DIMENSION: {dim}")
    lines.append("-" * 72)

    n_extreme = 0
    n_sig = 0
    missing = False

    if not pairs_path.exists():
        lines.append(f"  [pairs]  MISSING FILE: {pairs_path.name}")
        lines.append(f"  [stats]  (skipped — pairs CSV unavailable)")
        lines.append("")
        return lines, {
            "dim": dim, "n_extreme": 0, "n_sig": 0, "missing": True,
        }

    pair_rows = _read_csv(pairs_path)
    sectioned = _collapse_to_sections(pair_rows)

    extreme: list[tuple[str, str, float, float, float]] = []
    total_section_pairs = 0
    for fname in sorted(sectioned):
        for sec_name, _ in SECTION_DEFS:
            triple = sectioned[fname].get(sec_name)
            if triple is None:
                continue
            total_section_pairs += 1
            a, b, d = triple
            if d < -1 or d > 1:
                extreme.append((fname, sec_name, a, b, d))
    n_extreme = len(extreme)
    lines.append(
        f"  [pairs]  {n_extreme} of {total_section_pairs} section rows with "
        f"diff < -1 or diff > 1:"
    )
    if extreme:
        for fname, sec_name, a, b, d in extreme:
            lines.append(
                f"    - {fname} | {sec_name} | "
                f"A={a:g} B={b:g} diff={d:g}"
            )
    else:
        lines.append("    (none)")

    sig_rows: list[tuple[str, float, float, float, str]] = []
    total_files = 0
    for fname in sorted(sectioned):
        diffs = [
            sectioned[fname][sec_name][2]
            for sec_name, _ in SECTION_DEFS
            if sec_name in sectioned[fname]
        ]
        if len(diffs) < 2:
            continue
        total_files += 1
        result = _paired_t(diffs)
        if result is None:
            continue
        mean_diff, _sd, t_stat, p_val = result
        if p_val < alpha:
            direction = "A > B" if mean_diff > 0 else "A < B"
            sig_rows.append((fname, mean_diff, t_stat, p_val, direction))
    n_sig = len(sig_rows)
    lines.append(
        f"  [stats]  {n_sig} of {total_files} rows with a significant "
        f"difference:"
    )
    if sig_rows:
        for fname, mean_diff, t_stat, p_val, direction in sig_rows:
            lines.append(
                f"    - {fname} | "
                f"mean_diff_A_minus_B={mean_diff:.4f} "
                f"t={t_stat:.4f} "
                f"p={p_val:.6f} "
                f"direction={direction}"
            )
    else:
        lines.append("    (none)")

    lines.append("")
    return lines, {
        "dim": dim, "n_extreme": n_extreme, "n_sig": n_sig, "missing": missing,
    }


# ──────────────────────────────── header ───────────────────────────────── #


def derive_corpus_label(ratings_dir: Path) -> str:
    """Best-effort label for the corpus underlying ``ratings_dir``.

    The default vendored layout puts ratings under ``data/ratings/``, so the
    directory name alone isn't informative. We climb up to look for a sibling
    debates directory whose name (e.g. ``data/debates``) describes the
    corpus, falling back to the ratings dir path."""
    parent = ratings_dir.parent
    debates = parent / "debates"
    if debates.is_dir():
        return debates.name
    return ratings_dir.name


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ratings-dir", type=Path, default=DEFAULT_RATINGS_DIR,
        help="directory containing per-dimension subfolders "
             "(default: %(default)s)",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="report output path "
             "(default: <ratings-dir>/diff_significance_report_by_section.txt)",
    )
    p.add_argument(
        "--corpus-label", default=None,
        help="override the corpus name shown in the report header "
             "(default: derived from the sibling debates directory).",
    )
    args = p.parse_args()

    ratings_dir: Path = args.ratings_dir.resolve()
    if not ratings_dir.is_dir():
        print(f"ERROR: ratings dir does not exist: {ratings_dir}", file=sys.stderr)
        return 2

    out_path: Path = (
        args.out.resolve()
        if args.out is not None
        else ratings_dir / "diff_significance_report_by_section.txt"
    )
    corpus_label = args.corpus_label or derive_corpus_label(ratings_dir)

    report: list[str] = []
    report.append("=" * 72)
    report.append(f"DIMENSION DIFF / SIGNIFICANCE REPORT — {corpus_label}")
    report.append(f"Source: {ratings_dir}")
    report.append("Mode: by-section "
                  "(Opening + Section N = mean(Argument N, Response N))")
    report.append("=" * 72)
    report.append("")
    report.append(
        "Per dimension: (a) section-level rows whose A-minus-B mean-median "
        "diff is < -1 or > 1, and (b) per-debate paired t-test "
        "(four paired observations per file, df=3) flagged significant "
        "at α=0.05."
    )
    report.append("")

    summaries: list[dict] = []
    for dim in DIMS:
        lines, summary = analyse_dimension_by_section(dim, ratings_dir)
        report.extend(lines)
        summaries.append(summary)

    report.append("=" * 72)
    report.append("SUMMARY (totals across all dimensions)")
    report.append("=" * 72)
    report.append(f"{'dimension':<24} {'|diff|>1 rows':>14} {'significant rows':>18}")
    total_extreme = total_sig = 0
    for s in summaries:
        flag = "  [FILES MISSING]" if s["missing"] else ""
        report.append(
            f"{s['dim']:<24} {s['n_extreme']:>14} {s['n_sig']:>18}{flag}"
        )
        total_extreme += s["n_extreme"]
        total_sig += s["n_sig"]
    report.append("-" * 72)
    report.append(f"{'TOTAL':<24} {total_extreme:>14} {total_sig:>18}")
    report.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote report to {out_path}")
    print(f"  {total_extreme} pair rows with |diff| > 1; "
          f"{total_sig} significant stats rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
