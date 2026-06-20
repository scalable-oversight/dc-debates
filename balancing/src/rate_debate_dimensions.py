#!/usr/bin/env python3
"""Rate every turn of every debate on 11 stylistic dimensions, 5 times per turn.

This is the diagnostic rating step of the balancing pipeline described in
the ``Balancing`` section of the paper. For each (dimension, debate,
run):

  1. Read the corresponding ``<dim>_5-point-in-context.txt`` prompt from
     ``prompts/`` and substitute the debate transcript into ``{{TEXT}}``.
  2. Send the prompt to Claude Sonnet 4.6.
  3. Parse the trailing ``RATINGS:`` block to recover one integer 1-5 rating
     per turn (14 turns per debate).

Five independent runs per (dimension, debate) give five ratings per turn; the
median is taken as the canonical score. Per dimension, three CSVs are written
under ``data/ratings/<dim>/``:

  * ``turn_<dim>.csv``       — long format: filename, debater, turn,
    result_1..result_5, median_result (5 × 14 ratings per debate file).
  * ``turn_<dim>_pairs.csv`` — one row per (filename, turn label) with each
    debater's median for that turn label and the A−B diff.
  * ``turn_<dim>_stats.csv`` — per-debate paired t-test on the per-turn
    medians (two-tailed, α=0.05).

The section-level summary report quoted in the paper
(``diff_significance_report_by_section.txt``) is produced by the companion
``summarize_dimension_diffs.py`` script, which reads the per-dimension CSVs
without re-calling the API.

Raw API responses are cached under ``data/ratings/<dim>/raw_responses/<file>/run_<k>.txt``;
re-runs reuse the cache, so deleting specific files forces a re-call. The
cache shipped with the package is the rerevised round (final balancing
pass) used to compute the paper's reported numbers.

Concurrency: up to 8 concurrent API calls (asyncio.Semaphore).

Run with:

    ANTHROPIC_API_KEY=... python3 src/rate_debate_dimensions.py

Optional flags:
  --dims authoritative_tone,clarity   restrict to a subset of dimensions
  --debates-dir PATH                  directory of ``branch_*.txt`` transcripts
                                      (default: ``data/debates/``)
  --prompts-dir PATH                  directory of ``<dim>_5-point-in-context.txt``
                                      rubric prompts (default: ``prompts/``)
  --output-dir PATH                   where CSVs / cache are written
                                      (default: ``data/ratings/``)
  --dry-run                           report cache hits/misses without API calls
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import anthropic
from scipy import stats
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


# ──────────────────────────── paths ──────────────────────────── #

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
DEFAULT_PROMPTS_DIR = PACKAGE_DIR / "prompts"
DEFAULT_DEBATES_DIR = PACKAGE_DIR / "data" / "debates"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "data" / "ratings"

# Glob used to discover transcripts. Restricts to ``branch_*.txt`` so any
# sidecar files (e.g. ``*_warnings.txt``) in the debates directory are skipped.
DEBATE_GLOB = "branch_*.txt"


# ──────────────────────────── configuration ──────────────────────────── #

MODEL = "claude-sonnet-4-6"
RUNS_PER_FILE = 5
CONCURRENCY = 8
MAX_TOKENS = 8000
TEMPERATURE = 1.0
PARSE_RETRY_ATTEMPTS = 4  # up to 4 calls per (file, run); pick best.

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

# (debater, turn_label) tuples in the canonical file order. 14 turns total.
EXPECTED_TURNS: list[tuple[str, str]] = [
    ("A", "Opening"),
    ("B", "Argument 1"),
    ("A", "Response 1"),
    ("B", "Argument 2"),
    ("A", "Response 2"),
    ("B", "Argument 3"),
    ("A", "Response 3"),
    ("B", "Opening"),
    ("A", "Argument 1"),
    ("B", "Response 1"),
    ("A", "Argument 2"),
    ("B", "Response 2"),
    ("A", "Argument 3"),
    ("B", "Response 3"),
]

# Canonical turn labels for the per-turn pair rows (one row per label, A vs B).
TURN_ORDER = [
    "Opening",
    "Argument 1",
    "Response 1",
    "Argument 2",
    "Response 2",
    "Argument 3",
    "Response 3",
]

# Matches lines like "A | Opening | [[3]]" or "B | Argument 1 | [[4]]".
RATING_LINE_RE = re.compile(
    r"^([AB])\s*\|\s*([^|\n]+?)\s*\|\s*\[\[\s*([1-5])\s*\]\]\s*$",
    re.MULTILINE,
)


# ──────────────────────────── API plumbing ──────────────────────────── #


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        # Retry 5xx, treat 4xx as terminal (bad request, auth, etc.).
        return exc.status_code >= 500
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
async def _call_claude_once(client: anthropic.AsyncAnthropic, prompt: str) -> str:
    """Single API call with tenacity-backed retry on transient failures."""
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    return "\n".join(b.text for b in resp.content if b.type == "text")


# ──────────────────────────── parsing ──────────────────────────── #


def parse_ratings(text: str) -> dict[tuple[str, str], int]:
    """Extract (debater, turn_label) → integer rating from a response body.

    Scans the entire response, not just the trailing block, because the model
    sometimes interleaves the RATINGS table with surrounding prose. The regex
    requires the pipe-delimited "[[N]]" format so it only matches actual rating
    rows, not the explanatory paragraphs.
    """
    out: dict[tuple[str, str], int] = {}
    for m in RATING_LINE_RE.finditer(text):
        debater = m.group(1)
        turn_label = m.group(2).strip()
        rating = int(m.group(3))
        out[(debater, turn_label)] = rating
    return out


def is_complete(ratings: dict[tuple[str, str], int]) -> bool:
    return all(k in ratings for k in EXPECTED_TURNS)


def completeness(ratings: dict[tuple[str, str], int]) -> int:
    return sum(1 for k in EXPECTED_TURNS if k in ratings)


# ──────────────────────────── per-run orchestration ──────────────────────────── #


async def get_one_run(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    prompt: str,
    cache_path: Path,
    label: str,
) -> tuple[str, dict[tuple[str, str], int]]:
    """Produce one response (cached). If a parse is incomplete, the call is
    retried up to PARSE_RETRY_ATTEMPTS times; the most-complete attempt is
    cached. Returns (raw_text, parsed_ratings)."""

    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
        return text, parse_ratings(text)

    best_text = ""
    best_ratings: dict[tuple[str, str], int] = {}
    best_score = -1
    for attempt in range(1, PARSE_RETRY_ATTEMPTS + 1):
        async with sem:
            try:
                text = await _call_claude_once(client, prompt)
            except Exception as exc:  # noqa: BLE001 — surface but continue
                print(
                    f"ERROR: API failure on {label} attempt {attempt}: {exc!r}",
                    file=sys.stderr,
                )
                continue
        ratings = parse_ratings(text)
        score = completeness(ratings)
        if score > best_score:
            best_text, best_ratings, best_score = text, ratings, score
        if is_complete(ratings):
            break
        print(
            f"WARN: incomplete ratings on {label} attempt {attempt} "
            f"({score}/{len(EXPECTED_TURNS)} turns parsed)",
            file=sys.stderr,
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(best_text, encoding="utf-8")
    return best_text, best_ratings


# ──────────────────────────── CSV builders ──────────────────────────── #


def _csv_write(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def build_long_rows(
    files: list[str],
    per_turn: dict[tuple[str, str, str], list],
) -> list[dict]:
    """One row per (filename, debater, turn): result_1..result_5 + median."""
    rows: list[dict] = []
    for fname in files:
        for deb, turn in EXPECTED_TURNS:
            ratings = per_turn.get(
                (fname, deb, turn), [None] * RUNS_PER_FILE
            )
            row: dict = {"filename": fname, "debater": deb, "turn": turn}
            for i, r in enumerate(ratings, 1):
                row[f"result_{i}"] = "" if r is None else r
            present = [r for r in ratings if r is not None]
            row["median_result"] = median(present) if present else ""
            rows.append(row)
    return rows


def build_pair_rows(files: list[str], long_rows: list[dict]) -> list[dict]:
    """One row per (filename, turn) with both debaters' medians and the diff."""
    by_file_turn: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for r in long_rows:
        if r["median_result"] == "":
            continue
        by_file_turn[r["filename"]][(r["debater"], r["turn"])] = r["median_result"]

    pair_rows: list[dict] = []
    for fname in files:
        d = by_file_turn.get(fname, {})
        for turn in TURN_ORDER:
            a = d.get(("A", turn))
            b = d.get(("B", turn))
            if a is None or b is None:
                pair_rows.append({
                    "filename": fname,
                    "turn": turn,
                    "debater_a_median": "" if a is None else a,
                    "debater_b_median": "" if b is None else b,
                    "diff": "",
                })
                continue
            pair_rows.append({
                "filename": fname,
                "turn": turn,
                "debater_a_median": a,
                "debater_b_median": b,
                "diff": a - b,
            })
    return pair_rows


def build_stats_rows(files: list[str], pair_rows: list[dict]) -> list[dict]:
    """Per-debate paired t-test on the per-turn medians (two-tailed, α=0.05)."""
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in pair_rows:
        if r["diff"] == "":
            continue
        by_file[r["filename"]].append(r)

    stats_rows: list[dict] = []
    for fname in files:
        rs = by_file.get(fname, [])
        n = len(rs)
        a_vec = [float(r["debater_a_median"]) for r in rs]
        b_vec = [float(r["debater_b_median"]) for r in rs]

        if n == 0:
            mean_a = mean_b = mean_diff = sd_diff = float("nan")
            t_stat = p_val = float("nan")
            df: float | int = float("nan")
        else:
            mean_a = sum(a_vec) / n
            mean_b = sum(b_vec) / n
            diffs = [a - b for a, b in zip(a_vec, b_vec)]
            mean_diff = sum(diffs) / n
            if n >= 2 and len(set(diffs)) > 1:
                var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
                sd_diff = var_diff ** 0.5
                result = stats.ttest_rel(a_vec, b_vec)
                t_stat = float(result.statistic)
                p_val = float(result.pvalue)
                df = n - 1
            else:
                sd_diff = 0.0 if n >= 2 else float("nan")
                t_stat = float("nan")
                p_val = float("nan")
                df = n - 1 if n >= 2 else float("nan")

        if n >= 2 and p_val == p_val and p_val < 0.05:
            direction = "A > B" if mean_diff > 0 else "A < B"
        elif n >= 2:
            direction = "n.s."
        else:
            direction = ""

        def _fmt(x: float, places: int = 4) -> str | float:
            return "" if isinstance(x, float) and x != x else round(x, places)

        stats_rows.append({
            "filename": fname,
            "n_pairs": n,
            "mean_A": _fmt(mean_a),
            "mean_B": _fmt(mean_b),
            "mean_diff_A_minus_B": _fmt(mean_diff),
            "sd_diff": _fmt(sd_diff),
            "t_statistic": _fmt(t_stat),
            "df": df if isinstance(df, int) else "",
            "p_value": _fmt(p_val, 6),
            "alpha": 0.05,
            "significant": "yes" if (n >= 2 and p_val == p_val and p_val < 0.05) else "no",
            "direction": direction,
        })
    return stats_rows


# ──────────────────────────── per-dimension driver ──────────────────────────── #


async def run_dimension(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    dim: str,
    prompts_dir: Path,
    debates_dir: Path,
    output_dir: Path,
    files: list[str],
    dry_run: bool = False,
) -> None:
    prompt_path = prompts_dir / f"{dim}_5-point-in-context.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"missing prompt: {prompt_path}")
    prompt_template = prompt_path.read_text(encoding="utf-8")

    dim_dir = output_dir / dim
    raw_root = dim_dir / "raw_responses"

    # Build per-(file, run) tasks.
    tasks: list[asyncio.Task] = []
    plan: list[tuple[str, int, Path]] = []
    cache_hits = cache_misses = 0
    for fname in files:
        debate_text = (debates_dir / fname).read_text(encoding="utf-8")
        full_prompt = prompt_template.replace("{{TEXT}}", debate_text)
        for k in range(1, RUNS_PER_FILE + 1):
            cache_path = raw_root / fname / f"run_{k}.txt"
            if cache_path.exists():
                cache_hits += 1
            else:
                cache_misses += 1
            if not dry_run:
                tasks.append(
                    asyncio.create_task(
                        get_one_run(
                            client,
                            sem,
                            full_prompt,
                            cache_path,
                            label=f"{dim}/{fname}/run_{k}",
                        )
                    )
                )
                plan.append((fname, k, cache_path))

    print(
        f"[{dim}] cache hits: {cache_hits}, misses: {cache_misses} "
        f"(total {cache_hits + cache_misses})",
        file=sys.stderr,
    )
    if dry_run:
        return

    results = await asyncio.gather(*tasks)

    # Aggregate.
    per_turn: dict[tuple[str, str, str], list] = defaultdict(
        lambda: [None] * RUNS_PER_FILE
    )
    for (fname, k, _cache), (_text, ratings) in zip(plan, results):
        for (deb, turn), rating in ratings.items():
            if (deb, turn) in EXPECTED_TURNS:
                per_turn[(fname, deb, turn)][k - 1] = rating

    long_rows = build_long_rows(files, per_turn)
    pair_rows = build_pair_rows(files, long_rows)
    stats_rows = build_stats_rows(files, pair_rows)

    _csv_write(
        dim_dir / f"turn_{dim}.csv",
        [
            "filename", "debater", "turn",
            "result_1", "result_2", "result_3", "result_4", "result_5",
            "median_result",
        ],
        long_rows,
    )
    _csv_write(
        dim_dir / f"turn_{dim}_pairs.csv",
        [
            "filename", "turn",
            "debater_a_median", "debater_b_median", "diff",
        ],
        pair_rows,
    )
    _csv_write(
        dim_dir / f"turn_{dim}_stats.csv",
        [
            "filename", "n_pairs",
            "mean_A", "mean_B", "mean_diff_A_minus_B", "sd_diff",
            "t_statistic", "df", "p_value", "alpha",
            "significant", "direction",
        ],
        stats_rows,
    )

    missing_cells = sum(
        1 for r in long_rows
        for k in range(1, RUNS_PER_FILE + 1)
        if r[f"result_{k}"] == ""
    )
    if missing_cells:
        print(
            f"[{dim}] WARNING: {missing_cells} blank rating cells across "
            f"{len(long_rows)} turns",
            file=sys.stderr,
        )
    print(f"[{dim}] wrote CSVs to {dim_dir}", file=sys.stderr)


# ──────────────────────────── entry point ──────────────────────────── #

# The section-level summary report quoted in the paper is produced by the
# companion ``summarize_dimension_diffs.py`` script, which reads the
# per-dimension CSVs this script writes.


async def amain(
    dims: list[str],
    prompts_dir: Path,
    debates_dir: Path,
    output_dir: Path,
    dry_run: bool,
) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY") and not dry_run:
        print(
            "ERROR: ANTHROPIC_API_KEY environment variable is not set.",
            file=sys.stderr,
        )
        return 2

    if not prompts_dir.is_dir():
        print(f"ERROR: prompts dir does not exist: {prompts_dir}", file=sys.stderr)
        return 2
    if not debates_dir.is_dir():
        print(f"ERROR: debates dir does not exist: {debates_dir}", file=sys.stderr)
        return 2

    files = sorted(p.name for p in debates_dir.glob(DEBATE_GLOB))
    if not files:
        print(
            f"ERROR: no {DEBATE_GLOB} files found in {debates_dir}",
            file=sys.stderr,
        )
        return 2
    print(
        f"Found {len(files)} debate files matching {DEBATE_GLOB!r} in "
        f"{debates_dir}",
        file=sys.stderr,
    )

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)
    try:
        for dim in dims:
            print(f"=== {dim} (reading from {debates_dir}) ===", file=sys.stderr)
            await run_dimension(
                client, sem, dim, prompts_dir, debates_dir, output_dir, files,
                dry_run=dry_run,
            )
    finally:
        await client.close()

    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dims",
        default=",".join(DIMS),
        help="comma-separated subset of dimensions to run (default: all 11)",
    )
    p.add_argument(
        "--debates-dir",
        type=Path,
        default=DEFAULT_DEBATES_DIR,
        help=(
            "directory containing the ``branch_*.txt`` debate transcripts "
            "to rate (default: %(default)s)."
        ),
    )
    p.add_argument(
        "--prompts-dir",
        type=Path,
        default=DEFAULT_PROMPTS_DIR,
        help=(
            "directory containing the ``<dim>_5-point-in-context.txt`` "
            "rubric prompts (default: %(default)s)."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "directory to write per-dimension CSVs and the raw_responses "
            "cache to (default: %(default)s)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report cache hits/misses without calling the API",
    )
    args = p.parse_args()

    dims = [d.strip() for d in args.dims.split(",") if d.strip()]
    unknown = [d for d in dims if d not in DIMS]
    if unknown:
        print(f"ERROR: unknown dimensions: {unknown}", file=sys.stderr)
        return 2

    return asyncio.run(
        amain(
            dims,
            args.prompts_dir.resolve(),
            args.debates_dir.resolve(),
            args.output_dir.resolve(),
            args.dry_run,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
