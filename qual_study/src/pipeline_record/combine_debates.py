"""Combine per-debate grading xlsx files (process record; not runnable in the public release).

Normalises per-debate column names (debate_{n}_name -> debate_name etc.), extracts
per-section credences on the correct debater from answer_1..answer_3, adds integer
and logit forms of all four section credences, reorders, drops PII columns
(PROLIFIC_PID, STUDY_ID, pid_to_use), and writes one xlsx.

The per-debate xlsx files this script reads were withheld from the public
release because they retain PROLIFIC_PID. See src/pipeline_record/README.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PER_DEBATE_DIR = PROJECT_ROOT / "data" / "cleaned" / "per_debate"
DEFAULT_OUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "qual-1-all-debates-for-grading_noids.xlsx"

PII_COLS = ("PROLIFIC_PID", "STUDY_ID", "pid_to_use")


def extract_credence(answer_text, debater_letter):
    if not isinstance(answer_text, str) or not isinstance(debater_letter, str):
        return None
    pattern = rf"Debater\s+{re.escape(debater_letter)}:\s*(<0\.5|>99\.5|\d{{1,2}})%"
    match = re.search(pattern, answer_text)
    return f"{match.group(1)}%" if match else None


def credence_to_int(val):
    if pd.isna(val) or val is None or val == "":
        return np.nan
    s = str(val).strip()
    if s == ">99.5%":
        return 100
    if s == "<0.5%":
        return 0
    return int(s.replace("%", ""))


def credence_to_logit(integer_val):
    if pd.isna(integer_val):
        return np.nan
    p = integer_val / 100.0
    p = min(max(p, 0.01), 0.99)
    return np.log(p / (1.0 - p))


def combine_debates(
    per_debate_dir: Path = DEFAULT_PER_DEBATE_DIR,
    out_file: Path = DEFAULT_OUT_FILE,
) -> Path:
    per_debate_dir = Path(per_debate_dir)
    out_file = Path(out_file)

    dfs = []
    for i in range(1, 5):
        f = per_debate_dir / f"qual-1-debate-{i}-for-grading.xlsx"
        if not f.exists():
            raise FileNotFoundError(
                f"{f} not found. The per-debate intermediates were withheld "
                "from the public release because they retain PROLIFIC_PID and "
                "other identity columns; see src/pipeline_record/README.md. "
                "The de-identified equivalent of this pipeline's output is "
                "data/cleaned/qual-1-all-debates-for-grading_noids.xlsx."
            )
        df = pd.read_excel(f, engine="openpyxl")
        df = df.rename(columns={
            f"debate_{i}_name": "debate_name",
            f"condition_{i}": "order",
            f"debate_{i}_turn_4": "debate_turn_4_link",
        })
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.rename(columns={
        "final_credence_in_correct_answer": "section_4_credence_in_correct_answer",
    })

    for section in range(1, 4):
        combined[f"section_{section}_credence_in_correct_answer"] = combined.apply(
            lambda row, t=section: extract_credence(row[f"answer_{t}"], row["correct_debater"]),
            axis=1,
        )

    for section in range(1, 5):
        base = f"section_{section}_credence_in_correct_answer"
        combined[f"{base}_as_integer"] = combined[base].map(credence_to_int)
        combined[f"logit_{base}"] = combined[f"{base}_as_integer"].map(credence_to_logit)

    credence_cols = []
    for section in range(1, 5):
        base = f"section_{section}_credence_in_correct_answer"
        credence_cols.extend([base, f"{base}_as_integer", f"logit_{base}"])

    cols = list(combined.columns)
    for c in credence_cols:
        cols.remove(c)
    insert_idx = cols.index("other_comments") + 1
    for i, c in enumerate(credence_cols):
        cols.insert(insert_idx + i, c)
    combined = combined[cols]

    combined = combined.drop(columns=[c for c in PII_COLS if c in combined.columns])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_excel(out_file, index=False, engine="openpyxl")
    print(f"Wrote {len(combined)} rows to {out_file}")
    return out_file


if __name__ == "__main__":
    combine_debates()
