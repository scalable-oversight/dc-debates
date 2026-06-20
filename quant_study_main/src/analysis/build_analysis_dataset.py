"""Build the model-ready dataset from the no-IDs cleaned XLSX.

One row per analysed participant, with the columns required by the
hierarchical model in CF_analysis_plan.docx:

    participant_id, transcript, question, domain, order, y_primary, y_secondary, y_sign

Writes ``data/cleaned/quant-1-official-study-analysis-ready.csv`` and prints a
summary.

This is the cf_package port of the original
``quant_study_1/data/build_analysis_dataset.py``. Differences:

- Input is ``data/cleaned/quant-1-official-study-noids.xlsx`` (no
  PROLIFIC_PIDs, no self-reported demographics).
- ``participant_id`` is synthesised as ``p-NNNN`` from the row's position in
  the xlsx (1-based, zero-padded to four digits). Stable across re-runs.
- All variant filters (``--ai-taskers``, ``--non-ai-taskers``,
  ``--no-pangram``, ``--masters``, ``--teams-of-3``) were removed: the data
  needed for the PID-based filters (PROLIFIC_PID, Pangram results) isn't
  available in the no-IDs xlsx, and there's no need for the others without
  them.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "cleaned"
LOOKUPS_DIR = BASE_DIR / "lookups"

XLSX = DATA_DIR / "quant-1-official-study-noids.xlsx"
QMAP = LOOKUPS_DIR / "debate-to-name-and-question-mapping.csv"
OUT = DATA_DIR / "quant-1-official-study-analysis-ready.csv"

# Question -> domain. Spellings must match debate_question in QMAP exactly
# (note: the mapping file uses "Graviational Lens Modelling").
QUESTION_TO_DOMAIN = {
    # Astrophysics (4)
    "Early Universe Galaxies": "astrophysics",
    "Graviational Lens Modelling": "astrophysics",
    "Large Linear Structure": "astrophysics",
    "Little Red Dots": "astrophysics",
    # Coding (4) - reward hacking evaluations
    "Rust": "coding",
    "Database deletion": "coding",
    "ORM library": "coding",
    "Prefix-sum": "coding",
    # Brainteasers (4) - reference level in the model
    "Bergs": "brainteasers",
    "Probability puzzle": "brainteasers",
    "Switch": "brainteasers",
    "Internal Temperature of Stars": "brainteasers",
}


def main() -> int:
    if not XLSX.exists():
        sys.exit(
            f"ERROR: input xlsx not found: {XLSX}\n"
            f"Build it with: make all  (from {BASE_DIR})"
        )

    df = pd.read_excel(XLSX, engine="openpyxl")
    df = df.reset_index(drop=True)
    df["participant_id"] = [f"p-{i + 1:04d}" for i in range(len(df))]

    qmap = pd.read_csv(QMAP, encoding="utf-8")
    df["debate_id"] = df["debate_id"].astype(int)
    qmap["debate_id"] = qmap["debate_id"].astype(int)
    df = df.merge(qmap[["debate_id", "debate_question"]], on="debate_id", how="left")

    unmapped_q = sorted(set(df["debate_question"]) - set(QUESTION_TO_DOMAIN))
    if unmapped_q:
        sys.exit(f"ERROR: questions with no domain mapping: {unmapped_q}")
    df["domain"] = df["debate_question"].map(QUESTION_TO_DOMAIN)

    bad_order = set(df["correct_debater"]) - {"A", "B"}
    if bad_order:
        sys.exit(f"ERROR: unexpected correct_debater values: {bad_order}")
    df["order"] = (df["correct_debater"] == "B").astype(int)

    out = pd.DataFrame({
        "participant_id": df["participant_id"],
        "transcript": df["debate_id"],
        "question": df["debate_question"],
        "domain": df["domain"],
        "order": df["order"],
        "y_primary": df["logit_section_4_credence_in_correct_answer"],
        "y_secondary": df["credence_logits_change"],
    })

    out["y_sign"] = pd.Series(
        np.where(out["y_secondary"] > 0, 1.0,
                 np.where(out["y_secondary"] < 0, 0.0, np.nan)),
        index=out.index,
    )
    n_ties = int(out["y_sign"].isna().sum())
    print(
        f"y_sign: {(out['y_sign'] == 1).sum()} positive, "
        f"{(out['y_sign'] == 0).sum()} negative, "
        f"{n_ties} ties (NaN, dropped by sign model)"
    )

    missing = out[["y_primary", "y_secondary"]].isna().sum()
    if missing.any():
        sys.exit(f"ERROR: missing outcome values:\n{missing}")

    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Wrote {len(out)} rows to {OUT.name}")

    print("\nPer domain:")
    print(out["domain"].value_counts().rename_axis("domain").to_string())

    print("\nPer question:")
    print(out["question"].value_counts().rename_axis("question").sort_index().to_string())

    print("\nPer transcript x order cell:")
    cells = out.groupby(["transcript", "order"]).size().unstack(fill_value=0)
    cells.columns = ["A-honest (order=0)", "B-honest (order=1)"]
    cells["total"] = cells.sum(axis=1)
    print(cells.to_string())

    print(f"\nTotal n = {len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
