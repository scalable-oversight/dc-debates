"""Build a long-format dataset for the Beta-likelihood secondary model.

One row per (participant, section) for section in {1, 4}, with the raw
credence-in-correct-answer ``p`` in (0, 1) as the outcome instead of the
logit-difference ``y_secondary`` used by the existing pipeline. This dataset
is the input to ``fit_secondary_beta.py``.

This is the cf_package port of the original
``quant_study_1/data/build_beta_dataset.py``. Inputs come from the no-IDs
cleaned xlsx; ``participant_id`` is synthesised the same way it is in
``build_analysis_dataset.py`` (``p-NNNN`` from the xlsx row index, 1-based,
zero-padded to four digits), so the merge on
``(participant_id, transcript)`` lines up.

Motivation (unchanged from the original):
    ``y_secondary = logit(p_4) - logit(p_1)`` is mechanically bounded to
    ``+/-2 * logit(0.99) = +/-9.190`` because the upstream survey reports
    credences as integers in 1..99, and ``to_logit`` in
    ``process_quant_study.py`` clamps to [0.01, 0.99] before the logit.
    Modelling ``p`` directly with a Beta likelihood honors the bounded
    support without invoking heavy tails.

Output: ``data/cleaned/quant-1-official-study-beta-long.csv``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "cleaned"

XLSX = DATA_DIR / "quant-1-official-study-noids.xlsx"
ANALYSIS_CSV = DATA_DIR / "quant-1-official-study-analysis-ready.csv"
OUT = DATA_DIR / "quant-1-official-study-beta-long.csv"


def main() -> int:
    if not ANALYSIS_CSV.exists():
        sys.exit(
            f"ERROR: analysis-ready CSV not found: {ANALYSIS_CSV}\n"
            f"Build it with: python3 build_analysis_dataset.py"
        )
    if not XLSX.exists():
        sys.exit(f"ERROR: input xlsx not found: {XLSX}")

    print(f"Reading analysis-ready CSV : {ANALYSIS_CSV}")
    df_ar = pd.read_csv(ANALYSIS_CSV, encoding="utf-8")
    print(f"  {len(df_ar)} participants")

    print(f"Reading xlsx               : {XLSX}")
    df_x = pd.read_excel(XLSX, engine="openpyxl")
    df_x = df_x.reset_index(drop=True)

    needed = [
        "debate_id",
        "section_1_credence_in_correct_answer",
        "section_4_credence_in_correct_answer",
    ]
    missing = [c for c in needed if c not in df_x.columns]
    if missing:
        sys.exit(f"ERROR: xlsx is missing required columns: {missing}")

    df_x = df_x[needed].copy()
    df_x["participant_id"] = [f"p-{i + 1:04d}" for i in range(len(df_x))]
    df_x["debate_id"] = df_x["debate_id"].astype(int)
    df_x = df_x.rename(columns={"debate_id": "transcript"})

    df_ar = df_ar.copy()
    df_ar["transcript"] = df_ar["transcript"].astype(int)

    merged = df_ar.merge(
        df_x,
        on=["participant_id", "transcript"],
        how="left",
        validate="one_to_one",
    )

    p1_col = "section_1_credence_in_correct_answer"
    p4_col = "section_4_credence_in_correct_answer"
    bad = merged[merged[p1_col].isna() | merged[p4_col].isna()]
    if len(bad):
        sys.exit(
            f"ERROR: {len(bad)} analysis-ready rows have missing "
            f"section_1 or section_4 credence in the xlsx. First few:\n"
            f"{bad[['participant_id','transcript',p1_col,p4_col]].head().to_string()}"
        )

    for col in (p1_col, p4_col):
        oob = merged[(merged[col] <= 0.0) | (merged[col] >= 1.0)]
        if len(oob):
            sys.exit(
                f"ERROR: {len(oob)} {col} values are at or outside (0, 1); "
                f"Beta likelihood requires strict interior. "
                f"min={oob[col].min()} max={oob[col].max()}"
            )

    long_rows: list[dict] = []
    for row in merged.itertuples(index=False):
        base = {
            "participant_id": row.participant_id,
            "transcript": int(row.transcript),
            "question": row.question,
            "domain": row.domain,
            "order": int(row.order),
        }
        long_rows.append({
            **base,
            "section": 1,
            "section_is_4": 0,
            "p": float(getattr(row, p1_col)),
        })
        long_rows.append({
            **base,
            "section": 4,
            "section_is_4": 1,
            "p": float(getattr(row, p4_col)),
        })

    out = pd.DataFrame(long_rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(
        f"\nWrote {len(out)} rows "
        f"({len(merged)} participants x 2 sections) -> {OUT.name}"
    )

    print("\nRows per (section, domain):")
    print(
        out.groupby(["section", "domain"]).size()
        .unstack(fill_value=0).to_string()
    )
    print("\nRaw p summary by section:")
    print(out.groupby("section")["p"].describe().to_string())

    floor_ceil = (
        out.assign(at_floor=(out["p"] <= 0.011), at_ceiling=(out["p"] >= 0.989))
        .groupby("section")[["at_floor", "at_ceiling"]].sum()
    )
    print("\nBoundary pile-ups (p<=0.011 = 'at 1%' floor, p>=0.989 = 'at 99%' ceiling):")
    print(floor_ceil.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
