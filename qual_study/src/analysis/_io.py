"""Shared path constants and IO helpers for the analysis stage.

Every analysis script imports `INPUT_XLSX`, `ANALYSIS_DIR`, and optionally
`BY_DEBATE_DIR`, so the canonical input/output locations are defined once.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_XLSX = PROJECT_ROOT / "data" / "cleaned" / "qual-1-all-debates-for-grading_noids.xlsx"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "cleaned" / "analysis"
BY_DEBATE_DIR = ANALYSIS_DIR / "by_debate"


def load_combined(path: Path | None = None) -> pd.DataFrame:
    """Read the combined noids xlsx into a DataFrame."""
    return pd.read_excel(path or INPUT_XLSX, engine="openpyxl")


def ensure_analysis_dir(by_debate: bool = False) -> None:
    """Create the analysis output dir (and optionally the by_debate subdir)."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if by_debate:
        BY_DEBATE_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(debate_name: str) -> str:
    """Map a debate_name to a filesystem-safe stem (matches the original scripts)."""
    return debate_name.replace(" ", "_").replace("/", "_")
