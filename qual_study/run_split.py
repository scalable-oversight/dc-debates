"""Regenerate the per-debate splits under data/cleaned/utterance_ids/.

Reads:
    data/cleaned/qual-1-all-debates-for-grading_noids.xlsx
Writes (one xlsx per debate_name):
    data/cleaned/utterance_ids/{debate_name}.xlsx

This is the only part of stage 1 that a public-release user can re-execute,
since the upstream raw GuidedTrack CSVs and the PII-bearing per-debate
intermediates have been withheld. See src/pipeline_record/README.md for the
upstream stages, which are preserved as a process record only.

Usage:
    python run_split.py
"""

from src.split_by_debate import split_by_debate


def main() -> None:
    split_by_debate()


if __name__ == "__main__":
    main()
