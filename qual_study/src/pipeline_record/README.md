# `pipeline_record` — preserved for transparency, not runnable from this distribution

The two scripts in this directory — `extract_per_debate.py` and `combine_debates.py` — are the stage-1 pipeline that produced `data/cleaned/qual-1-all-debates-for-grading_noids.xlsx` from the raw GuidedTrack exports. They are checked in here as a faithful record of the data-processing steps that were applied to the source data.

**They cannot be run end-to-end from this public distribution.** The inputs they consume — `data/raw/qual-1-debate-{1..4}.csv` and the per-debate intermediates under `data/cleaned/per_debate/` — were withheld because they retain `PROLIFIC_PID` (Prolific's per-participant identifier) and other identity columns whose presence would let a third party re-link participants to the Prolific accounts they used to take part. The `PROLIFIC_PID` column of `data/lookups/study_assignment_by_participant_extended.csv` was dropped for the same reason, which also removes the join key these scripts rely on. A single literal `PROLIFIC_PID` value that appeared inside a hard-coded manual override has been replaced with `[REDACTED]` in `extract_per_debate.py`; the live override was applied upstream before the `_noids` xlsx was written, so this redaction does not affect the cleaned dataset.

If either script is invoked it will detect the missing input and exit immediately with a `FileNotFoundError` that points back to this README and to the de-identified equivalent of its output.

## How to use the cleaned data instead

The canonical entry point for downstream work is `data/cleaned/qual-1-all-debates-for-grading_noids.xlsx`. The analysis stage (everything under `src/analysis/`, driven by `run_analysis.py`) reads it directly, and `src/split_by_debate.py` (driven by `run_split.py`) regenerates the per-debate splits under `data/cleaned/utterance_ids/` from it.

## What the scripts do

`docs/pipeline.md` contains a step-by-step description of both stages, kept as a process record alongside this code.
