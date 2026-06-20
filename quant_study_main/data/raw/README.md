# `data/raw` — withheld from the public release

This directory would contain the two raw inputs that stage 1 of the pipeline consumes:

- `quant-1-official-study.csv` — the raw GuidedTrack survey export. Retains `PROLIFIC_PID` and self-reported demographic columns.
- `ai_tasker_demographics/` — a directory of Prolific demographic export CSVs. The union of their `Participant id` columns defines the `ai_tasker` flag in the cleaned dataset.

Both were withheld from the public distribution because they include identifiers that would let a third party re-link participants to the Prolific accounts they used to take part.

The cleaned, de-identified equivalent that the analysis pipeline reads from is `../cleaned/quant-1-official-study-noids.xlsx`. The stage-1+2 scripts that produced it are preserved (but not runnable) under `../../src/pipeline_record/`.
