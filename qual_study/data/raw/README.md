# `data/raw` — withheld from the public release

This directory would contain four GuidedTrack export CSVs (`qual-1-debate-{1..4}.csv`), one per debate slot in Qual Study 1. They were withheld from the public distribution because they include `PROLIFIC_PID`, `STUDY_ID`, and `pid_to_use` — identifiers that would let a third party re-link participants to the Prolific accounts they used to take part.

The cleaned, de-identified equivalent that the rest of this package reads from is `../cleaned/qual-1-all-debates-for-grading_noids.xlsx`. The stage-1 scripts that produced it from the raw CSVs are preserved (but not runnable) under `../../src/pipeline_record/`.
