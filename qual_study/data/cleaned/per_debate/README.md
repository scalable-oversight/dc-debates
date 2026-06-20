# `data/cleaned/per_debate` — withheld from the public release

This directory would contain four `qual-1-debate-{n}-for-grading.xlsx` files written by `src/pipeline_record/extract_per_debate.py`. They are stage-1.5 intermediates: each one is the result of joining `data/raw/qual-1-debate-{n}.csv` against the participant assignment lookup and applying the decoding / scoring described in `docs/pipeline.md`. They retain the same PII columns (`PROLIFIC_PID`, `STUDY_ID`, `pid_to_use`) that the raw CSVs had, so they were withheld from this release for the same reasons.

The combined, de-identified equivalent that the rest of this package reads from is `../qual-1-all-debates-for-grading_noids.xlsx`.
