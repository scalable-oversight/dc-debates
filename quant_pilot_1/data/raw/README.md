# `data/raw` — withheld from the public release

This directory would contain the raw GuidedTrack export `quant-1-pilot.csv` that stage 1 of the pipeline consumes. It was withheld from the public distribution because it includes `PROLIFIC_PID` and self-reported demographic columns — identifiers that would let a third party re-link participants to the Prolific accounts they used to take part.

The cleaned, de-identified equivalent that stage 3 of the pipeline reads from is `../cleaned/quant-1-pilot-no-ids.xlsx`. The stage-1+2 scripts that produced it are preserved (but not runnable) under `../../src/pipeline_record/`.
