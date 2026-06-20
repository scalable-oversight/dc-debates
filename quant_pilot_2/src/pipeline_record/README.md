# `pipeline_record` — preserved for transparency, not runnable from this distribution

The two scripts in this directory — `process_quant_pilot.py` and `make_no_ids_xlsx.py` — are stages 1 and 2 of the quant-pilot-2 pipeline that produced `data/cleaned/quant-1-pilot-2-no-ids.xlsx`. They are checked in here as a faithful record of the data-processing steps applied to the source data.

**They cannot be run from this public distribution.** Their inputs were withheld because they retain `PROLIFIC_PID` (Prolific's per-participant identifier) and self-reported demographic columns whose presence would let a third party re-link participants to the Prolific accounts they used to take part:

- `process_quant_pilot.py` reads `data/raw/quant-1-pilot-2.csv` (the raw GuidedTrack export) and writes `data/cleaned/quant-1-pilot-2.xlsx` plus `data/cleaned/warnings.txt`. Both the raw CSV and the with-IDs xlsx were withheld; `warnings.txt` was withheld because individual warning lines quote `PROLIFIC_PID` values.
- `make_no_ids_xlsx.py` reads the with-IDs xlsx and strips `PROLIFIC_PID` plus self-reported demographics to produce `data/cleaned/quant-1-pilot-2-no-ids.xlsx`.

If either script is invoked it will detect the missing input and exit immediately with a `FileNotFoundError` that points back to this README and to the de-identified equivalent of its output.

## How to use the cleaned data instead

The canonical entry point for downstream work is `data/cleaned/quant-1-pilot-2-no-ids.xlsx`. Stage 3 (the 16 scripts under `src/analyses/descriptives/`, driven by `make descriptives`) reads it directly.

## What the scripts do

`docs/PIPELINE.md` contains a step-by-step description of all three stages, kept as a process record alongside this code.
