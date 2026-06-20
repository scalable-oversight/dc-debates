# `data/cleaned`

This directory's canonical content is `quant-1-official-study-noids.xlsx`, the de-identified cleaned dataset that is the input to the Bayesian analysis pipeline under `src/analysis/`.

Two upstream artefacts were withheld from the public release:

- `quant-1-official-study.xlsx` — the with-IDs cleaned XLSX produced by `process_quant_study.py`. Retained `PROLIFIC_PID` and self-reported demographic columns.
- `warnings.txt` — runtime warnings emitted by `process_quant_study.py`. Individual warning lines quote `PROLIFIC_PID` values (e.g. "Duplicate PROLIFIC_PID ..."), so the file as a whole could not be released as-is.

The other two files in this directory are regenerable analysis intermediates produced by stage 3:

- `quant-1-official-study-analysis-ready.csv` — built by `src/analysis/build_analysis_dataset.py` from the noids xlsx. One row per analysed participant with the columns the hierarchical model needs.
- `quant-1-official-study-beta-long.csv` — built by `src/analysis/build_beta_dataset.py` from the noids xlsx and the analysis-ready CSV. Long format (one row per `(participant, section)` for sections 1 and 4) for the Beta-likelihood secondary model.

See `../../src/pipeline_record/README.md` for the upstream stage-1+2 scripts (preserved as a process record only).
