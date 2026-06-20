# `data/cleaned`

This directory's canonical content is `quant-1-pilot-no-ids.xlsx`, the de-identified cleaned dataset that is the input to stage 3 of the pipeline and any downstream analysis.

Two upstream artefacts were withheld from the public release:

- `quant-1-pilot.xlsx` — the with-IDs cleaned XLSX produced by `process_quant_pilot.py`. Retained `PROLIFIC_PID` and self-reported demographic columns.
- `warnings.txt` — runtime warnings emitted by `process_quant_pilot.py`. Individual warning lines quote `PROLIFIC_PID` values (e.g. "Duplicate PROLIFIC_PID ..."), so the file as a whole could not be released as-is.

See `../../src/pipeline_record/README.md` for the upstream scripts (preserved as a process record only).
