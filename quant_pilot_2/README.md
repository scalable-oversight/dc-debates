# quant_pilot_2 (public release)

This is the public-release version of the Challenge Fund Quantitative Study Pilot 2 grading pipeline, the canonical de-identified dataset (`data/cleaned/quant-1-pilot-2-no-ids.xlsx`), the descriptive-analysis code under `src/analyses/descriptives/`, and the descriptive-analysis outputs under `data/analyses/descriptives/`. Several upstream artefacts have been withheld for PII reasons; the next section enumerates them.

## What's been withheld and why

To delink participants from the Prolific accounts they used to take part, the following were removed from the public distribution:

- `data/raw/quant-1-pilot-2.csv` — raw GuidedTrack export (held `PROLIFIC_PID` and self-reported demographics).
- `data/cleaned/quant-1-pilot-2.xlsx` — with-IDs cleaned dataset (held the same identity and demographic columns).
- `data/cleaned/warnings.txt` — runtime warnings emitted by stage 1 (individual warning lines quote `PROLIFIC_PID` values).

The two stage-1+2 scripts that consumed those inputs (`process_quant_pilot.py`, `make_no_ids_xlsx.py`) are preserved under `src/pipeline_record/` as a record of how the de-identified dataset was produced. They are not runnable from this distribution and will exit immediately with an explanatory `FileNotFoundError` if invoked. `data/cleaned/quant-1-pilot-2-no-ids.xlsx` is the canonical entry point for downstream analyses; anything that previously read `quant-1-pilot-2.xlsx` should read the `-no-ids` version instead.

## Layout

```
quant_pilot_2/
├── Makefile
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                              # placeholder; contents withheld for PII reasons
│   ├── cleaned/
│   │   ├── README.md                     # notes which upstream files were withheld
│   │   └── quant-1-pilot-2-no-ids.xlsx   # canonical dataset
│   └── analyses/
│       └── descriptives/                 # stage-3 outputs (regenerable)
├── lookups/                              # Hand-authored mapping tables
│   ├── assignment-condition-turn-mapping.csv
│   ├── debate-to-name-and-sourcetype-mapping.csv
│   └── debate-to-name-and-question-mapping.csv
├── src/
│   ├── pipeline_record/                  # NOT runnable; preserved as a process record
│   │   ├── README.md
│   │   ├── process_quant_pilot.py
│   │   └── make_no_ids_xlsx.py
│   └── analyses/
│       └── descriptives/                 # 16 descriptive-analysis scripts (stage 3)
├── docs/
│   └── PIPELINE.md                       # data-flow walkthrough (preserved as process record)
└── experimental_materials/               # Survey instrument + stimuli
```

## Reproducing

### What you can re-run

```bash
pip install -r requirements.txt
make descriptives        # equivalently, `make all`
```

This runs the 16 stage-3 scripts under `src/analyses/descriptives/` against the shipped `quant-1-pilot-2-no-ids.xlsx` and writes their outputs to `data/analyses/descriptives/`.

To run a single descriptives script:

```bash
python src/analyses/descriptives/credence_change.py \
    --input data/cleaned/quant-1-pilot-2-no-ids.xlsx \
    --mapping lookups/debate-to-name-and-question-mapping.csv \
    --out-dir data/analyses/descriptives/
```

`make clean` removes only the regenerable analysis outputs under `data/analyses/descriptives/`. The shipped `quant-1-pilot-2-no-ids.xlsx` is not regenerable from this distribution and is intentionally preserved across `make clean`.

### What's preserved as a process record only

The stage-1+2 pipeline (`src/pipeline_record/process_quant_pilot.py` and `make_no_ids_xlsx.py`) cannot be run against this public distribution because the raw GuidedTrack CSV and the with-IDs intermediates have been withheld. The scripts and `docs/PIPELINE.md` are kept on disk so the data-processing steps used to produce the `-no-ids` xlsx remain documented and auditable. See `src/pipeline_record/README.md`.

## Anonymization invariant

Every analysis script under `src/analyses/` MUST read `data/cleaned/quant-1-pilot-2-no-ids.xlsx`, never `quant-1-pilot-2.xlsx`. The Makefile encodes this: the `descriptives` target depends on `$(NOIDS_XLSX)`, not on any with-IDs file. Any new analysis script added under `src/analyses/` should follow this rule.

## What each folder is for

- **`data/raw/`** — would hold the immutable input; contents withheld in this release.
- **`data/cleaned/`** — `quant-1-pilot-2-no-ids.xlsx` is canonical; upstream variants withheld.
- **`data/analyses/`** — derived analysis outputs; safe to delete and regenerate.
- **`lookups/`** — parameters of the analysis (which assignment hash maps to which debate condition, etc.). Hand-authored, not derived from raw data.
- **`src/analyses/`** — every script takes input/output paths as CLI args; no hardcoded paths.
- **`src/pipeline_record/`** — stage-1+2 scripts preserved as a process record only.
- **`experimental_materials/`** — the GuidedTrack survey definition and stimuli used to *produce* the raw data. Not part of the runnable pipeline.

See `docs/PIPELINE.md` for the end-to-end data flow.
