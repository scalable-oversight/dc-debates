# quant_study_main (public release)

This is the public-release version of the Challenge Fund Quantitative Main Study grading and analysis pipeline, the canonical de-identified dataset (`data/cleaned/quant-1-official-study-noids.xlsx`), the Bayesian analysis code under `src/analysis/`, and the analysis outputs under `src/analysis/output/`. Several upstream artefacts have been withheld for PII reasons; the next section enumerates them.

## What's been withheld and why

To delink participants from the Prolific accounts they used to take part, the following were removed from the public distribution:

- `data/raw/quant-1-official-study.csv` — raw GuidedTrack export (held `PROLIFIC_PID` and self-reported demographics).
- `data/raw/ai_tasker_demographics/` — directory of Prolific demographic export CSVs. The union of their `Participant id` columns defines the `ai_tasker` flag in the cleaned dataset.
- `data/cleaned/quant-1-official-study.xlsx` — with-IDs cleaned dataset (held the same identity and demographic columns).
- `data/cleaned/warnings.txt` — runtime warnings emitted by stage 1 (individual warning lines quote `PROLIFIC_PID` values).

The two stage-1+2 scripts that consumed those inputs (`process_quant_study.py`, `make_no_ids_xlsx.py`) are preserved under `src/pipeline_record/` as a record of how the de-identified dataset was produced. They are not runnable from this distribution and will exit immediately with an explanatory `FileNotFoundError` if invoked. `data/cleaned/quant-1-official-study-noids.xlsx` is the canonical entry point for downstream analyses; anything that previously read `quant-1-official-study.xlsx` should read the `-noids` version instead.

## Layout

```
quant_study_main/
├── Makefile                                          # informational stub (see Reproducing)
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                                          # placeholder; contents withheld for PII reasons
│   └── cleaned/
│       ├── README.md                                 # notes which upstream files were withheld
│       ├── quant-1-official-study-noids.xlsx         # canonical dataset
│       ├── quant-1-official-study-analysis-ready.csv # built by src/analysis/build_analysis_dataset.py
│       └── quant-1-official-study-beta-long.csv      # built by src/analysis/build_beta_dataset.py
├── lookups/                                          # Hand-authored mapping tables
│   ├── assignment-condition-turn-mapping.csv
│   ├── debate-to-name-and-sourcetype-mapping.csv
│   └── debate-to-name-and-question-mapping.csv
├── src/
│   ├── pipeline_record/                              # NOT runnable; preserved as a process record
│   │   ├── README.md
│   │   ├── process_quant_study.py
│   │   └── make_no_ids_xlsx.py
│   └── analysis/                                     # Bayesian analysis pipeline (stage 3+)
│       ├── run_all.sh                                # main entrypoint
│       ├── build_analysis_dataset.py
│       ├── build_beta_dataset.py
│       ├── fit_*.py / derive_tiers*.py / plot_*.py / compare_*.py / model_fit_checks*.py / ...
│       └── output/                                   # *.log, *.csv, *.nc, *.png, *.txt
├── docs/
│   └── PIPELINE.md                                   # stage-1+2 walkthrough (preserved as process record)
└── experimental_materials/                           # Survey instrument + stimuli
```

## Reproducing

### What you can re-run

```bash
pip install -r requirements.txt
cd src/analysis && ./run_all.sh
```

`run_all.sh` reads `data/cleaned/quant-1-official-study-noids.xlsx`, builds the analysis-ready and beta-long datasets, fits the hierarchical Bayesian models, derives the difficulty tiers (primary, secondary, secondary-Student-t, secondary-sign, secondary-Beta; 4-tier and 3-tier), and writes per-stage logs, fit objects (`*.nc`), summary CSVs, and plot PNGs to `src/analysis/output/`. The Bayesian fits are long-running; see the comments in `run_all.sh` for the full sequence. The contents of `src/analysis/output/` shipped in this release (including the per-stage `*.log` files) are the artefacts from the canonical run; re-running `run_all.sh` will overwrite them.

Individual stages can also be run directly, e.g.:

```bash
cd src/analysis
python3 build_analysis_dataset.py                       # rebuild the analysis-ready CSV
python3 derive_tiers.py primary                         # rebuild one tier-classification CSV
python3 plot_outcome_hdis.py                            # rebuild one plot
```

`make all` (or just `make`) prints a short message pointing at `run_all.sh`; the upstream stage-1+2 targets it used to encode are gone because their inputs were withheld.

### What's preserved as a process record only

The stage-1+2 pipeline (`src/pipeline_record/process_quant_study.py` and `make_no_ids_xlsx.py`) cannot be run against this public distribution because the raw GuidedTrack CSV, the `ai_tasker_demographics/` directory, and the with-IDs intermediates have been withheld. The scripts and `docs/PIPELINE.md` are kept on disk so the data-processing steps used to produce the `-noids` xlsx remain documented and auditable. See `src/pipeline_record/README.md`.

## Anonymization invariant

Every analysis script under `src/analysis/` MUST read `data/cleaned/quant-1-official-study-noids.xlsx`, never `quant-1-official-study.xlsx`. Any new analysis script added under `src/analysis/` should follow this rule.

## What each folder is for

- **`data/raw/`** — would hold the immutable inputs; contents withheld in this release.
- **`data/cleaned/`** — `quant-1-official-study-noids.xlsx` is canonical; upstream variants withheld. The two `*.csv` files under it are regenerable analysis intermediates.
- **`lookups/`** — parameters of the analysis (assignment hash → debate condition, debate index → debate name). Hand-authored, not derived from raw data.
- **`src/analysis/`** — Bayesian analysis pipeline; reads only from `data/cleaned/`.
- **`src/pipeline_record/`** — stage-1+2 scripts preserved as a process record only.
- **`experimental_materials/`** — the GuidedTrack survey definition and stimuli used to *produce* the raw data. Not part of the runnable pipeline.

See `docs/PIPELINE.md` for the stage-1+2 data flow (preserved as a process record).
