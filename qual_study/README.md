# Qual Study 1 Pipeline and Data (public release)

This is the public-release version of the Qual Study 1 grading pipeline, the canonical de-identified dataset (`data/cleaned/qual-1-all-debates-for-grading_noids.xlsx`), the analysis code under `src/analysis/`, and the analysis outputs under `data/cleaned/analysis/`. Several upstream artefacts have been withheld for PII reasons; the next section enumerates them.

## What's been withheld and why

To delink participants from the Prolific accounts they used to take part, the following were removed from the public distribution:

- `data/raw/qual-1-debate-{1..4}.csv` — raw GuidedTrack exports (held `PROLIFIC_PID`, `STUDY_ID`, `pid_to_use`).
- `data/cleaned/per_debate/qual-1-debate-{n}-for-grading.xlsx` — per-debate intermediates (held the same PII columns).
- `data/cleaned/qual-1-all-debates-for-grading.xlsx` — earlier with-IDs combined dataset (superseded by the `_noids` version below).
- The `PROLIFIC_PID` column of `data/lookups/study_assignment_by_participant_extended.csv`.
- A single literal `PROLIFIC_PID` value hard-coded inside a manual override in `src/pipeline_record/extract_per_debate.py`, redacted to `[REDACTED]`.

The two stage-1 scripts that consumed those inputs (`extract_per_debate.py`, `combine_debates.py`) are preserved under `src/pipeline_record/` as a record of how the de-identified dataset was produced. They are not runnable from this distribution and will exit immediately with an explanatory `FileNotFoundError` if invoked. `data/cleaned/qual-1-all-debates-for-grading_noids.xlsx` is the canonical entry point for downstream analyses; anything that previously read the with-IDs `qual-1-all-debates-for-grading.xlsx` should read the `_noids` version instead.

## Layout

```
qual_study/
├── README.md
├── requirements.txt
├── run_split.py                 # stage 1.5: regenerate per-debate splits from the noids xlsx
├── run_analysis.py              # stage 2: stats + plots from the noids xlsx
├── src/
│   ├── split_by_debate.py       # _noids.xlsx -> data/cleaned/utterance_ids/*.xlsx
│   ├── pipeline_record/         # NOT runnable; preserved as a process record
│   │   ├── README.md
│   │   ├── extract_per_debate.py
│   │   └── combine_debates.py
│   └── analysis/                # analysis modules invoked by run_analysis.py
│       ├── _io.py               # shared paths and loader
│       ├── credence_stats.py
│       ├── credence_change_stats.py
│       ├── variance_components.py
│       ├── sample_size_with_uncertainty.py
│       └── plots/
│           ├── boxplots.py
│           ├── histograms.py
│           ├── swarmplots.py
│           └── change_swarmplots.py
├── data/
│   ├── raw/                     # placeholder; contents withheld for PII reasons
│   ├── lookups/                 # study_assignment_by_participant_extended.csv (no PROLIFIC_PID),
│   │                            # debate_ids.xlsx
│   └── cleaned/
│       ├── per_debate/          # placeholder; contents withheld for PII reasons
│       ├── qual-1-all-debates-for-grading_noids.xlsx   # canonical dataset
│       ├── utterance_ids/       # per-debate splits (regenerable via run_split.py)
│       └── analysis/            # stage-2 outputs (txt, csv, png)
│           └── by_debate/       # per-debate plot variants
├── docs/
│   ├── pipeline.md              # data-flow walkthrough (preserved as process record)
│   └── columns.md               # column dictionary for the _noids xlsx
└── experimental_materials/      # GuidedTrack .gt sources + debate stimuli
```

## Reproducing

### What you can re-run

```bash
pip install -r requirements.txt
python run_analysis.py    # stats reports, CSVs, and plots from the noids xlsx
python run_split.py       # regenerate data/cleaned/utterance_ids/ from the noids xlsx
```

`run_analysis.py` writes its outputs to `data/cleaned/analysis/`, with per-debate plot variants in `data/cleaned/analysis/by_debate/`. Individual analysis modules can also be invoked directly, e.g.:

```bash
python -m src.analysis.plots.boxplots                   # regenerate one plot
python -m src.analysis.variance_components --out foo.txt
```

### What's preserved as a process record only

The stage-1 pipeline (`src/pipeline_record/extract_per_debate.py` and `combine_debates.py`) cannot be run against the public distribution because the raw GuidedTrack CSVs and per-debate intermediates have been withheld. The scripts and `docs/pipeline.md` are kept on disk so that the data-processing steps used to produce the `_noids` xlsx remain documented and auditable. See `src/pipeline_record/README.md`.

## Inputs

- `data/lookups/study_assignment_by_participant_extended.csv` — for each `participant_id`, the four debates they were assigned, the condition (Honest first / Dishonest first), and the URL of the debate transcript. The `PROLIFIC_PID` column was dropped from this file for the public release; downstream of the `_noids` xlsx everything keys off `participant_id`, so the dropped column is only needed if you re-run stage 1 (which requires the withheld raw CSVs).
- `data/lookups/debate_ids.xlsx` — per-debate explanatory message appended to `participant_message`.

## Notes

- Downstream analyses that previously read `qual-1-all-debates-for-grading.xlsx` should read `qual-1-all-debates-for-grading_noids.xlsx` instead; the two have the same shape modulo the dropped PII columns.
- Debate stimuli (HTML transcripts) live under `experimental_materials/stimuli/` and are referenced by URL in `debate_turn_4_link`.
