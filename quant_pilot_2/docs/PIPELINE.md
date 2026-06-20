# Pipeline

> **Note for public-release readers.** This document is preserved as a process record. Stages 1 and 2 (`process_quant_pilot.py`, `make_no_ids_xlsx.py`) cannot be re-executed from the public distribution because their inputs were withheld for PII reasons; see `src/pipeline_record/README.md`. The pipeline's de-identified output (`data/cleaned/quant-1-pilot-2-no-ids.xlsx`) is shipped, and stage 3 (descriptives) can be re-executed against it via `make descriptives`.

## Data flow

```
data/raw/quant-1-pilot-2.csv ─┐
                              ├─►  process_quant_pilot.py  ─►  data/cleaned/quant-1-pilot-2.xlsx
lookups/*.csv                ─┘                            └─►  data/cleaned/warnings.txt

data/cleaned/quant-1-pilot-2.xlsx  ─►  make_no_ids_xlsx.py  ─►  data/cleaned/quant-1-pilot-2-no-ids.xlsx

data/cleaned/quant-1-pilot-2-no-ids.xlsx ─┐
                                          ├─►  src/analyses/descriptives/*.py  ─►  data/analyses/descriptives/*
lookups/debate-to-name-and-question-mapping.csv ─┘
```

## Anonymization invariant

Every analysis script under `src/` MUST read `quant-1-pilot-2-no-ids.xlsx`,
never `quant-1-pilot-2.xlsx`. The only exemption is `make_no_ids_xlsx.py`,
whose job is to *derive* the no-ids file from the main one. The Makefile
encodes this rule: the descriptives target (and any future analysis target)
depends on `$(NOIDS_XLSX)`, not `$(MAIN_XLSX)`.

## Stage 1 — `process_quant_pilot.py`

Reads the raw GuidedTrack CSV plus two lookup tables and produces the main
cleaned XLSX. Key transformations:

1. **Column trim.** Drops the duplicate "question-text" columns from the
   GuidedTrack export, keeping ~38 core fields (run metadata, consent,
   demographics, encoded answers, comprehension checks). Also reads the
   `assignments_to_sourcetypes` column into a private helper column for the
   sourcetype derivation in step 7.
2. **Answer decode.** Each `answer_N` field arrives wrapped in `[ROUNDN]…[ROUNDN]`
   markers with a custom hex-like encoding. The script decodes the payload
   and handles double-encoded and embedded-encoded cases.
3. **Timing extraction.** Parses `START TIME` / `END TIME` from the decoded
   answer text to compute per-section minutes and `total_minutes_evaluating_debates`.
4. **Mapping joins:**
   - `lookups/assignment-condition-turn-mapping.csv` — assignment hash →
     condition (e.g., `01-A`) plus four debate URLs.
   - `lookups/debate-to-name-and-sourcetype-mapping.csv` — debate index →
     debate name. (The `sourcetype` column in this lookup is unused; the
     per-row sourcetype comes from the raw CSV — see step 7.)
5. **Validation.** Flags invalid/duplicate Prolific PIDs, missing `ROUNDn`
   markers, too-short evaluation time, and unknown assignment hashes. All
   warnings written to `data/cleaned/warnings.txt`.
6. **Row filter.** Keep only rows with a valid PID, `ROUND1` in `answer_1`,
   `ROUND4` in `answer_4`, and ≥9 min total evaluation time.
7. **Sourcetype derivation** (pilot 2 specific). The raw CSV column
   `assignments_to_sourcetypes` carries a per-run JSON dict mapping
   assignment hash → sourcetype string (e.g., `quote`). For each row, look up
   that row's `assignment` in its own dict to produce the `sourcetype`
   column. The helper column is dropped afterwards.
8. **Derived columns:**
   - `condition`, `section_{1..4}_url`, `debate_id`, `correct_debater`,
     `debate_name`.
   - `section_{1..4}_credence_in_correct_answer` — regex-extracted from
     decoded answers (e.g. `Debater A: 72%` → `0.72`; `<0.5` → `0.0025`;
     `>99.5` → `0.9975`).
   - `logit_section_{1..4}_credence_in_correct_answer` — log-odds with
     clamping to `[0.01, 0.99]`.
   - `participant_is_correct`, `participant_leans_toward` — derived from the
     section-4 credence vs. `correct_debater`.
   - `credence_change`, `credence_logits_change` — section-1 vs. section-4
     deltas.
   - `red_flags` — empty column reserved for manual review.
9. **Column order.** `answer_1..4` are moved to just after `section_4_url`;
   `sourcetype` is moved to just after `assignment`.
10. **XLSX write.** Bold Arial headers, wrapped cells, clickable hyperlinks
    in the four `section_N_url` columns.

## Stage 2 — `make_no_ids_xlsx.py`

Reads the cleaned XLSX and drops `PROLIFIC_PID` plus all self-reported
demographic columns (degree, education, physics/CS background, gender, age,
country, english level, etc.). Preserves the URL hyperlinks. Output is safe
to share without identifying participants, and is the canonical input for
all downstream analyses in this package.

## Stage 3 — `src/analyses/descriptives/`

Sixteen scripts each producing one or more files under
`data/analyses/descriptives/`. Every script takes the same CLI surface:

```
--input    path to data/cleaned/quant-1-pilot-2-no-ids.xlsx
--mapping  path to lookups/debate-to-name-and-question-mapping.csv
--out-dir  path to data/analyses/descriptives/
```

This uniformity is what lets the Makefile loop over all of them. Four of
the scripts do not in fact use both args:

- `minutes.py`, `section_credences.py`, `section_minutes.py` accept-and-ignore
  `--mapping` (they only read the cleaned XLSX).
- `generate_debate_color_legend.py` accept-and-ignores `--input` (it only
  reads the mapping CSV — it produces a legend keyed on debate names, no
  participant data needed).

The other twelve use both args.

| Script | Outputs |
|---|---|
| `credence_change.py` | `credence_change_descriptives.txt`, `credence_change_histogram.png`, `credence_change_by_debate_swarmplot.png` |
| `credence_logits_change.py` | `credence_logits_change_descriptives.txt`, `credence_logits_change_histogram.png`, `credence_logits_change_by_debate_swarmplot.png` |
| `credence_s1_vs_s4.py` | `credence_s1_vs_s4_by_debate.png` |
| `generate_debate_color_legend.py` | `debate_color_legend.png` |
| `minutes.py` | `minutes_descriptives.txt`, `minutes_histograms.png`, `minutes_swarmplots.png` |
| `question_pair_credence_extremes.py` | `question_pair_credence_extremes.txt` |
| `question_pair_logit_credence_extremes.py` | `question_pair_logit_credence_extremes.txt` |
| `section1_credence.py` | `section1_credence_descriptives.txt`, `section1_credence_histogram.png`, `section1_credence_by_debate_swarmplot.png` |
| `section1_logit.py` | `section1_logit_descriptives.txt`, `section1_logit_histogram.png`, `section1_logit_by_debate_swarmplot.png` |
| `section1_vs_section4_credence_by_debate.py` | `section1_vs_section4_credence_by_debate.txt` |
| `section1_vs_section4_credence_by_question.py` | `section1_vs_section4_credence_by_question.txt` |
| `section1_vs_section4_credence_logits_by_question.py` | `section1_vs_section4_credence_logits_by_question.txt` |
| `section4_credence.py` | `section4_credence_descriptives.txt`, `section4_credence_histogram.png`, `section4_credence_by_debate_swarmplot.png` |
| `section4_logit.py` | `section4_logit_descriptives.txt`, `section4_logit_histogram.png`, `section4_logit_by_debate_swarmplot.png` |
| `section_credences.py` | `section_credences_descriptives.txt`, `section_credences_swarmplot.png` |
| `section_minutes.py` | `section_minutes_descriptives.txt`, `section_minutes_swarmplot.png` |

## Lookup tables

| File | Schema | Used for |
|---|---|---|
| `assignment-condition-turn-mapping.csv` | `Hash \| Condition \| turn1 \| turn2 \| turn3 \| turn4` | Map assignment hash → condition + per-section debate URL |
| `debate-to-name-and-sourcetype-mapping.csv` | `index \| json \| sourcetype` | Map debate index (1–24) → debate filename (the `sourcetype` column is unused by this pipeline) |
| `debate-to-name-and-question-mapping.csv` | extended with `debate_question` | Used by 13 of the 16 descriptives scripts (3 ignore it) |
