# Pipeline

> **Note for public-release readers.** This document is preserved as a process record. Stages 1 and 2 (`process_quant_study.py`, `make_no_ids_xlsx.py`) cannot be re-executed from the public distribution because their inputs were withheld for PII reasons; see `src/pipeline_record/README.md`. The pipeline's de-identified output (`data/cleaned/quant-1-official-study-noids.xlsx`) is shipped, and the downstream Bayesian analysis pipeline can be re-executed against it via `src/analysis/run_all.sh`.

## Data flow

```
data/raw/quant-1-official-study.csv     ─┐
data/raw/ai_tasker_demographics/*.csv   ─┤
                                         ├─►  process_quant_study.py  ─►  data/cleaned/quant-1-official-study.xlsx
lookups/*.csv                           ─┘                            └─►  data/cleaned/warnings.txt

data/cleaned/quant-1-official-study.xlsx  ─►  make_no_ids_xlsx.py  ─►  data/cleaned/quant-1-official-study-noids.xlsx
```

## Stage 1 — `process_quant_study.py`

Reads the raw GuidedTrack CSV plus two lookup tables and the AI-tasker
demographic directory, then produces the main cleaned XLSX. Key
transformations:

1. **Column trim.** Drops duplicate "question-text" columns; keeps ~38 core
   fields. Reads `assignments_to_sourcetypes` into a private helper column
   for the sourcetype derivation in step 8.
2. **Whitespace strip.** Removes spaces / `\r` / `\n` from the four
   `answer_N` cells before decoding (incoming submissions occasionally have
   whitespace injected into the hex payload).
3. **Answer decode.** Each `answer_N` field arrives wrapped in
   `[ROUNDN]…[ROUNDN]` markers with a custom hex-like encoding (case
   insensitive). The decoder is lenient: it accepts payloads with only the
   opening marker (closing marker may be missing on resubmission), splits
   on internal `[ROUNDN]` markers, decodes each chunk, and handles
   double-encoded and embedded-encoded cases.
4. **Timing extraction.** Parses `START TIME` / `END TIME` from the decoded
   answer text to compute per-section minutes and
   `total_minutes_evaluating_debates`.
5. **Mapping joins:**
   - `lookups/assignment-condition-turn-mapping.csv` — assignment hash →
     condition (e.g., `01-A`) plus four debate URLs.
   - `lookups/debate-to-name-and-sourcetype-mapping.csv` — debate index →
     debate name. The `sourcetype` column in this lookup is unused; per-row
     sourcetype comes from the raw CSV (step 8).
6. **Warnings.** Computed only for rows that pass the `iAgreeNoLLMs` check.
   Flags invalid PIDs, missing `ROUNDn` markers, total time ≤9 min, unknown
   assignment hashes, valid-PID-but-filtered-out rows, and (deferred)
   duplicate PIDs whose all submissions failed. Written to
   `data/cleaned/warnings.txt`.
7. **Enrichment on all rows.** All derived columns (URLs, debate ID/name,
   sourcetype, credences, logits, deltas) are computed on every row —
   including rows that will ultimately be rejected — so the deferred
   duplicate-PID warning can see the full picture.
8. **Sourcetype derivation.** The raw CSV column
   `assignments_to_sourcetypes` carries a per-run JSON dict mapping
   assignment hash → sourcetype string. For each row, look up that row's
   `assignment` in its own dict to produce the `sourcetype` column.
9. **Per-row rejection reasons.** A row is rejected if any of:
   `iAgreeNoLLMs == "No"`, invalid PID, missing `ROUND1`/`ROUND4`, missing
   or `≤9` min total time, or missing section-1 / section-4 credence.
   Rejected rows are split off into a `rejects` frame (kept in memory only,
   *not* written to disk).
10. **Successful-PID dedup.** Drop from the `rejects` frame any submission
    whose PID already appears in `out`. Then emit a deferred duplicate-PID
    warning for any valid PID that appears more than once in
    `out ∪ rejects` but is absent from `out` (i.e. a participant whose
    every submission failed).
11. **AI-tasker / masters flags** (added to `out` only):
    - `ai_tasker` — True iff `PROLIFIC_PID` appears in the union of
      `Participant id` columns across every CSV in
      `data/raw/ai_tasker_demographics/`.
    - `masters_or_doctorate` — True iff `educationCompleted` is "Graduate
      degree (MA, MSc, MPhil, or equivalent)" or "Doctorate degree (PhD,
      DPhil, or equivalent)".
12. **Column order.** `answer_1..4` are moved to just after `section_4_url`;
    `sourcetype` is moved to just after `assignment`.
13. **XLSX write.** Bold Arial headers, wrapped cells, clickable hyperlinks
    in the four `section_N_url` columns. ASCII control characters (other
    than `\t\n\r`) are stripped from cell values — corrupted submissions
    can produce bytes that openpyxl refuses to write.

## Stage 2 — `make_no_ids_xlsx.py`

Reads the cleaned XLSX and drops `PROLIFIC_PID` plus all self-reported
demographic columns (degree, education, physics/CS background, gender, age,
country, English level, etc.). Preserves URL hyperlinks. Output is safe to
share without identifying participants.

## Lookup tables

| File | Schema | Used for |
|---|---|---|
| `assignment-condition-turn-mapping.csv` | `Hash | Condition | turn1 | turn2 | turn3 | turn4` | Map assignment hash → condition + per-section debate URL |
| `debate-to-name-and-sourcetype-mapping.csv` | `index | json | sourcetype` | Map debate index (1–24) → debate filename (the `sourcetype` column is unused by this pipeline) |
| `debate-to-name-and-question-mapping.csv` | extended with `debate_question` | Used by downstream analyses (not by this pipeline) |

## Deviations from the original `quant_study_1/data/` pipeline

These are intentional simplifications that produce the same `quant-1-official-study.xlsx` content:

- **Rejects XLSX not written.** The original wrote `quant-1-rejects.xlsx`
  alongside the main file. No script consumes that file, so this package
  skips the write. The internal `rejects` frame is still computed because
  the deferred duplicate-PID warning needs it.
- **`DUMMY_PID` removed.** The sentinel `cabba6ecabba6ecabba6eca6` and its
  conditional bypasses were removed. No row in any input CSV carries that
  PID, so the simplification is behavior-preserving.
