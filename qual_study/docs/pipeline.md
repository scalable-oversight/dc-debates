# Pipeline

> **Note for public-release readers.** This document is preserved as a process record. Stage 1 (`extract_per_debate.py`, `combine_debates.py`) cannot be re-executed from the public distribution because its raw inputs and per-debate intermediates were withheld for PII reasons; see `src/pipeline_record/README.md`. The pipeline's de-identified output (`data/cleaned/qual-1-all-debates-for-grading_noids.xlsx`) is shipped, and stage 2 (and the stage-1.5 split via `run_split.py`) can be re-executed against it.

```
data/raw/qual-1-debate-{n}.csv   data/lookups/study_assignment_*.csv
                       |                       |
                       +-----------+-----------+
                                   v
                       src/extract_per_debate.py   <-- run once per n in 1..4
                                   v
            data/cleaned/per_debate/qual-1-debate-{n}-for-grading.xlsx
                                   v
                       src/combine_debates.py
                                   v
        data/cleaned/qual-1-all-debates-for-grading_noids.xlsx
```

## Stage 1: `extract_per_debate.py`

For each debate n, reads `qual-1-debate-{n}.csv` and produces a per-debate
"for grading" xlsx. Steps:

1. **Join with assignments.** Merge in the `participant_id`, `group`,
   `simplified-category`, `category`, `debate_{n}_name`, `condition_{n}`,
   and `debate_{n}_turn_4` columns from
   `study_assignment_by_participant_extended.csv`. The `debate_{n}_turn_4`
   field has its local stimuli path rewritten to a public
   `moduloresearch.com` URL.
2. **Derive `correct_debater`** from `condition_{n}`:
   `Honest first → A`, `Dishonest first → B`.
3. **Filter** out rows whose raw (pre-decoded) `answer_4` is ≤ 15 characters
   long — these are participants who effectively didn't submit a final answer.
4. **Decode `answer_1..answer_4`** by interpreting them as a hex string and
   decoding as UTF-8. The Redshift GuidedTrack task stores free-text answers
   in this form.
5. **Extract final credence.** From the decoded `answer_4`, find the line
   matching `Debater {correct_debater}: NN%` (or `>99.5%` / `<0.5%`) and
   store it as `final_credence_in_correct_answer`.
6. **Classify correctness.** Derive `participant_is_correct ∈ {Yes, -, No}`
   from the credence (>50 / =50 / <50).
7. **Build `participant_message`.** Compose a phrase keyed off the credence
   bucket ("As you concluded...", "As you suspected...", "...was actually
   correct"), then append the per-debate sentence from `debate_ids.xlsx`.
8. **Compute per-section minutes.** Each decoded `answer_i` carries a
   `START TIME` timestamp; the per-section duration is the gap to the next
   timestamp (or `Time Finished (UTC)` for section 4).
9. **Add grading scaffolding columns:** empty `red_flags`, `stage_1..stage_4`.
   `red_flags` is populated with `"PROLIFIC_PID != pid_to_use"` when those
   two columns disagree.
10. **Write xlsx** to `data/cleaned/per_debate/qual-1-debate-{n}-for-grading.xlsx`
    with the `debate_{n}_turn_4` cell rendered as a clickable hyperlink.

## Stage 2: `combine_debates.py`

1. **Load** the 4 per-debate xlsx files and rename the debate-specific
   columns to common names so the frames can be stacked:
   - `debate_{n}_name → debate_name`
   - `condition_{n} → order`
   - `debate_{n}_turn_4 → debate_turn_4_link`
2. **Concatenate** the 4 frames row-wise.
3. **Rename** `final_credence_in_correct_answer → section_4_credence_in_correct_answer`.
4. **Extract sections 1–3 credences** by running the same `Debater X: NN%`
   regex over `answer_1`, `answer_2`, `answer_3` against `correct_debater`,
   storing as `section_1..3_credence_in_correct_answer`.
5. **Add integer + logit forms** for each of the four section credences:
   - `_as_integer`: `>99.5% → 100`, `<0.5% → 0`, else integer percent.
   - `logit_*`: logit of `p = integer/100`, clamped to `[0.01, 0.99]`.
6. **Reorder** so all twelve new credence columns sit immediately after
   `other_comments`.
7. **Drop PII columns** (`PROLIFIC_PID`, `STUDY_ID`, `pid_to_use`) from the
   combined frame. Stripping happens only on the combined file at this stage;
   the per-debate intermediates retained these columns and were therefore
   withheld from the public release.
8. **Write** `data/cleaned/qual-1-all-debates-for-grading_noids.xlsx`.
