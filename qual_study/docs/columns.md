# Column dictionary — `qual-1-all-debates-for-grading_noids.xlsx`

Each row is one (participant, debate) submission. A participant who completed
all four debates appears in four rows.

PII columns (`PROLIFIC_PID`, `STUDY_ID`, `pid_to_use`) were dropped from the
combined xlsx documented here. They were retained by the per-debate
intermediates upstream of this file, which is why those intermediates were
withheld from the public release; see `../src/pipeline_record/README.md`.

## Provenance / identity

| Column | Description |
|---|---|
| `Run`, `Program Version`, `guidedTrackID` | GuidedTrack run metadata. |
| `Time Started (UTC)`, `Time Finished (UTC)`, `Minutes Spent` | Session-level timing. |
| `Position` | GuidedTrack position. |
| `participant_id`, `group` | From `study_assignment_by_participant_extended.csv`. |
| `simplified-category`, `category` | Topic categorisation of the debate. |
| `debate_name` | Name of this debate (renamed from `debate_{n}_name`). |
| `order` | Condition for this debate: `Honest first` or `Dishonest first` (renamed from `condition_{n}`). |
| `correct_debater` | `A` if `order == Honest first`, else `B`. |
| `debate_turn_4_link` | URL of the full debate transcript (renamed from `debate_{n}_turn_4`). |
| `device` | Device the participant used. |
| `iAgreeNoLLMs`, `consentedToAll` | Consent fields. |

## Per-section timing

| Column | Description |
|---|---|
| `answer_1_minutes` | Minutes between `answer_1`'s embedded START TIME and `answer_2`'s. |
| `answer_2_minutes` | Same for `answer_2 → answer_3`. |
| `answer_3_minutes` | Same for `answer_3 → answer_4`. |
| `answer_4_and_comments_minutes` | Minutes between `answer_4`'s START TIME and `Time Finished (UTC)`. |

## Free-text answers

| Column | Description |
|---|---|
| `answer_1..answer_4` | The participant's free-text answer at the end of debate sections 1–4, hex-decoded. Each begins with a `START TIME` line and includes their per-debater credences as `Debater A: NN%` / `Debater B: NN%`. |
| `other_comments` | Free-form end-of-debate comments. |

## Per-section credence on the correct debater

For each section ∈ {1, 2, 3, 4}:

| Column | Description |
|---|---|
| `section_{n}_credence_in_correct_answer` | String form: `NN%`, `>99.5%`, or `<0.5%` extracted from `answer_{n}` for `correct_debater`. |
| `section_{n}_credence_in_correct_answer_as_integer` | Integer 0–100. `>99.5% → 100`, `<0.5% → 0`. |
| `logit_section_{n}_credence_in_correct_answer` | Logit of `p = integer/100`, clamped to `[0.01, 0.99]`. |

## Derived correctness and feedback

| Column | Description |
|---|---|
| `participant_is_correct` | `Yes` / `-` / `No`, derived from `section_4_credence_in_correct_answer_as_integer`: `>50 → Yes`, `=50 → -`, `<50 → No`. |
| `participant_message` | A short feedback phrase composed from two parts: a credence-bucket opener (e.g. "As you concluded…", "As you suspected…", "…was actually correct") followed by the per-debate sentence from `data/lookups/debate_ids.xlsx`. |

## Grading scaffolding (reserved for analysis)

| Column | Description |
|---|---|
| `red_flags` | Reserved for analysis; shipped blank, except auto-populated with `"PROLIFIC_PID != pid_to_use"` if those two fields disagreed in the upstream per-debate intermediate. |
| `stage_1`, `stage_2`, `stage_3`, `stage_4` | Reserved for analysis; shipped blank. |
