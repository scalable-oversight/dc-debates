# balancing

Diagnostic balancing pipeline for the 24 debate transcripts released with
the Challenge Fund stimulus set. This is the procedure described in
section "Balancing" of the paper.

This package includes the prompts, the final balanced transcripts, the
rerevised rating CSVs, and the two scripts used to produce them.

## Headline result

For the 24 final transcripts, with sections defined as Opening,
mean(Argument 1, Response 1), mean(Argument 2, Response 2), and
mean(Argument 3, Response 3):

- **0 of 264** per-debate-per-dimension paired t-tests showed a
  significant Debater-A-vs-Debater-B difference (α = 0.05).
- **13 of 1,056** section × dimension pairs (1.2%) had an
  |A − B median| greater than one rating point on the 1-5 scale; the
  remaining 98.8% were within ±1.

See `data/ratings/diff_significance_report_by_section.txt` for the
per-dimension breakdown.

## Layout

```
balancing/
├── README.md                                  this file
├── requirements.txt                           anthropic, tenacity, scipy
├── prompts/                                   the 11 rubric prompts
│   ├── _5-point-in-context_TEMPLATE.txt          shared structural template
│   ├── all-dimension-descriptions.txt            one-paragraph defs of all 11
│   ├── authoritative_tone_5-point-in-context.txt
│   ├── clarity_5-point-in-context.txt
│   ├── ... (one per dimension)
│   └── tentativeness_5-point-in-context.txt
├── data/
│   ├── debates/                               24 final balanced transcripts
│   │   ├── branch_bergs_modified.txt
│   │   ├── branch_bergs_modified_2.txt
│   │   ├── ... (one branch per question × A-honest / B-honest pair)
│   │   └── branch_switch_modified_v2.txt
│   └── ratings/                               rerevised ratings (final round)
│       ├── diff_significance_report_by_section.txt
│       ├── authoritative_tone/
│       │   ├── turn_authoritative_tone.csv          long: 1 row per (file, debater, turn)
│       │   ├── turn_authoritative_tone_pairs.csv    1 row per (file, turn label)
│       │   ├── turn_authoritative_tone_stats.csv    per-debate paired t-test
│       │   └── raw_responses/                        cached Claude API responses
│       │       └── branch_bergs_modified.txt/
│       │           ├── run_1.txt
│       │           ├── ... run_2..5 ...
│       │           └── run_5.txt
│       └── ... (one subfolder per dimension)
└── src/
    ├── rate_debate_dimensions.py              calls Claude; writes CSVs + report
    └── summarize_dimension_diffs.py           reads CSVs; writes report only
```


