# Debates

This directory contains the source debate transcripts that were written, revised, and converted across the stages of stimulus development for the *DC-Debates* paper, in multiple formats (HTML, plain text, XML, JSON) to support downstream re-use. It is organised by stage of development rather than by experiment.

## Where the experiment stimuli actually live

**The HTML files in this directory are *not* the files that were shown to participants.** The participant-facing HTML stimuli used in each of the experiments described in the paper live in the `experimental_materials/` subdirectories of the root-level study folders:

- `qual_study/experimental_materials/` — qualitative study (free-form protocol)
- `quant_pilot_1/experimental_materials/` — quantitative Pilot 1 (recursive protocol)
- `quant_pilot_2/experimental_materials/` — quantitative Pilot 2 (recursive protocol, revised)
- `quant_study_main/experimental_materials/` — main difficulty-ranking study (recursive protocol, final)

The trees under `debates/` are upstream artefacts: drafts, alternate format conversions, and intermediate restructurings produced while building the eventual stimuli.

## Layout

```
debates/
├── initial_drafts/      Hand-authored and LLM-generated free-form initial drafts
│   ├── problem-texts/                 question prompts (one per debate topic)
│   ├── manual-debate-texts/           hand-authored free-form transcripts
│   ├── llm-cleaned-debate-texts/      LLM-generated free-form transcripts
│   │   ├── claude-opus-4-5-cleaned/
│   │   ├── gemini-pro-cleaned/
│   │   └── gpt-5.2-cleaned/
│   ├── prompts/                       templates used for the LLM cleanup passes
│   │   ├── prompt-templates/
│   │   └── instantiated-cleanup-prompts/
│   ├── html/                          HTML renderings of the drafts
│   └── images/                        figure assets referenced by the drafts
│
├── qual_study/          Free-form stimuli used in the qualitative study, including Redshift (used in the judge selection process)
│   ├── html/  text/  xml/  images/
│
├── qual_treeified/      Intermediate "restructured" / treeified versions of the
│   │                    qualitative study free-form transcripts, produced while exploring move
│   │                    to a recursive protocol (12 transcripts; HTML/TXT/XML).
│   └── images/          figure assets referenced by the transcripts
│
├── quant_pilot_1/       Recursive-protocol transcripts used in quantitative Pilot 1
│   └── html/  json/  text/  xml/  images/         (24 transcripts each)
│
├── quant_pilot_2/       Recursive-protocol transcripts used in quantitative Pilot 2
│   └── html/  json/  text/  xml/  images/         (24 transcripts each)
│
└── quant_study_main/    Recursive-protocol transcripts used in the main study
    └── html/  json/  text/  xml/  images/         (24 transcripts each)
```

