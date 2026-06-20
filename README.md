This repository accompanies the paper *"DC-Debates: Difficulty-Calibrated Debate Transcripts as Stimuli for Scalable Oversight Research with Human Participants."* It contains the debate transcripts released as stimuli, the pipelines used to produce and balance them, and the data, code, and materials from the qualitative study, the two quantitative pilots, and the main quantitative study described in the paper.

Each of the six top-level folders contains its own README with more detail on contents, layout, and reproduction steps where relevant.

## Headline pointers

- **Main-study data:** the de-identified dataset for the main experiment described in the paper is `quant_study_main/data/cleaned/quant-1-official-study-noids.xlsx`.
- **Final debate transcripts:** the final versions of the debates described in the paper are in `debates/quant_study_main/`, provided in HTML, JSON, plain text, and XML formats.

## Top-level layout

- **`balancing/`** — Diagnostic balancing pipeline applied to the final 24 debate transcripts: prompts, balanced transcripts, rerevised rating CSVs, and the scripts used to produce them.
- **`debates/`** — Source debate transcripts across stages of stimulus development (initial drafts, qualitative-study versions, "treeified" intermediate restructurings, pilot 1, pilot 2, and the final main-study set), in HTML, JSON, plain text, and XML formats. Organised by stage of development rather than by experiment.
- **`qual_study/`** — Qualitative study (free-form protocol): de-identified dataset, analysis code, analysis outputs, and experimental materials.
- **`quant_pilot_1/`** — Quantitative Pilot 1 (recursive protocol): de-identified dataset, descriptive-analysis code and outputs, lookups, and experimental materials.
- **`quant_pilot_2/`** — Quantitative Pilot 2 (recursive protocol, revised): de-identified dataset, descriptive-analysis code and outputs, lookups, and experimental materials.
- **`quant_study_main/`** — Main quantitative difficulty-ranking study (recursive protocol, final): de-identified dataset, Bayesian analysis code, analysis outputs, lookups, and experimental materials.
