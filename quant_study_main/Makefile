.PHONY: all

# Stages 1 and 2 of the pipeline (raw CSV + Prolific demographic CSVs ->
# with-IDs xlsx -> no-ids xlsx) cannot be run from this public distribution
# because their inputs were withheld for PII reasons. The stage-1+2 scripts
# are preserved under src/pipeline_record/ as a process record; see
# src/pipeline_record/README.md.
#
# The de-identified output of those stages is shipped at
# data/cleaned/quant-1-official-study-noids.xlsx. The full Bayesian analysis
# pipeline (stages 3+) reads it directly and is driven by
# src/analysis/run_all.sh, which writes outputs to src/analysis/output/.

all:
	@echo "Stages 1 and 2 cannot be run from this public release; their inputs"
	@echo "were withheld for PII reasons. See README.md and"
	@echo "src/pipeline_record/README.md for context."
	@echo
	@echo "To run the Bayesian analysis pipeline against the shipped no-ids xlsx:"
	@echo
	@echo "    cd src/analysis && ./run_all.sh"
