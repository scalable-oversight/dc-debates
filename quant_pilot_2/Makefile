.PHONY: all clean descriptives

PYTHON ?= python3

CLEANED_DIR   := data/cleaned
NOIDS_XLSX    := $(CLEANED_DIR)/quant-1-pilot-2-no-ids.xlsx

LOOKUPS_DIR   := lookups
MAPPING       := $(LOOKUPS_DIR)/debate-to-name-and-question-mapping.csv

DESC_DIR      := data/analyses/descriptives
DESC_SCRIPTS  := $(wildcard src/analyses/descriptives/*.py)
DESC_STAMP    := $(DESC_DIR)/.stamp

# Stages 1 and 2 of the pipeline (raw CSV -> with-IDs xlsx -> no-ids xlsx)
# cannot be run from this public distribution because their inputs were
# withheld for PII reasons. The stage-1+2 scripts are preserved under
# src/pipeline_record/ as a process record; see src/pipeline_record/README.md.
# The de-identified output of those stages is shipped at $(NOIDS_XLSX), and
# stage 3 (descriptives) can be re-executed against it.

all: descriptives

descriptives: $(DESC_STAMP)

$(DESC_STAMP): $(NOIDS_XLSX) $(MAPPING) $(DESC_SCRIPTS)
	mkdir -p $(DESC_DIR)
	@for s in $(DESC_SCRIPTS); do \
	    echo "→ $$s"; \
	    $(PYTHON) $$s --input $(NOIDS_XLSX) --mapping $(MAPPING) --out-dir $(DESC_DIR) || exit 1; \
	done
	touch $(DESC_STAMP)

# `make clean` only removes regenerable analysis outputs. The shipped no-ids
# xlsx is NOT regenerable from this public distribution and is intentionally
# preserved across `make clean`.
clean:
	rm -rf $(DESC_DIR)
