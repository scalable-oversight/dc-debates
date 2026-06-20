#!/usr/bin/env bash
# Run the full analysis pipeline in order, teeing each stage's stdout +
# stderr to <OUT_DIR>/<NN>-<name>.log. Exits on the first failing stage.
#
# Usage:
#   ./run_all.sh
#
# Reads:    ../../data/cleaned/quant-1-official-study-noids.xlsx
# Outputs:  ../../data/cleaned/quant-1-official-study-analysis-ready.csv
#           ../../data/cleaned/quant-1-official-study-beta-long.csv
#           ./output/*.nc, *.csv, *.png, *.txt, *.log

set -euo pipefail

cd "$(dirname "$0")"

OUT_DIR="output"
mkdir -p "$OUT_DIR"

# Resolve to absolute paths so the Python scripts pick them up regardless of
# how this script was invoked.
export ANALYSIS_OUT_DIR
ANALYSIS_OUT_DIR="$(pwd)/${OUT_DIR}"

run_stage() {
    local n="$1" name="$2"
    shift 2
    local log="${ANALYSIS_OUT_DIR}/${n}-${name}.log"
    echo
    echo "=== [${n}] ${name} :: $* ==="
    "$@" 2>&1 | tee "$log"
}

echo "Pipeline started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output dir   : ${ANALYSIS_OUT_DIR}"

run_stage 00 build_analysis_dataset python3 build_analysis_dataset.py

run_stage 01 fit_primary       python3 fit_primary_model.py
run_stage 02 fit_secondary     python3 fit_secondary_model.py
run_stage 03 tiers_primary     python3 derive_tiers.py primary
run_stage 04 tiers_secondary   python3 derive_tiers.py secondary
run_stage 05 hyp_primary       python3 compute_hypothesis_delta.py primary
run_stage 06 hyp_secondary     python3 compute_hypothesis_delta.py secondary
run_stage 07 compare           python3 compare_primary_secondary.py
run_stage 08 model_fit_checks  python3 model_fit_checks.py primary secondary
run_stage 09 plot_tiers        python3 plot_tier_classification.py
run_stage 10 plot_outcomes     python3 plot_outcome_hdis.py

# ---------------------------------------------------------------------------
# Alternative-residual refit for the secondary outcome (stages 11-15).
# Per CF_analysis_plan.docx: "If the posterior predictive checks reveal a
# qualitative mismatch with the data (e.g., bimodality, extreme skew,
# floor/ceiling effects not captured by the model), we will fit an alternative
# model with Student-t residuals and report both." Stage 8 finds excess
# kurtosis of ~3.7 on the Normal-residual secondary fit; these stages produce
# the parallel Student-t version of secondary and re-run stage 8 to include it
# in the model-fit summary.
# ---------------------------------------------------------------------------
run_stage 11 fit_secondary_t          python3 fit_secondary_student_t.py
run_stage 12 tiers_secondary_t        python3 derive_tiers.py secondary-t
run_stage 13 hyp_secondary_t          python3 compute_hypothesis_delta.py secondary-t
run_stage 14 compare_secondary_t      python3 compare_primary_secondary_t.py
run_stage 15 model_fit_checks_all     python3 model_fit_checks.py primary secondary secondary-t
run_stage 15b plot_outcome_hdi_t      python3 plot_outcome_hdis_t.py

# ---------------------------------------------------------------------------
# Exploratory 3-tier classification (stages 16-18). Parallels stages 03/04/09
# but bins the 24 transcripts into 3 equal tiers of 8 instead of 4 of 6.
# Primary and secondary outcomes only (no secondary-t).
# ---------------------------------------------------------------------------
run_stage 16 tiers3_primary           python3 derive_tiers_3.py primary
run_stage 17 tiers3_secondary         python3 derive_tiers_3.py secondary
run_stage 18 plot_tiers3              python3 plot_tier_classification_3.py

# Side-by-side comparison plot of the primary and secondary fits' fixed
# effects + variance components, for quick visual inspection.
run_stage 19 plot_fx_hdi              python3 plot_fixed_effects_hdi.py

# 3 (domain) x 4 (modal tier) grid placement of each transcript, one PNG
# per outcome.
run_stage 19b plot_modal_tier_grid    python3 plot_modal_tier_grid.py

# ---------------------------------------------------------------------------
# Exploratory Bernoulli model on the SIGN of the secondary outcome (stages
# 20-23): 1 = participant updated toward the correct answer between part 1
# and part 4, 0 = updated away; exact ties (~10% of rows) are dropped.
# ---------------------------------------------------------------------------
run_stage 20 fit_secondary_sign       python3 fit_secondary_sign.py
run_stage 21 tiers_secondary_sign     python3 derive_tiers.py secondary-sign
run_stage 22 plot_outcome_hdi_sign    python3 plot_outcome_hdis_sign.py
run_stage 23 plot_tiers_sign          python3 plot_tier_classification_sign.py

# ---------------------------------------------------------------------------
# Extended model-fit diagnostics (stage 24): PSIS-LOO + Pareto k for all four
# fits, QQ plots vs the model-implied distribution for the continuous-outcome
# fits, LOO comparison of secondary Normal vs Student-t, and Bernoulli-specific
# diagnostics (calibration curve, aggregate PPC, per-transcript predicted-vs-
# observed, discrimination metrics) for the sign model.
# ---------------------------------------------------------------------------
run_stage 24 model_fit_checks_extra   python3 model_fit_checks_extra.py primary secondary secondary-t secondary-sign

# ---------------------------------------------------------------------------
# Exploratory Beta-likelihood alternative for the secondary outcome
# (stages 25-27). Motivation: y_secondary is mechanically bounded to
# +/-9.190 by the [0.01, 0.99] clamp upstream of to_logit, so the Normal
# and Student-t fits are both misspecified w.r.t. the support. The Beta
# model fits raw section credences in (0, 1) directly.
# ---------------------------------------------------------------------------
run_stage 25 build_beta_dataset       python3 build_beta_dataset.py
run_stage 26 fit_secondary_beta       python3 fit_secondary_beta.py
run_stage 27 model_fit_checks_beta    python3 model_fit_checks_beta.py
run_stage 28 plot_outcome_hdi_beta    python3 plot_outcome_hdis_beta.py

# ---------------------------------------------------------------------------
# Beta-fit tier classifications + plots paralleling the Normal/Student-t
# tier plots. Stage 29 derives the 4-tier and 3-tier classifications from
# secondary-beta-fit.nc; stage 29b also runs derive_tiers_3.py on the
# Student-t fit so the t-vs-beta comparison plot in stage 30c has both
# inputs. Stages 30/30b/30c produce the Beta forest, the Beta
# modal-tier grid, and the side-by-side Student-t vs Beta tier-uncertainty
# plots.
# ---------------------------------------------------------------------------
run_stage 29  tiers_secondary_beta    python3 derive_tiers_secondary_beta.py
run_stage 29b tiers3_secondary_t      python3 derive_tiers_3.py secondary-t
run_stage 30  plot_fx_hdi_beta        python3 plot_fixed_effects_hdi_beta.py
run_stage 30b plot_modal_tier_grid_beta  python3 plot_modal_tier_grid_beta.py
run_stage 30c plot_tiers_t_vs_beta    python3 plot_tier_classification_t_vs_beta.py

echo
echo "Pipeline completed at $(date '+%Y-%m-%d %H:%M:%S')"
