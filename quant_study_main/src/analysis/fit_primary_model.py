"""Fit the primary Bayesian hierarchical model from CF_analysis_plan.docx.

Model:
    y_primary_i ~ Normal(mu_i, sigma_residual)
    mu_i = alpha
         + beta_order * order_i
         + beta_domain_1 * I(domain_i = "astrophysics")
         + beta_domain_2 * I(domain_i = "coding")
         + u_question[q(i)]      with u_question ~ Normal(0, sigma_question)
         + u_transcript[v(i)]    with u_transcript ~ Normal(0, sigma_transcript)

Priors:
    alpha ~ Normal(0, 2)
    beta_order, beta_domain_k ~ Normal(0, 1)
    sigma_residual, sigma_question, sigma_transcript ~ HalfNormal(1)

Sampling and diagnostics follow the preregistration. The escalation cascade,
non-centered-parameterization assertion, sigma-compression prior tightening
(step c), R-hat/ESS doubled-draws refit, energy/E-BFMI/acceptance/tree-depth
reporting, and tree-depth-saturation rule are implemented in ``fit_protocol``.

Output: <out_dir>/primary-fit.nc  (ArviZ InferenceData via NetCDF).
"""

import os
# PyTensor's default C backend needs python3-dev / Python.h, which is missing
# in this container. Numba mode JITs the model log-prob and is comparable in
# speed to the C backend (~10x faster than pure-Python fallback). Set before
# importing pytensor/pymc/bambi.
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=,mode=NUMBA")

import sys
from pathlib import Path

import arviz as az
import bambi as bmb
import pandas as pd

import fit_protocol

LABEL = "primary"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cleaned"
OUT_DIR = Path(os.environ.get("ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent / "output")))
OUT_DIR.mkdir(exist_ok=True)
DATA = Path(os.environ.get("ANALYSIS_DATA_CSV", str(DATA_DIR / "quant-1-official-study-analysis-ready.csv")))
OUT = OUT_DIR / f"{LABEL}-fit.nc"

CHAINS = 4
TUNE = 2000
DRAWS = 2000
SEED = 42

# Term name as produced by formulae for the domain C(...) wrapper. Used both
# in the formula string and as the prior key, so they have to stay in sync.
DOMAIN_TERM = "C(domain, Treatment('brainteasers'))"

# ---------------------------------------------------------------------------
# 1. Load data; question/transcript as categoricals for the grouping factors.
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA, encoding="utf-8")
df["question"] = df["question"].astype("category")
df["transcript"] = df["transcript"].astype("category")

print(f"Loaded {len(df)} rows from {DATA.name}")
print(f"  domains:       {dict(df['domain'].value_counts())}")
print(f"  n_questions:   {df['question'].nunique()}")
print(f"  n_transcripts: {df['transcript'].nunique()}")


# ---------------------------------------------------------------------------
# 2. Model builder. ``tighten_sigmas`` lets fit_protocol implement
#    preregistration step (c) by rebuilding with HalfNormal(0.5) priors on
#    a variance component pressed against zero.
# ---------------------------------------------------------------------------
def build_model(tighten_sigmas=None):
    tighten_sigmas = tighten_sigmas or {}

    def group_prior(term_key):
        hn_sigma = tighten_sigmas.get(term_key, 1.0)
        return bmb.Prior(
            "Normal", mu=0,
            sigma=bmb.Prior("HalfNormal", sigma=hn_sigma),
        )

    sigma_resid = bmb.Prior(
        "HalfNormal", sigma=tighten_sigmas.get("sigma", 1.0)
    )

    priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=2),
        "order": bmb.Prior("Normal", mu=0, sigma=1),
        DOMAIN_TERM: bmb.Prior("Normal", mu=0, sigma=1),
        "1|question": group_prior("1|question"),
        "1|transcript": group_prior("1|transcript"),
        "sigma": sigma_resid,
    }
    return bmb.Model(
        f"y_primary ~ 1 + order + {DOMAIN_TERM}"
        " + (1|question) + (1|transcript)",
        data=df,
        priors=priors,
    )


print("\nModel:")
print(build_model())

# ---------------------------------------------------------------------------
# 3. Run the preregistered fit protocol.
# ---------------------------------------------------------------------------
result = fit_protocol.run_protocol(
    label=LABEL,
    build_model=build_model,
    sigma_var_names=["1|question_sigma", "1|transcript_sigma"],
    chains=CHAINS, tune=TUNE, draws=DRAWS, seed=SEED,
    out_dir=OUT_DIR,
)

# ---------------------------------------------------------------------------
# 4. Save InferenceData.
# ---------------------------------------------------------------------------
result.idata.to_netcdf(str(OUT))
print(f"\nSaved InferenceData -> {OUT}")
if result.tightened_sigmas:
    print(
        f"NOTE: prior tightening (step c) was applied: "
        f"{result.tightened_sigmas}"
    )

# ---------------------------------------------------------------------------
# 5. Report key parameter summaries.
# ---------------------------------------------------------------------------
print("\nFixed effects + variance components:")
focus = az.summary(
    result.idata,
    var_names=["Intercept", "order", "C(domain", "sigma", "_sigma"],
    filter_vars="like",
)
print(focus.to_string())

print(f"\nFinal divergent transitions: {result.final_divergences}")

if not result.convergence_ok:
    print(
        "\nWARNING: convergence diagnostics still failing after the full "
        "preregistered protocol."
    )
    bad = result.summary.query(
        f"r_hat >= {fit_protocol.RHAT_CEILING} "
        f"or ess_bulk < {fit_protocol.ESS_FLOOR} "
        f"or ess_tail < {fit_protocol.ESS_FLOOR}"
    )
    if len(bad):
        print("Offending parameters (R-hat / ESS):")
        print(bad.to_string())
    if not result.ebfmi_ok:
        print(f"E-BFMI is below {fit_protocol.EBFMI_FLOOR} on at least one chain.")
    if result.final_divergences > 0:
        print(f"Divergent transitions remain: {result.final_divergences}.")
    sys.exit(1)

print("\nAll convergence checks passed.")
