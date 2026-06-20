"""Fit the secondary model with Student-t residuals (alternative residual spec).

From CF_analysis_plan.docx:
  "If the posterior predictive checks reveal a qualitative mismatch with the
  data (e.g., bimodality, extreme skew, floor/ceiling effects not captured by
  the model), we will fit an alternative model with Student-t residuals and
  report both."

Stage 8 (model_fit_checks.py) found excess kurtosis of +3.69 on the Normal-
residual secondary fit -- at the plan's switch-to-Student-t threshold -- plus
wide tails (min residual = -8.99). This script fits the same hierarchical
specification but with Student-t residuals.

Model:
    y_secondary_i ~ StudentT(nu, mu_i, sigma)
    mu_i = alpha
         + beta_order * order_i
         + beta_domain_1 * I(domain_i = "astrophysics")
         + beta_domain_2 * I(domain_i = "coding")
         + u_question[q(i)]      with u_question ~ Normal(0, sigma_question)
         + u_transcript[v(i)]    with u_transcript ~ Normal(0, sigma_transcript)

Priors (identical to fit_secondary_model.py except for nu):
    alpha ~ Normal(0, 2)
    beta_order, beta_domain_k ~ Normal(0, 1)
    sigma_residual, sigma_question, sigma_transcript ~ HalfNormal(1)
    nu ~ Gamma(2, 0.1)            # Bambi / Stan / brms convention

The nu prior puts most mass on small-to-moderate degrees of freedom (median
~11), letting the data pull nu wherever it needs. Small nu -> heavy tails;
large nu -> approximately Normal.

Sampling and diagnostics follow the preregistration via ``fit_protocol``.

Output: output/secondary-t-fit.nc
"""

import os
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=,mode=NUMBA")

import sys
from pathlib import Path

import arviz as az
import bambi as bmb
import pandas as pd

import fit_protocol

LABEL = "secondary-t"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cleaned"
OUT_DIR = Path(os.environ.get("ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent / "output")))
OUT_DIR.mkdir(exist_ok=True)
DATA = Path(os.environ.get("ANALYSIS_DATA_CSV", str(DATA_DIR / "quant-1-official-study-analysis-ready.csv")))
OUT = OUT_DIR / f"{LABEL}-fit.nc"

CHAINS = 4
TUNE = 2000
DRAWS = 2000
SEED = 42

DOMAIN_TERM = "C(domain, Treatment('brainteasers'))"

df = pd.read_csv(DATA, encoding="utf-8")
df["question"] = df["question"].astype("category")
df["transcript"] = df["transcript"].astype("category")

print(f"Loaded {len(df)} rows from {DATA.name}")
print(f"  domains:       {dict(df['domain'].value_counts())}")
print(f"  n_questions:   {df['question'].nunique()}")
print(f"  n_transcripts: {df['transcript'].nunique()}")


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
        "nu": bmb.Prior("Gamma", alpha=2, beta=0.1),
    }
    return bmb.Model(
        f"y_secondary ~ 1 + order + {DOMAIN_TERM}"
        " + (1|question) + (1|transcript)",
        data=df,
        family="t",
        priors=priors,
    )


print("\nModel:")
print(build_model())

result = fit_protocol.run_protocol(
    label=LABEL,
    build_model=build_model,
    sigma_var_names=["1|question_sigma", "1|transcript_sigma"],
    chains=CHAINS, tune=TUNE, draws=DRAWS, seed=SEED,
    out_dir=OUT_DIR,
)

result.idata.to_netcdf(str(OUT))
print(f"\nSaved InferenceData -> {OUT}")
if result.tightened_sigmas:
    print(
        f"NOTE: prior tightening (step c) was applied: "
        f"{result.tightened_sigmas}"
    )

print("\nFixed effects + variance components + nu:")
focus = az.summary(
    result.idata,
    var_names=["Intercept", "order", "C(domain", "sigma", "_sigma", "nu"],
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
