"""Exploratory: fit a Bernoulli GLM to the SIGN of the secondary outcome.

Same fixed/random structure as the secondary model, but the outcome is the
binary

    y_sign = 1 if y_secondary > 0 else 0    (rows where y_secondary == 0 are
                                             dropped: see build_analysis_dataset.py)

so we model the probability that the participant's credence in the honest
answer moved toward correct between part 1 and part 4 of the debate.

Model:
    y_sign_i ~ Bernoulli(p_i),  logit(p_i) = mu_i
    mu_i = alpha + beta_order * order_i
         + beta_domain_1 * I(domain_i = "astrophysics")
         + beta_domain_2 * I(domain_i = "coding")
         + u_question[q(i)]      with u_question ~ Normal(0, sigma_question)
         + u_transcript[v(i)]    with u_transcript ~ Normal(0, sigma_transcript)

Priors: alpha ~ Normal(0, 2); beta_order, beta_domain ~ Normal(0, 1);
sigma_question, sigma_transcript ~ HalfNormal(1). (No residual sigma: the
Bernoulli likelihood has no separate scale parameter.)

Sampling and diagnostics follow the preregistration via ``fit_protocol``.

Output: <out_dir>/secondary-sign-fit.nc  (ArviZ InferenceData via NetCDF).
"""

import os
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=,mode=NUMBA")

import sys
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

import fit_protocol

LABEL = "secondary-sign"
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

# ---------------------------------------------------------------------------
# 1. Load data; drop ties (y_sign NaN); cast grouping factors.
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA, encoding="utf-8")
n_total = len(df)
# y_sign is added by build_analysis_dataset.py, but older analysis-ready CSVs
# predate that column. y_sign is just sign(y_secondary), so derive it here
# when missing rather than failing on a stale file.
if "y_sign" not in df.columns:
    df["y_sign"] = np.where(
        df["y_secondary"] > 0, 1.0,
        np.where(df["y_secondary"] < 0, 0.0, np.nan),
    )
df = df.dropna(subset=["y_sign"]).copy()
df["y_sign"] = df["y_sign"].astype(int)
df["question"] = df["question"].astype("category")
df["transcript"] = df["transcript"].astype("category")
n_dropped = n_total - len(df)

print(f"Loaded {n_total} rows from {DATA.name}; dropped {n_dropped} ties; fitting on {len(df)}.")
print(f"  y_sign=1 (toward correct): {(df['y_sign'] == 1).sum()}")
print(f"  y_sign=0 (away from correct): {(df['y_sign'] == 0).sum()}")
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

    priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=2),
        "order": bmb.Prior("Normal", mu=0, sigma=1),
        DOMAIN_TERM: bmb.Prior("Normal", mu=0, sigma=1),
        "1|question": group_prior("1|question"),
        "1|transcript": group_prior("1|transcript"),
    }
    return bmb.Model(
        f"y_sign ~ 1 + order + {DOMAIN_TERM}"
        " + (1|question) + (1|transcript)",
        data=df,
        family="bernoulli",
        link="logit",
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

print("\nFixed effects + variance components:")
focus = az.summary(
    result.idata,
    var_names=["Intercept", "order", "C(domain", "_sigma"],
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
