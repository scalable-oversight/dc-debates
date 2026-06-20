"""Fit the Beta-likelihood alternative for the secondary outcome.

Exploratory robustness check for the secondary fit, motivated by the
boundary-clipping diagnostic in
``../analysis_plan_chat/student-t.txt`` (the Student-t table's row
"Consistency w/ boundary-clipping hypothesis :: Student-t doesn't address
this cleanly").

The existing secondary model fits ``y_secondary = logit(p_4) - logit(p_1)``
with a Normal residual; ``y_secondary`` is mechanically bounded to
``+/-2 * logit(0.99) = +/-9.190`` because the upstream pipeline clamps each
section's credence to [0.01, 0.99]. The Student-t alternative absorbs the
resulting pile-ups by shrinking nu to ~2.75 (infinite-variance territory)
while halving sigma_transcript. This script fits a separate model that
honors the bounded support directly: a hierarchical Beta on the raw
section credences in (0, 1).

Model:
    p_i ~ Beta(mu_i * kappa, (1 - mu_i) * kappa)
    logit(mu_i) = alpha
                + beta_order * order_i
                + beta_section * I(section_i = 4)
                + beta_domain_1 * I(domain_i = "astrophysics")
                + beta_domain_2 * I(domain_i = "coding")
                + u_question[q(i)]              with u_question         ~ Normal(0, sigma_question)
                + u_transcript[v(i)]            with u_transcript       ~ Normal(0, sigma_transcript)
                + u_transcript_sec[v(i)] * I(section_i = 4)
                                                with u_transcript_sec   ~ Normal(0, sigma_transcript_section)
                + u_participant[p(i)]           with u_participant      ~ Normal(0, sigma_participant)

  ``section_is_4`` is 0 for the section-1 observation and 1 for the section-4
  observation of each participant; its coefficient is the population-mean
  shift in logit credence-toward-correct from section 1 to section 4 -- the
  Beta analogue of the y_secondary intercept in the Normal-residual fit.

  ``(0 + section_is_4 | transcript)`` adds a per-transcript section slope --
  the strict Beta analogue of the secondary model's transcript random effect.
  With it, the per-transcript expected logit-credence update from section 1
  to section 4 is ``beta_section + u_transcript_sec[v]``, which varies across
  transcripts; without it the per-transcript update collapses to the global
  ``beta_section`` for every transcript. The intercept term ``(1|transcript)``
  still captures per-transcript variation in section-1 logit credence; the
  two transcript random effects are fit as independent draws (we did not add
  a correlation parameter between them).

Priors (matching fit_secondary_model.py where the parameters overlap):
    alpha                                            ~ Normal(0, 2)
    beta_order, beta_section, beta_dom_k             ~ Normal(0, 1)
    sigma_{question,transcript,transcript_section,participant}
                                                     ~ HalfNormal(1)
    kappa (Beta precision)                           ~ Bambi default (HalfCauchy)

Sampling and diagnostics follow the preregistration via ``fit_protocol``.

Output: ``output/secondary-beta-fit.nc``

Usage:
    python3 fit_secondary_beta.py

When invoked from ``run_all.sh``:
    Reads the beta-long CSV path derived from ``$ANALYSIS_DATA_CSV`` (so
    variant suffixes like ``-no-pangram`` flow through), and writes the fit
    under ``$ANALYSIS_OUT_DIR``.
"""

import os
# Match the Numba backend choice from the other fit scripts (the container
# is missing python3-dev, so PyTensor's C backend can't compile).
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=,mode=NUMBA")

import sys
from pathlib import Path

import arviz as az
import bambi as bmb
import pandas as pd

import fit_protocol

LABEL = "secondary-beta"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cleaned"
OUT_DIR = Path(
    os.environ.get(
        "ANALYSIS_OUT_DIR",
        str(Path(__file__).resolve().parent / "output"),
    )
)
OUT_DIR.mkdir(exist_ok=True)


def _default_beta_long_csv() -> Path:
    """Derive the beta-long CSV path from ANALYSIS_DATA_CSV when set.

    Lets ``./run_all.sh no-pangram`` (etc.) automatically pick up the matching
    ``-no-pangram`` beta-long CSV without any extra config.
    """
    ar = os.environ.get("ANALYSIS_DATA_CSV")
    if ar:
        p = Path(ar)
        if "analysis-ready" in p.name:
            return p.parent / p.name.replace("analysis-ready", "beta-long")
    return DATA_DIR / "quant-1-official-study-beta-long.csv"


DATA = Path(os.environ.get("ANALYSIS_BETA_LONG_CSV", str(_default_beta_long_csv())))
OUT = OUT_DIR / f"{LABEL}-fit.nc"

CHAINS = 4
TUNE = 2000
DRAWS = 2000
SEED = 42

DOMAIN_TERM = "C(domain, Treatment('brainteasers'))"

if not DATA.exists():
    sys.exit(
        f"ERROR: beta-long CSV not found: {DATA}\n"
        f"Build it with: python3 ../data/build_beta_dataset.py"
    )

df = pd.read_csv(DATA, encoding="utf-8")
df["question"] = df["question"].astype("category")
df["transcript"] = df["transcript"].astype("category")
df["participant_id"] = df["participant_id"].astype("category")

print(f"Loaded {len(df)} rows from {DATA.name}")
print(f"  participants:  {df['participant_id'].nunique()}")
print(f"  transcripts:   {df['transcript'].nunique()}")
print(f"  questions:     {df['question'].nunique()}")
print(f"  p range:       [{df['p'].min():.3f}, {df['p'].max():.3f}]")


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
        "section_is_4": bmb.Prior("Normal", mu=0, sigma=1),
        DOMAIN_TERM: bmb.Prior("Normal", mu=0, sigma=1),
        "1|question": group_prior("1|question"),
        "1|transcript": group_prior("1|transcript"),
        "section_is_4|transcript": group_prior("section_is_4|transcript"),
        "1|participant_id": group_prior("1|participant_id"),
    }
    return bmb.Model(
        f"p ~ 1 + order + section_is_4 + {DOMAIN_TERM}"
        " + (1|question) + (1|transcript) + (0 + section_is_4|transcript)"
        " + (1|participant_id)",
        data=df,
        family="beta",
        priors=priors,
    )


print("\nModel:")
print(build_model())

result = fit_protocol.run_protocol(
    label=LABEL,
    build_model=build_model,
    sigma_var_names=[
        "1|question_sigma",
        "1|transcript_sigma",
        "section_is_4|transcript_sigma",
        "1|participant_id_sigma",
    ],
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

print("\nFixed effects + variance components + Beta precision:")
focus = az.summary(
    result.idata,
    var_names=[
        "Intercept",
        "order",
        "section_is_4",
        "C(domain",
        "_sigma",
        "kappa",
        "phi",
    ],
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
