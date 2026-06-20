"""Posterior of the preregistered hypothesis contrast Delta.

From CF_analysis_plan.docx:

    Delta = mean(delta_v : v in {Bergs 1, Bergs 2,
                                 Database deletion 1, Database deletion 2})
          - mean(delta_v : v in {Early Universe Galaxies 1, EUG 2,
                                 Internal Temperature of Stars 1, ITS 2})

A positive Delta means the harder-group transcripts are estimated as harder
than the easier-group transcripts on the logit-credence scale, consistent
with the preregistered expectation.

Reports the posterior mean, 95% HDI, and P(Delta > 0).

Usage: python3 compute_hypothesis_delta.py [primary|secondary]
       (default: primary)
"""

import os
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=,mode=NUMBA")

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

# Analysis-ready CSV lives in ../data/; fits live in ./output/.
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cleaned"
OUT_DIR = Path(os.environ.get("ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent / "output")))

outcome = sys.argv[1] if len(sys.argv) > 1 else "primary"
if outcome not in {"primary", "secondary", "secondary-t"}:
    sys.exit(
        f"Unknown outcome '{outcome}'. "
        f"Use 'primary', 'secondary', or 'secondary-t'."
    )

FIT = OUT_DIR / f"{outcome}-fit.nc"
DATA = Path(os.environ.get("ANALYSIS_DATA_CSV", str(DATA_DIR / "quant-1-official-study-analysis-ready.csv")))

DOMAIN_VAR = "C(domain, Treatment('brainteasers'))"
DOMAIN_DIM = f"{DOMAIN_VAR}_dim"

HARDER_QUESTIONS = {"Bergs", "Database deletion"}
EASIER_QUESTIONS = {"Early Universe Galaxies", "Internal Temperature of Stars"}

# ---------------------------------------------------------------------------
# Load fit + transcript lookup.
# ---------------------------------------------------------------------------
idata = az.from_netcdf(FIT)
post = idata.posterior

df = pd.read_csv(DATA, encoding="utf-8")
lookup = (
    df[["transcript", "question", "domain"]]
    .drop_duplicates()
    .sort_values("transcript")
    .reset_index(drop=True)
)

post_transcripts = [str(t) for t in post["transcript__factor_dim"].values]
post_questions = [str(q) for q in post["question__factor_dim"].values]
post_domains = [str(d) for d in post[DOMAIN_DIM].values]

# ---------------------------------------------------------------------------
# delta_v for the 8 transcripts in the two contrast groups.
# ---------------------------------------------------------------------------
u_transcript = post["1|transcript"].stack(sample=("chain", "draw")).values
u_question = post["1|question"].stack(sample=("chain", "draw")).values
beta_domain = post[DOMAIN_VAR].stack(sample=("chain", "draw")).values
n_draws = u_transcript.shape[1]


def delta_for(transcript_id, question, domain):
    u_t = u_transcript[post_transcripts.index(str(transcript_id))]
    u_q = u_question[post_questions.index(question)]
    if domain in post_domains:
        b_d = beta_domain[post_domains.index(domain)]
    else:
        b_d = np.zeros(n_draws)
    return -(b_d + u_q + u_t)


def group_delta(question_set):
    rows = lookup[lookup["question"].isin(question_set)]
    if len(rows) != 4:
        sys.exit(
            f"ERROR: expected 4 transcripts for {question_set}, got {len(rows)}"
        )
    arr = np.stack([
        delta_for(r["transcript"], r["question"], r["domain"])
        for _, r in rows.iterrows()
    ])  # (4, n_draws)
    print(
        f"  group transcripts: "
        + ", ".join(f"{r['question']} {r['transcript']}" for _, r in rows.iterrows())
    )
    return arr.mean(axis=0)


print(f"Loading {FIT.name} ({n_draws} draws).")
print("\nHarder group:")
harder = group_delta(HARDER_QUESTIONS)
print("\nEasier group:")
easier = group_delta(EASIER_QUESTIONS)

delta_contrast = harder - easier

# ---------------------------------------------------------------------------
# Summarise.
# ---------------------------------------------------------------------------
hdi = az.hdi(delta_contrast, hdi_prob=0.95)
p_pos = float((delta_contrast > 0).mean())

print(f"\nDelta = mean(delta_v in {{Bergs, Database deletion}})")
print(f"      - mean(delta_v in {{EUG, ITS}})")
print()
print(f"  posterior mean   : {delta_contrast.mean():+.3f}")
print(f"  posterior median : {np.median(delta_contrast):+.3f}")
print(f"  95% HDI          : [{hdi[0]:+.3f}, {hdi[1]:+.3f}]")
print(f"  P(Delta > 0)     : {p_pos:.3f}")
