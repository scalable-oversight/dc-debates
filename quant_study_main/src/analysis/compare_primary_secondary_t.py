"""Compare the primary and Student-t-secondary fits via Kendall's tau + tier cross-tab.

Mirrors compare_primary_secondary.py but uses secondary-t-fit.nc in place of
secondary-fit.nc. Lets us see whether switching the secondary's residual
distribution to Student-t changes the primary-vs-secondary rank-correlation
or modal-tier crosstab story.

For each posterior draw we compute Kendall's tau between the 24-vector of
primary delta_v values and the 24-vector of secondary-t delta_v values
(paired by draw index; the two MCMC runs are independent, so any pairing is
valid).

Reports the posterior mean + 95% HDI of tau, prints the 4x4 cross-tab of
modal tier assignments, and writes output/tier-crosstab-primary-vs-secondary-t.csv.
"""

import os
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=,mode=NUMBA")

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cleaned"
OUT_DIR = Path(os.environ.get("ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent / "output")))
OUT_DIR.mkdir(exist_ok=True)
DATA = Path(os.environ.get("ANALYSIS_DATA_CSV", str(DATA_DIR / "quant-1-official-study-analysis-ready.csv")))
PRIM = OUT_DIR / "primary-fit.nc"
SEC_T = OUT_DIR / "secondary-t-fit.nc"
OUT = OUT_DIR / "tier-crosstab-primary-vs-secondary-t.csv"

DOMAIN_VAR = "C(domain, Treatment('brainteasers'))"
DOMAIN_DIM = f"{DOMAIN_VAR}_dim"

# ---------------------------------------------------------------------------
# Shared: build delta_v matrix of shape (n_draws, 24) from a fit.
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA, encoding="utf-8")
lookup = (
    df[["transcript", "question", "domain"]]
    .drop_duplicates()
    .sort_values("transcript")
    .reset_index(drop=True)
)
assert len(lookup) == 24


def compute_delta_v(idata):
    post = idata.posterior
    u_t = post["1|transcript"].stack(sample=("chain", "draw")).values
    u_q = post["1|question"].stack(sample=("chain", "draw")).values
    beta_d = post[DOMAIN_VAR].stack(sample=("chain", "draw")).values
    n_draws = u_t.shape[1]

    p_t = [str(t) for t in post["transcript__factor_dim"].values]
    p_q = [str(q) for q in post["question__factor_dim"].values]
    p_d = [str(d) for d in post[DOMAIN_DIM].values]

    out = np.empty((n_draws, 24), dtype=float)
    for vi, row in lookup.iterrows():
        v_post = u_t[p_t.index(str(row["transcript"]))]
        q_post = u_q[p_q.index(row["question"])]
        if row["domain"] in p_d:
            d_post = beta_d[p_d.index(row["domain"])]
        else:
            d_post = np.zeros(n_draws)
        out[:, vi] = -(d_post + q_post + v_post)
    return out


prim_idata = az.from_netcdf(PRIM)
sec_t_idata = az.from_netcdf(SEC_T)
delta_prim = compute_delta_v(prim_idata)
delta_sec_t = compute_delta_v(sec_t_idata)

n = min(delta_prim.shape[0], delta_sec_t.shape[0])
delta_prim = delta_prim[:n]
delta_sec_t = delta_sec_t[:n]
print(f"Comparing on {n} paired draws, 24 transcripts each.")

# ---------------------------------------------------------------------------
# Posterior Kendall's tau, paired by draw index.
# ---------------------------------------------------------------------------
taus = np.empty(n, dtype=float)
for i in range(n):
    taus[i], _ = kendalltau(delta_prim[i], delta_sec_t[i])

tau_hdi = az.hdi(taus, hdi_prob=0.95)
print("\nKendall's tau between primary and secondary-t delta_v vectors:")
print(f"  posterior mean   : {taus.mean():+.3f}")
print(f"  posterior median : {float(np.median(taus)):+.3f}")
print(f"  95% HDI          : [{tau_hdi[0]:+.3f}, {tau_hdi[1]:+.3f}]")
print(f"  P(tau > 0)       : {(taus > 0).mean():.3f}")

tau_point, _ = kendalltau(delta_prim.mean(axis=0), delta_sec_t.mean(axis=0))
print(f"  tau on posterior-mean delta_v: {tau_point:+.3f}")

# ---------------------------------------------------------------------------
# Modal-tier cross-tabulation.
# ---------------------------------------------------------------------------
def modal_tiers(delta):
    order = np.argsort(delta, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks, order, np.broadcast_to(np.arange(24), order.shape), axis=1,
    )
    tiers = ranks // 6 + 1
    n_d = delta.shape[0]
    probs = np.stack([(tiers == (t + 1)).sum(axis=0) / n_d for t in range(4)], axis=1)
    return probs.argmax(axis=1) + 1


modal_prim = modal_tiers(delta_prim)
modal_sec_t = modal_tiers(delta_sec_t)

crosstab = pd.crosstab(
    pd.Series(modal_prim, name="primary"),
    pd.Series(modal_sec_t, name="secondary-t"),
).reindex(index=[1, 2, 3, 4], columns=[1, 2, 3, 4], fill_value=0)

print("\nModal tier cross-tab (rows = primary, cols = secondary-t):")
print(crosstab.to_string())

agreement = (modal_prim == modal_sec_t).mean()
print(f"\nExact-tier agreement: {agreement * 100:.1f}% "
      f"({(modal_prim == modal_sec_t).sum()}/{24} transcripts)")

crosstab.to_csv(OUT, encoding="utf-8")
print(f"\nWrote {OUT.name}")
