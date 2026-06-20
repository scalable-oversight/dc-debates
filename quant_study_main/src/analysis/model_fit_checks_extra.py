"""Extended model-fit diagnostics: PSIS-LOO + Pareto k, QQ plots, and
Bernoulli-specific diagnostics for the sign model.

Complements model_fit_checks.py (residual moments, PPC, sigma compression)
with the diagnostics that are most informative for adjudicating between the
Normal- and Student-t-residual secondary fits, and for assessing the sign
model on its own (Bernoulli) terms.

For each fit (primary, secondary, secondary-t, secondary-sign):
  * Energy plot (az.plot_energy): overlay of the marginal energy distribution
    and the energy-transition distribution. Healthy NUTS sampling has the two
    distributions on top of each other (numerically summarised by BFMI > 0.3).

For each continuous-outcome fit (primary, secondary, secondary-t):
  * QQ plot of standardized residuals against the model-implied theoretical
    distribution (Normal for primary/secondary; Student-t at posterior-mean
    nu for the Student-t fit).
  * PSIS-LOO with Pareto k diagnostics: ELPD, p_loo, Pareto k distribution,
    and a flagged list of high-leverage observations (k > 0.7) -- these are
    the points driving residual-tail behaviour, and are the right targets if
    we want to ask "is nu being held down by a tiny minority of points?".

For the secondary Normal vs Student-t pair:
  * az.compare ranking by expected log predictive density (elpd_loo).

For the secondary-sign Bernoulli fit:
  * PSIS-LOO + Pareto k.
  * Calibration curve: bin observations by predicted p, compare mean predicted
    p to observed positive rate in each bin.
  * Aggregate PPC: distribution of total y_rep=1 across draws, observed total
    overlaid.
  * Per-transcript predicted-proportion vs observed-proportion scatter.
  * Discrimination summary: AUC, accuracy at 0.5, Brier score (+ skill vs
    base-rate), log loss (+ skill vs base-rate).

Outputs (in OUT_DIR):
  energy-<outcome>.png                      one per outcome (incl. sign)
  qq-<outcome>.png                          one per continuous outcome
  pareto-k-<outcome>.png                    one per outcome (incl. sign)
  loo-compare-secondary-vs-t.txt            text table from az.compare
  sign-calibration.png                      sign-only
  sign-ppc-aggregate.png                    sign-only
  sign-per-transcript.png                   sign-only
  model-fit-extras-summary.txt              consolidated text summary

Usage:
  python3 model_fit_checks_extra.py
  python3 model_fit_checks_extra.py primary secondary secondary-t secondary-sign
"""

import os
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=,mode=NUMBA")

import sys
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import xarray as xr

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cleaned"
OUT_DIR = Path(os.environ.get(
    "ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent / "output")
))
OUT_DIR.mkdir(exist_ok=True)
DATA = Path(os.environ.get(
    "ANALYSIS_DATA_CSV", str(DATA_DIR / "quant-1-official-study-analysis-ready.csv")
))
SUMMARY_PATH = OUT_DIR / "model-fit-extras-summary.txt"

DOMAIN_VAR = "C(domain, Treatment('brainteasers'))"
DOMAIN_DIM = f"{DOMAIN_VAR}_dim"

OUTCOME_TO_COL = {
    "primary": "y_primary",
    "secondary": "y_secondary",
    "secondary-t": "y_secondary",
    "secondary-sign": "y_sign",
}
OUTCOME_ORDER = ["primary", "secondary", "secondary-t", "secondary-sign"]
CONTINUOUS_OUTCOMES = {"primary", "secondary", "secondary-t"}

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Shared mu_i reconstruction (post-hoc, from saved fit).
# ---------------------------------------------------------------------------
def load_dataframe(outcome_label):
    """Return the analysis-ready frame appropriate for this outcome.

    The sign model drops y_secondary == 0 rows; everything else uses the full
    frame. y_sign is derived on the fly so older CSVs without that column still
    work (matches fit_secondary_sign.py).
    """
    df = pd.read_csv(DATA, encoding="utf-8")
    if outcome_label == "secondary-sign":
        if "y_sign" not in df.columns:
            df["y_sign"] = np.where(
                df["y_secondary"] > 0, 1.0,
                np.where(df["y_secondary"] < 0, 0.0, np.nan),
            )
        df = df.dropna(subset=["y_sign"]).reset_index(drop=True).copy()
        df["y_sign"] = df["y_sign"].astype(int)
    return df


def reconstruct_mu(idata, df):
    """Return mu_i with shape (n_chain, n_draw, n_obs).

    Stacked-then-reshaped so az.loo sees a (chain, draw, obs) log-likelihood.
    """
    post = idata.posterior
    n_chain = post.sizes["chain"]
    n_draw = post.sizes["draw"]
    n_total = n_chain * n_draw

    intercept = post["Intercept"].stack(s=("chain", "draw")).values  # (S,)
    b_order = post["order"].stack(s=("chain", "draw")).values
    b_dom = post[DOMAIN_VAR].stack(s=("chain", "draw")).values  # (2, S)
    u_q = post["1|question"].stack(s=("chain", "draw")).values  # (12, S)
    u_t = post["1|transcript"].stack(s=("chain", "draw")).values  # (24, S)

    p_q = [str(q) for q in post["question__factor_dim"].values]
    p_t = [str(t) for t in post["transcript__factor_dim"].values]
    p_d = [str(d) for d in post[DOMAIN_DIM].values]

    obs_q = df["question"].map(lambda q: p_q.index(q)).values
    obs_v = df["transcript"].map(lambda t: p_t.index(str(t))).values
    obs_order = df["order"].values.astype(float)
    obs_d = df["domain"].values

    # (n_obs, S)
    mu = (
        intercept[None, :]
        + obs_order[:, None] * b_order[None, :]
        + u_q[obs_q, :]
        + u_t[obs_v, :]
    )
    if "astrophysics" in p_d:
        mask = obs_d == "astrophysics"
        mu[mask] += b_dom[p_d.index("astrophysics"), :]
    if "coding" in p_d:
        mask = obs_d == "coding"
        mu[mask] += b_dom[p_d.index("coding"), :]

    # Reshape to (n_chain, n_draw, n_obs) for the InferenceData log_likelihood
    # group. The original stack order is (chain, draw) so reversing the stack
    # axis with shape (n_chain, n_draw) is correct.
    mu_3d = mu.T.reshape(n_chain, n_draw, mu.shape[0])
    return mu_3d


def get_param_3d(idata, name):
    """Return a scalar posterior param as (n_chain, n_draw)."""
    arr = idata.posterior[name].values
    # Already (chain, draw) for scalars.
    return arr


def attach_log_likelihood(idata, ll_3d, var_name):
    """Attach a (chain, draw, obs) log-likelihood array to idata as a group."""
    n_chain, n_draw, n_obs = ll_3d.shape
    ll_da = xr.DataArray(
        ll_3d,
        dims=("chain", "draw", f"{var_name}_dim_0"),
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            f"{var_name}_dim_0": np.arange(n_obs),
        },
        name=var_name,
    )
    ll_ds = xr.Dataset({var_name: ll_da})
    # InferenceData.add_groups exists for this; assigning to .log_likelihood
    # directly also works in recent ArviZ versions.
    idata.add_groups({"log_likelihood": ll_ds})
    return idata


# ---------------------------------------------------------------------------
# Per-outcome log-likelihood + LOO.
# ---------------------------------------------------------------------------
def loglik_normal(y, mu_3d, sigma_2d):
    """(n_chain, n_draw, n_obs) Normal log-likelihood."""
    n_chain, n_draw, n_obs = mu_3d.shape
    sigma_b = sigma_2d[:, :, None]  # broadcast over obs
    y_b = y[None, None, :]
    return st.norm.logpdf(y_b, loc=mu_3d, scale=sigma_b)


def loglik_t(y, mu_3d, sigma_2d, nu_2d):
    """(n_chain, n_draw, n_obs) Student-t log-likelihood."""
    sigma_b = sigma_2d[:, :, None]
    nu_b = nu_2d[:, :, None]
    y_b = y[None, None, :]
    return st.t.logpdf(y_b, df=nu_b, loc=mu_3d, scale=sigma_b)


def loglik_bernoulli(y, mu_3d):
    """(n_chain, n_draw, n_obs) Bernoulli-with-logit-link log-likelihood.

    Numerically stable via -softplus(-mu) for log p and -softplus(mu) for
    log(1-p), avoiding overflow when |mu| is large.
    """
    # log p = log sigmoid(mu) = -softplus(-mu)
    # log (1-p) = -softplus(mu)
    log_p = -np.logaddexp(0.0, -mu_3d)
    log_1mp = -np.logaddexp(0.0, mu_3d)
    y_b = y[None, None, :].astype(float)
    return y_b * log_p + (1.0 - y_b) * log_1mp


def run_loo(outcome_label, idata, df, outcome_col):
    """Reconstruct log-likelihood + run PSIS-LOO; return ELPDData."""
    y = df[outcome_col].values.astype(float)
    mu_3d = reconstruct_mu(idata, df)

    if outcome_label == "secondary-t":
        sigma_2d = get_param_3d(idata, "sigma")
        nu_2d = get_param_3d(idata, "nu")
        ll_3d = loglik_t(y, mu_3d, sigma_2d, nu_2d)
    elif outcome_label == "secondary-sign":
        ll_3d = loglik_bernoulli(y, mu_3d)
    else:
        sigma_2d = get_param_3d(idata, "sigma")
        ll_3d = loglik_normal(y, mu_3d, sigma_2d)

    attach_log_likelihood(idata, ll_3d, var_name="y_obs")
    loo = az.loo(idata, pointwise=True)
    return loo, mu_3d


# ---------------------------------------------------------------------------
# QQ plot of standardized residuals vs theoretical distribution.
# ---------------------------------------------------------------------------
def qq_plot(out_path, label, residuals, sigma_mean, nu_mean=None):
    z = residuals / sigma_mean
    n = len(z)
    p = (np.arange(1, n + 1) - 0.5) / n
    if nu_mean is None:
        theoretical = st.norm.ppf(p)
        ref_label = "N(0, 1)"
    else:
        theoretical = st.t.ppf(p, df=nu_mean)
        ref_label = f"t(nu = {nu_mean:.2f})"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(theoretical, np.sort(z), s=10, alpha=0.55, color="#1f77b4")
    lim = max(
        abs(theoretical.min()), abs(theoretical.max()),
        abs(z.min()), abs(z.max()),
    )
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=1, ls="--",
            label="y = x (perfect fit)")
    ax.set_xlabel(f"Theoretical {ref_label} quantiles")
    ax.set_ylabel("Standardized residual (residual / sigma_post_mean)")
    ax.set_title(
        f"QQ plot of standardized residuals vs {ref_label}  ({label})"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Energy plot (NUTS sampler health).
# ---------------------------------------------------------------------------
def energy_plot(out_path, label, idata):
    """Overlay marginal energy (pi_E) and energy-transition (pi_dE) KDEs.

    Healthy NUTS sampling has the two distributions overlap (BFMI > 0.3).
    A wider pi_E than pi_dE means the sampler can't keep up with the
    posterior's energy variation -- a sign of pathological geometry that
    R-hat / ESS can miss.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    az.plot_energy(idata, ax=ax)
    bfmi = az.bfmi(idata)
    ax.set_title(
        f"Energy plot ({label}) -- "
        f"BFMI per chain: {', '.join(f'{b:.2f}' for b in bfmi)}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return [float(b) for b in bfmi]


# ---------------------------------------------------------------------------
# Pareto k plot.
# ---------------------------------------------------------------------------
def pareto_k_plot(out_path, label, loo):
    k = np.asarray(loo.pareto_k.values)
    n = len(k)
    idx = np.arange(n)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = np.where(k < 0.5, "#1f77b4",
             np.where(k < 0.7, "#ff7f0e", "#d62728"))
    ax.scatter(idx, k, c=colors, s=10, alpha=0.7)
    for threshold, color, name in [
        (0.5, "#ff7f0e", "0.5 (ok)"),
        (0.7, "#d62728", "0.7 (bad)"),
    ]:
        ax.axhline(threshold, color=color, lw=1, ls="--",
                   label=f"k = {name}")
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Pareto k")
    ax.set_title(f"PSIS-LOO Pareto k by observation ({label})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bernoulli-specific plots.
# ---------------------------------------------------------------------------
def sign_calibration_plot(out_path, p_mean, y, n_bins=10):
    bins = np.quantile(p_mean, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-9
    bin_idx = np.digitize(p_mean, bins[1:-1])
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        rows.append({
            "pred_mean": float(p_mean[mask].mean()),
            "obs_rate": float(y[mask].mean()),
            "n": int(mask.sum()),
            # SE on observed rate (binomial proportion).
            "se": float(np.sqrt(
                y[mask].mean() * (1 - y[mask].mean()) / mask.sum()
            )),
        })
    cal = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--",
            label="perfect calibration")
    ax.errorbar(
        cal["pred_mean"], cal["obs_rate"], yerr=cal["se"],
        fmt="o", color="#1f77b4", capsize=3, label="binned observed",
    )
    for _, r in cal.iterrows():
        ax.annotate(
            f"n={int(r['n'])}",
            (r["pred_mean"], r["obs_rate"]),
            xytext=(5, 5), textcoords="offset points", fontsize=7,
            color="gray",
        )
    lo = min(cal["pred_mean"].min(), cal["obs_rate"].min()) - 0.05
    hi = max(cal["pred_mean"].max(), cal["obs_rate"].max()) + 0.05
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Mean predicted p in bin")
    ax.set_ylabel("Observed positive rate in bin (+/- SE)")
    ax.set_title("Sign model calibration curve (10 quantile bins)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return cal


def sign_ppc_aggregate_plot(out_path, p_post_2d, y):
    """p_post_2d shape (n_obs, n_draws). Plot histogram of total y_rep=1."""
    n_obs, n_draws = p_post_2d.shape
    # Sample y_rep per (obs, draw).
    y_rep = (RNG.random((n_obs, n_draws)) < p_post_2d).astype(int)
    totals = y_rep.sum(axis=0)
    obs_total = int(y.sum())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(totals, bins=40, color="#1f77b4", alpha=0.6,
            label=f"y_rep totals  (mean {totals.mean():.0f}, sd {totals.std():.0f})")
    ax.axvline(obs_total, color="black", lw=2,
               label=f"observed total = {obs_total}")
    lo, hi = np.percentile(totals, [2.5, 97.5])
    ax.axvline(lo, color="gray", lw=1, ls="--")
    ax.axvline(hi, color="gray", lw=1, ls="--",
               label=f"95% PPC interval [{lo:.0f}, {hi:.0f}]")
    p_value = float((totals < obs_total).mean())
    ax.set_title(
        f"Sign model aggregate PPC -- Bayesian p-value = {p_value:.3f}"
    )
    ax.set_xlabel("Total number of y_rep == 1 across observations")
    ax.set_ylabel("Posterior draws")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return p_value


def sign_per_transcript_plot(out_path, df, p_mean, y):
    df = df.copy()
    df["p_mean"] = p_mean
    df["y"] = y
    per_t = (
        df.groupby("transcript")
        .agg(n=("y", "size"), obs=("y", "mean"), pred=("p_mean", "mean"))
        .reset_index()
    )
    per_t["se"] = np.sqrt(per_t["obs"] * (1 - per_t["obs"]) / per_t["n"])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--",
            label="y = x")
    ax.errorbar(
        per_t["pred"], per_t["obs"], yerr=per_t["se"],
        fmt="o", color="#1f77b4", capsize=3,
    )
    for _, r in per_t.iterrows():
        ax.annotate(
            str(r["transcript"]),
            (r["pred"], r["obs"]),
            xytext=(4, 4), textcoords="offset points", fontsize=7,
            color="gray",
        )
    lo = min(per_t["pred"].min(), per_t["obs"].min()) - 0.05
    hi = max(per_t["pred"].max(), per_t["obs"].max()) + 0.05
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Per-transcript posterior-mean predicted p")
    ax.set_ylabel("Per-transcript observed positive rate (+/- SE)")
    ax.set_title("Sign model: predicted vs observed per transcript")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return per_t


def discrimination_metrics(p_mean, y):
    base = float(y.mean())
    # Brier.
    brier_model = float(np.mean((y - p_mean) ** 2))
    brier_base = float(np.mean((y - base) ** 2))
    # Log loss (clipped for numerical safety).
    eps = 1e-9
    pm = np.clip(p_mean, eps, 1 - eps)
    ll_model = float(
        -np.mean(y * np.log(pm) + (1 - y) * np.log(1 - pm))
    )
    ll_base = float(
        -np.mean(y * np.log(base) + (1 - y) * np.log(1 - base))
    )
    # Accuracy at 0.5.
    pred = (p_mean > 0.5).astype(int)
    acc = float((pred == y).mean())
    acc_base = max(base, 1 - base)
    # AUC: hand-rolled (no sklearn dep).
    pos = p_mean[y == 1]
    neg = p_mean[y == 0]
    auc_num = 0.0
    for v in pos:
        auc_num += float((neg < v).sum()) + 0.5 * float((neg == v).sum())
    auc = auc_num / (len(pos) * len(neg)) if len(pos) and len(neg) else float("nan")
    return {
        "base_rate": base,
        "auc": auc,
        "accuracy_at_0.5": acc,
        "accuracy_baseline": acc_base,
        "brier_model": brier_model,
        "brier_baseline": brier_base,
        "brier_skill": 1 - brier_model / brier_base,
        "log_loss_model": ll_model,
        "log_loss_baseline": ll_base,
        "log_loss_skill": 1 - ll_model / ll_base,
    }


# ---------------------------------------------------------------------------
# Per-outcome driver.
# ---------------------------------------------------------------------------
def run_continuous(outcome_label):
    print(f"\n=== {outcome_label.upper()} (continuous) ===")
    df = load_dataframe(outcome_label)
    y_col = OUTCOME_TO_COL[outcome_label]
    idata = az.from_netcdf(OUT_DIR / f"{outcome_label}-fit.nc")
    loo, mu_3d = run_loo(outcome_label, idata, df, y_col)

    # Posterior-mean mu per obs -> residuals.
    mu_mean = mu_3d.mean(axis=(0, 1))
    residuals = df[y_col].values - mu_mean

    sigma_mean = float(idata.posterior["sigma"].mean().item())
    nu_mean = (
        float(idata.posterior["nu"].mean().item())
        if "nu" in idata.posterior.data_vars else None
    )

    qq_path = OUT_DIR / f"qq-{outcome_label}.png"
    qq_plot(qq_path, outcome_label, residuals, sigma_mean, nu_mean)
    print(f"  wrote {qq_path.name}")

    pk_path = OUT_DIR / f"pareto-k-{outcome_label}.png"
    pareto_k_plot(pk_path, outcome_label, loo)
    print(f"  wrote {pk_path.name}")

    en_path = OUT_DIR / f"energy-{outcome_label}.png"
    bfmi = energy_plot(en_path, outcome_label, idata)
    print(f"  wrote {en_path.name}")

    k = np.asarray(loo.pareto_k.values)
    bad_idx = np.where(k > 0.7)[0]
    ok_idx = np.where((k > 0.5) & (k <= 0.7))[0]
    bad_obs = []
    if bad_idx.size:
        sub = df.iloc[bad_idx][["transcript", "question", "domain", y_col]].copy()
        sub["pareto_k"] = k[bad_idx]
        sub = sub.sort_values("pareto_k", ascending=False)
        bad_obs = sub.to_dict(orient="records")

    return {
        "outcome": outcome_label,
        "loo": loo,
        "n_high_k": int(bad_idx.size),
        "n_borderline_k": int(ok_idx.size),
        "k_max": float(k.max()),
        "k_median": float(np.median(k)),
        "bad_obs": bad_obs,
        "bfmi": bfmi,
    }


def run_sign():
    print(f"\n=== SECONDARY-SIGN (Bernoulli) ===")
    df = load_dataframe("secondary-sign")
    y = df["y_sign"].values.astype(int)
    idata = az.from_netcdf(OUT_DIR / "secondary-sign-fit.nc")
    loo, mu_3d = run_loo("secondary-sign", idata, df, "y_sign")

    # Per-obs posterior of p.
    n_chain, n_draw, n_obs = mu_3d.shape
    mu_2d = mu_3d.reshape(n_chain * n_draw, n_obs).T  # (n_obs, S)
    p_post_2d = 1.0 / (1.0 + np.exp(-mu_2d))
    p_mean = p_post_2d.mean(axis=1)

    # Plots.
    cal_path = OUT_DIR / "sign-calibration.png"
    cal_df = sign_calibration_plot(cal_path, p_mean, y)
    print(f"  wrote {cal_path.name}")

    ppc_path = OUT_DIR / "sign-ppc-aggregate.png"
    ppc_p = sign_ppc_aggregate_plot(ppc_path, p_post_2d, y)
    print(f"  wrote {ppc_path.name}")

    pt_path = OUT_DIR / "sign-per-transcript.png"
    per_t = sign_per_transcript_plot(pt_path, df, p_mean, y)
    print(f"  wrote {pt_path.name}")

    pk_path = OUT_DIR / "pareto-k-secondary-sign.png"
    pareto_k_plot(pk_path, "secondary-sign", loo)
    print(f"  wrote {pk_path.name}")

    en_path = OUT_DIR / "energy-secondary-sign.png"
    bfmi = energy_plot(en_path, "secondary-sign", idata)
    print(f"  wrote {en_path.name}")

    disc = discrimination_metrics(p_mean, y)
    k = np.asarray(loo.pareto_k.values)
    return {
        "outcome": "secondary-sign",
        "loo": loo,
        "n_high_k": int((k > 0.7).sum()),
        "n_borderline_k": int(((k > 0.5) & (k <= 0.7)).sum()),
        "k_max": float(k.max()),
        "k_median": float(np.median(k)),
        "discrimination": disc,
        "ppc_p_value": ppc_p,
        "calibration": cal_df,
        "per_transcript": per_t,
        "n_obs": int(len(y)),
        "n_dropped_ties": int(
            len(pd.read_csv(DATA, encoding="utf-8")) - len(y)
        ),
        "bfmi": bfmi,
    }


# ---------------------------------------------------------------------------
# Pick outcomes, run, write summary.
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        outcomes = sys.argv[1:]
        bad = [o for o in outcomes if o not in OUTCOME_TO_COL]
        if bad:
            sys.exit(f"Unknown outcome(s) {bad}. Choose from {sorted(OUTCOME_TO_COL)}.")
    else:
        outcomes = [
            o for o in OUTCOME_ORDER if (OUT_DIR / f"{o}-fit.nc").exists()
        ]
        if not outcomes:
            sys.exit(f"No *-fit.nc files in {OUT_DIR}.")

    print(f"Processing outcomes: {outcomes}")

    results = []
    sign_result = None
    for label in outcomes:
        if label in CONTINUOUS_OUTCOMES:
            results.append(run_continuous(label))
        elif label == "secondary-sign":
            sign_result = run_sign()
        else:
            print(f"  (skipping unknown outcome {label})")

    # az.compare for the secondary Normal vs Student-t pair.
    compare_table = None
    by_label = {r["outcome"]: r for r in results}
    if {"secondary", "secondary-t"}.issubset(by_label.keys()):
        # az.compare wants a dict mapping label -> idata (with log_likelihood
        # already attached, which run_loo did).
        # The ELPDData objects on `results` were produced from the idata
        # objects passed to az.loo; to compare via az.compare we re-load and
        # re-attach.
        compare_dict = {}
        for lbl in ("secondary", "secondary-t"):
            df = load_dataframe(lbl)
            idata = az.from_netcdf(OUT_DIR / f"{lbl}-fit.nc")
            run_loo(lbl, idata, df, OUTCOME_TO_COL[lbl])  # attaches log_lik
            compare_dict[lbl] = idata
        compare_table = az.compare(compare_dict, ic="loo", scale="log")
        compare_path = OUT_DIR / "loo-compare-secondary-vs-t.txt"
        compare_path.write_text(compare_table.to_string(), encoding="utf-8")
        print(f"\nWrote {compare_path.name}")

    # Build summary text.
    lines = ["Extended model-fit diagnostics", "=" * 36, ""]
    for r in results:
        loo = r["loo"]
        lines += [
            f"--- {r['outcome'].upper()} ---",
            f"  BFMI per chain    = {', '.join(f'{b:.3f}' for b in r['bfmi'])}"
            "   (> 0.3 = healthy energy mixing)",
            f"  PSIS-LOO elpd     = {loo.elpd_loo:.2f}  (se {loo.se:.2f})",
            f"  p_loo (eff. params) = {loo.p_loo:.2f}",
            f"  Pareto k median   = {r['k_median']:.3f}, max = {r['k_max']:.3f}",
            f"  # obs with k > 0.5 (borderline) = {r['n_borderline_k']}",
            f"  # obs with k > 0.7 (bad)        = {r['n_high_k']}",
        ]
        if r["bad_obs"]:
            lines.append("  High-k observations (k > 0.7):")
            for o in r["bad_obs"][:10]:
                y_col = OUTCOME_TO_COL[r["outcome"]]
                lines.append(
                    f"    transcript={o['transcript']:<3} "
                    f"question={o['question']:<24} "
                    f"domain={o['domain']:<12} "
                    f"y={o[y_col]:+7.3f}  k={o['pareto_k']:.3f}"
                )
            if len(r["bad_obs"]) > 10:
                lines.append(f"    ... ({len(r['bad_obs']) - 10} more)")
        lines.append("")

    if compare_table is not None:
        lines += [
            "--- LOO COMPARISON: secondary (Normal) vs secondary-t (Student-t) ---",
            compare_table.to_string(),
            "",
            "Interpretation: 'rank' 0 is the preferred model. elpd_diff is the",
            "expected log predictive density difference relative to the top model;",
            "dse is the standard error of that difference. |elpd_diff| > 2*dse is",
            "the conventional threshold for a meaningful preference.",
            "",
        ]

    if sign_result is not None:
        s = sign_result
        loo = s["loo"]
        d = s["discrimination"]
        lines += [
            f"--- SECONDARY-SIGN (Bernoulli) ---",
            f"  n_obs fitted       = {s['n_obs']}",
            f"  ties dropped       = {s['n_dropped_ties']}",
            f"  base rate (y=1)    = {d['base_rate']:.4f}",
            f"  BFMI per chain     = {', '.join(f'{b:.3f}' for b in s['bfmi'])}"
            "   (> 0.3 = healthy energy mixing)",
            "",
            "  PSIS-LOO:",
            f"    elpd            = {loo.elpd_loo:.2f}  (se {loo.se:.2f})",
            f"    p_loo           = {loo.p_loo:.2f}",
            f"    Pareto k median = {s['k_median']:.3f}, max = {s['k_max']:.3f}",
            f"    # obs k > 0.5   = {s['n_borderline_k']}",
            f"    # obs k > 0.7   = {s['n_high_k']}",
            "",
            "  Discrimination:",
            f"    AUC                = {d['auc']:.4f}",
            f"    Accuracy @ 0.5     = {d['accuracy_at_0.5']:.4f}   "
            f"(baseline {d['accuracy_baseline']:.4f})",
            f"    Brier score        = {d['brier_model']:.4f}   "
            f"(baseline {d['brier_baseline']:.4f}; "
            f"skill {d['brier_skill']:+.4f})",
            f"    Log loss           = {d['log_loss_model']:.4f}   "
            f"(baseline {d['log_loss_baseline']:.4f}; "
            f"skill {d['log_loss_skill']:+.4f})",
            "",
            f"  Aggregate PPC Bayesian p-value (total y_rep=1) = "
            f"{s['ppc_p_value']:.3f}   (0.5 = perfect)",
            "",
            "  Calibration (10 quantile bins of predicted p):",
        ]
        for _, row in s["calibration"].iterrows():
            lines.append(
                f"    pred_mean={row['pred_mean']:.3f}  "
                f"obs_rate={row['obs_rate']:.3f}  "
                f"n={int(row['n'])}"
            )
        lines.append("")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {SUMMARY_PATH.name}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
