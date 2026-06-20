"""Combined model-fit diagnostics for the primary and secondary fits.

Runs the three checks listed under 'Model-fit checks' in CF_analysis_plan.docx:

  1. Posterior predictive checks comparing simulated final logit credences to
     observed values, grouped by transcript. (Per-transcript KDEs of pooled
     posterior-predictive draws with observed values as rug ticks.)

  2. Residuals vs fitted plot at the participant level.

  3. Check that the posterior of sigma_transcript is not compressed against
     zero -- numerical summary plus the joint posterior of all three sigmas.
     For Student-t fits, the sigma plot adds a fourth panel showing the
     posterior of the Student-t degrees-of-freedom parameter nu, and the text
     summary reports nu's posterior mean / 95% HDI and the probability mass
     below nu=30 (a rough "still meaningfully t-shaped" reference).

Outputs (per outcome: primary / secondary / secondary-t):
  data/ppc-<outcome>.png            24-panel per-transcript PPC
  data/residuals-<outcome>.png      residuals vs fitted scatter
  data/sigmas-<outcome>.png         sigma_question / sigma_transcript /
                                    sigma_residual posteriors (+ nu for
                                    Student-t fits)
  data/model-fit-summary.txt        numerical summary of all checks

Usage:
  python3 model_fit_checks.py
  python3 model_fit_checks.py primary secondary secondary-t

When called with no arguments, auto-discovers which *-fit.nc files are present
in OUT_DIR and processes them in the canonical order primary, secondary,
secondary-t. When called with explicit arguments, processes only those.
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

# Analysis-ready CSV in ../data/; fits + diagnostic outputs in ./output/.
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cleaned"
OUT_DIR = Path(os.environ.get("ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent / "output")))
OUT_DIR.mkdir(exist_ok=True)
DATA = Path(os.environ.get("ANALYSIS_DATA_CSV", str(DATA_DIR / "quant-1-official-study-analysis-ready.csv")))
SUMMARY_PATH = OUT_DIR / "model-fit-summary.txt"

DOMAIN_VAR = "C(domain, Treatment('brainteasers'))"
DOMAIN_DIM = f"{DOMAIN_VAR}_dim"

# Pilot estimates for context (CF_analysis_plan.docx 'Sample size rationale').
PILOT_SIGMA_TRANSCRIPT = 0.65
PILOT_SIGMA_RESIDUAL = 1.08

# Map outcome label -> column name in the analysis-ready CSV.
# secondary-t shares the y_secondary column; only its likelihood differs.
OUTCOME_TO_COL = {
    "primary": "y_primary",
    "secondary": "y_secondary",
    "secondary-t": "y_secondary",
}
# Canonical processing order when no CLI args are passed.
OUTCOME_ORDER = ["primary", "secondary", "secondary-t"]

RNG = np.random.default_rng(42)

df_all = pd.read_csv(DATA, encoding="utf-8")


def compute_posterior_predictive(idata, outcome_col):
    """Reconstruct mu_i for each observation, then sample y_rep ~ N(mu_i, sigma).

    Returns (mu_mean, residuals, y_rep_pooled_per_transcript, sigma_post),
    where y_rep_pooled is a dict transcript -> 1D array of pooled draws over
    all observations and posterior draws (for plotting per-transcript KDEs),
    and sigma_post is the posterior of the residual SD.
    """
    post = idata.posterior

    # Posterior arrays, stacked over (chain, draw) -> sample axis.
    intercept = post["Intercept"].stack(sample=("chain", "draw")).values
    b_order = post["order"].stack(sample=("chain", "draw")).values
    b_dom = post[DOMAIN_VAR].stack(sample=("chain", "draw")).values  # (2, S)
    u_q = post["1|question"].stack(sample=("chain", "draw")).values  # (12, S)
    u_t = post["1|transcript"].stack(sample=("chain", "draw")).values  # (24, S)
    sigma = post["sigma"].stack(sample=("chain", "draw")).values  # (S,)
    n_draws = sigma.shape[0]

    # Student-t fits also have a `nu` (degrees of freedom) parameter. When
    # present, posterior-predictive draws must come from a t(nu, mu, sigma),
    # not Normal(mu, sigma) -- otherwise the PPC calibration check
    # under-represents the tail mass the model itself expects.
    nu = (
        post["nu"].stack(sample=("chain", "draw")).values
        if "nu" in post.data_vars
        else None
    )

    p_q = [str(q) for q in post["question__factor_dim"].values]
    p_t = [str(t) for t in post["transcript__factor_dim"].values]
    p_d = [str(d) for d in post[DOMAIN_DIM].values]

    df = df_all.copy()
    obs_q = df["question"].map(lambda q: p_q.index(q)).values
    obs_v = df["transcript"].map(lambda t: p_t.index(str(t))).values
    obs_order = df["order"].values.astype(float)
    obs_y = df[outcome_col].values
    obs_d = df["domain"].values

    # mu shape: (n_obs, n_draws).
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
    # brainteasers is the reference; no addition needed.

    # Sample one posterior predictive y per (obs, draw).
    # For Student-t fits, eps is a standard t draw with per-draw nu; for
    # Normal fits, a standard Normal draw. (RNG.standard_t broadcasts
    # df=(1, n_draws) against size=(n_obs, n_draws) so each obs at draw s uses
    # the matching nu[s].)
    if nu is not None:
        eps = RNG.standard_t(df=nu[None, :], size=mu.shape)
    else:
        eps = RNG.standard_normal(mu.shape)
    y_rep = mu + sigma[None, :] * eps

    mu_mean = mu.mean(axis=1)
    residuals = obs_y - mu_mean

    # Pool y_rep across all observations and draws within each transcript.
    y_rep_per_transcript = {}
    for tid in sorted(df["transcript"].unique()):
        rows = (df["transcript"] == tid).values
        y_rep_per_transcript[int(tid)] = y_rep[rows].ravel()

    return {
        "mu_mean": mu_mean,
        "residuals": residuals,
        "y_obs": obs_y,
        "y_rep_per_transcript": y_rep_per_transcript,
        "sigma_residual": sigma,
        "df": df,
    }


def plot_ppc_per_transcript(out_path, outcome_label, pp, df_obs, outcome_col):
    """6x4 grid: per-transcript KDE of pooled y_rep with observed rug ticks."""
    transcripts = sorted(df_obs["transcript"].unique())
    assert len(transcripts) == 24
    # Order by question so paired transcripts sit next to each other.
    q_lookup = (
        df_obs[["transcript", "question"]].drop_duplicates().set_index("transcript")
    )
    transcripts = sorted(
        transcripts,
        key=lambda t: (q_lookup.loc[t, "question"], t),
    )

    fig, axes = plt.subplots(6, 4, figsize=(14, 16), sharex=True)
    for ax, tid in zip(axes.ravel(), transcripts):
        y_rep = pp["y_rep_per_transcript"][int(tid)]
        rows = df_obs["transcript"] == tid
        y_obs = df_obs.loc[rows, outcome_col].values
        q = df_obs.loc[rows, "question"].iloc[0]
        n = rows.sum()
        # KDE via numpy histogram for speed (8000 draws * ~8 obs each).
        # range=(-7, 7) matches xlim below: for Student-t fits, rare extreme
        # tail draws would otherwise stretch the auto-range to ~[-400, 400]
        # and leave a single ~13-logit-wide bin covering the visible window.
        ax.hist(y_rep, bins=60, density=True, range=(-7, 7), color="#1f77b4",
                alpha=0.4, edgecolor="none", label="y_rep")
        for y in y_obs:
            ax.axvline(y, color="black", alpha=0.7, lw=1.0)
        ax.set_title(f"t={tid} | {q} (n={n})", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-7, 7)
        ax.set_yticks([])
    fig.suptitle(
        f"Posterior predictive check by transcript ({outcome_label}; "
        f"bars = pooled y_rep, vertical lines = observed)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_residuals(out_path, outcome_label, pp, df_obs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    domains = df_obs["domain"].values
    palette = {
        "brainteasers": "#d62728",
        "astrophysics": "#1f77b4",
        "coding": "#2ca02c",
    }
    colors = [palette[d] for d in domains]
    axes[0].scatter(pp["mu_mean"], pp["residuals"], c=colors, alpha=0.6, s=18)
    axes[0].axhline(0, color="gray", lw=1)
    axes[0].set_xlabel("Fitted mu_i (posterior mean)")
    axes[0].set_ylabel("Residual y_i - mu_i")
    axes[0].set_title(f"Residuals vs fitted ({outcome_label})")
    for d, c in palette.items():
        axes[0].scatter([], [], c=c, label=d, s=18)
    axes[0].legend(fontsize=8, loc="best")

    # Residual histogram + normal-fit overlay.
    res = pp["residuals"]
    axes[1].hist(res, bins=30, density=True, color="#888", alpha=0.7)
    xs = np.linspace(res.min(), res.max(), 200)
    sd = res.std(ddof=1)
    axes[1].plot(
        xs, np.exp(-0.5 * (xs / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
        color="black", lw=1.5, label=f"N(0, {sd:.2f})",
    )
    axes[1].set_xlabel("Residual")
    axes[1].set_title(f"Residual marginal ({outcome_label})")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sigmas(out_path, outcome_label, idata):
    # Student-t fits have an extra `nu` parameter; show it as a 4th panel.
    has_nu = "nu" in idata.posterior.data_vars
    var_names = ["1|question_sigma", "1|transcript_sigma", "sigma"]
    titles = ["sigma_question", "sigma_transcript", "sigma_residual"]
    pilots = [None, None, None]
    if has_nu:
        var_names.append("nu")
        titles.append("nu (Student-t df)")
        pilots.append(None)

    n_panels = len(var_names)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    for ax, vn, title, pilot in zip(axes, var_names, titles, pilots):
        vals = idata.posterior[vn].stack(sample=("chain", "draw")).values
        ax.hist(vals, bins=50, density=True, color="#1f77b4", alpha=0.7)
        ax.axvline(0, color="gray", lw=1)
        if pilot is not None:
            ax.axvline(pilot, color="red", lw=1.5, ls="--",
                       label=f"pilot ~ {pilot}")
            ax.legend(fontsize=8)
        ax.set_title(f"{title} ({outcome_label})", fontsize=10)
        ax.set_xlabel(title)
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def summarize_sigma(idata, var_name, pilot=None):
    """Posterior summary of a non-negative scale parameter.

    Returns mean, median, 95% HDI, P(sigma < 0.1) -- the "compressed at zero"
    diagnostic -- and P(sigma > pilot) when a pilot estimate is supplied.
    """
    vals = idata.posterior[var_name].stack(sample=("chain", "draw")).values
    hdi = az.hdi(vals, hdi_prob=0.95)
    out = {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "hdi_lo": float(hdi[0]),
        "hdi_hi": float(hdi[1]),
        "p_below_0.1": float((vals < 0.1).mean()),
    }
    if pilot is not None:
        out["pilot"] = float(pilot)
        out["p_above_pilot"] = float((vals > pilot).mean())
    return out


def summarize_nu(idata):
    """Posterior summary of the Student-t df parameter; None for Normal fits."""
    if "nu" not in idata.posterior.data_vars:
        return None
    vals = idata.posterior["nu"].stack(sample=("chain", "draw")).values
    hdi = az.hdi(vals, hdi_prob=0.95)
    # nu>30 is a common rule-of-thumb "indistinguishable from Normal" threshold;
    # report mass below it as "evidence that Student-t is helping."
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "hdi_lo": float(hdi[0]),
        "hdi_hi": float(hdi[1]),
        "p_below_30": float((vals < 30).mean()),
    }


def diagnostics_for(outcome_label, outcome_col):
    fit = OUT_DIR / f"{outcome_label}-fit.nc"
    print(f"\n=== {outcome_label.upper()} ===")
    print(f"Loading {fit.name}...")
    idata = az.from_netcdf(fit)
    pp = compute_posterior_predictive(idata, outcome_col)

    ppc_path = OUT_DIR / f"ppc-{outcome_label}.png"
    res_path = OUT_DIR / f"residuals-{outcome_label}.png"
    sig_path = OUT_DIR / f"sigmas-{outcome_label}.png"

    plot_ppc_per_transcript(ppc_path, outcome_label, pp, pp["df"], outcome_col)
    plot_residuals(res_path, outcome_label, pp, pp["df"])
    plot_sigmas(sig_path, outcome_label, idata)
    print(f"  wrote {ppc_path.name}, {res_path.name}, {sig_path.name}")

    res = pp["residuals"]
    res_summary = {
        "n": int(res.size),
        "mean": float(res.mean()),
        "sd": float(res.std(ddof=1)),
        "min": float(res.min()),
        "max": float(res.max()),
        "skew": float(((res - res.mean()) ** 3).mean() / res.std() ** 3),
        "kurt_excess": float(
            ((res - res.mean()) ** 4).mean() / res.std() ** 4 - 3.0
        ),
    }
    sigmas = {
        "sigma_question": summarize_sigma(idata, "1|question_sigma"),
        "sigma_transcript": summarize_sigma(
            idata, "1|transcript_sigma", pilot=PILOT_SIGMA_TRANSCRIPT
        ),
        "sigma_residual": summarize_sigma(
            idata, "sigma", pilot=PILOT_SIGMA_RESIDUAL
        ),
    }

    # PPC quick numerical check: are observed y values within the posterior
    # predictive support? Report the fraction of observations whose value
    # falls outside the 95% PPC interval (a rough calibration check).
    y_obs = pp["y_obs"]
    # Recompute mu+sigma summary per-observation predictive intervals.
    # For efficiency, derive from y_rep_per_transcript indirectly: build
    # per-observation 95% intervals on the fly.
    # Simpler: compute fraction observed in the 95% interval of y_rep_per_obs
    # by sampling. We use y_rep_per_transcript already pooled, which gives a
    # transcript-level interval; that's the right granularity for the plan's
    # 'grouped by transcript' framing.
    out_of_interval = 0
    total_obs = 0
    for tid, y_rep_pool in pp["y_rep_per_transcript"].items():
        rows = pp["df"]["transcript"] == tid
        observed = pp["df"].loc[rows, outcome_col].values
        lo, hi = np.percentile(y_rep_pool, [2.5, 97.5])
        out_of_interval += int(((observed < lo) | (observed > hi)).sum())
        total_obs += observed.size
    pct_out = 100.0 * out_of_interval / total_obs

    return {
        "outcome": outcome_label,
        "residuals": res_summary,
        "sigmas": sigmas,
        "nu": summarize_nu(idata),
        "ppc_out_of_95pct_pct": pct_out,
        "n_obs": total_obs,
    }


# ---------------------------------------------------------------------------
# Pick which fits to process.
#   * no CLI args  -> auto-discover *-fit.nc files in OUT_DIR
#   * CLI args     -> only those outcomes (in the order given)
# ---------------------------------------------------------------------------
if len(sys.argv) > 1:
    requested = sys.argv[1:]
    bad = [o for o in requested if o not in OUTCOME_TO_COL]
    if bad:
        sys.exit(
            f"Unknown outcome(s) {bad}. "
            f"Choose from {sorted(OUTCOME_TO_COL)}."
        )
    outcomes = requested
else:
    outcomes = [
        o for o in OUTCOME_ORDER if (OUT_DIR / f"{o}-fit.nc").exists()
    ]
    if not outcomes:
        sys.exit(
            f"No *-fit.nc files found in {OUT_DIR}. "
            f"Run fit_primary_model.py / fit_secondary_model.py first."
        )

print(f"Processing outcomes: {outcomes}")

results = [
    diagnostics_for(label, OUTCOME_TO_COL[label]) for label in outcomes
]

lines = ["Model-fit diagnostics summary", "=" * 36, ""]
for r in results:
    lines += [
        f"--- {r['outcome'].upper()}  (n_obs = {r['n_obs']}) ---",
        "",
        "Residuals (y - posterior-mean mu):",
        f"  mean   = {r['residuals']['mean']:+.3f}",
        f"  sd     = {r['residuals']['sd']:.3f}",
        f"  min    = {r['residuals']['min']:+.3f}",
        f"  max    = {r['residuals']['max']:+.3f}",
        f"  skew   = {r['residuals']['skew']:+.3f}",
        f"  excess kurt = {r['residuals']['kurt_excess']:+.3f}",
        "",
    ]
    for sname, s in r["sigmas"].items():
        lines += [
            f"{sname} posterior:",
            f"  mean   = {s['mean']:.3f}",
            f"  median = {s['median']:.3f}",
            f"  95% HDI= [{s['hdi_lo']:.3f}, {s['hdi_hi']:.3f}]",
            f"  P({sname} < 0.1) = {s['p_below_0.1']:.3f}"
            f"   (high -> compressed at zero)",
        ]
        if "pilot" in s:
            lines.append(
                f"  P({sname} > {s['pilot']} [pilot]) = {s['p_above_pilot']:.3f}"
            )
        lines.append("")
    if r["nu"] is not None:
        lines += [
            "nu (Student-t degrees of freedom) posterior:",
            f"  mean   = {r['nu']['mean']:.2f}",
            f"  median = {r['nu']['median']:.2f}",
            f"  95% HDI= [{r['nu']['hdi_lo']:.2f}, {r['nu']['hdi_hi']:.2f}]",
            f"  P(nu < 30) = {r['nu']['p_below_30']:.3f}"
            f"   (high -> residuals are meaningfully fatter-tailed than Normal)",
            "",
        ]
    lines += [
        "Per-transcript PPC calibration:",
        f"  observations outside per-transcript 95% PPC interval: "
        f"{r['ppc_out_of_95pct_pct']:.1f}% "
        f"(reference ~5% under correct calibration)",
        "",
    ]

SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"\nWrote {SUMMARY_PATH.name}")
print()
print("\n".join(lines))
