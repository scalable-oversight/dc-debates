"""Model-fit diagnostics for the Beta-likelihood secondary fit.

Parallels ``model_fit_checks.py`` for the Normal / Student-t fits, adapted
for the Beta family (likelihood on p in (0,1) instead of y_secondary in R):

  1. Posterior predictive check, grouped by transcript. For each observation
     we reconstruct mu_i = inv_logit(eta_i) from posterior draws and sample
     y_rep_i ~ Beta(mu_i * kappa, (1 - mu_i) * kappa). The 24-panel PNG
     plots a histogram of pooled y_rep per transcript with vertical lines
     for the observed p values from that transcript. We do this twice --
     once on section-4 observations (post-debate, comparable to the primary
     PPC) and once on section-1 (pre-debate, sanity check).

  2. Randomised quantile residuals. Beta residuals on the (0,1) scale aren't
     directly comparable to the Normal/Student-t residuals; the standard
     replacement is randomised quantile residuals (Dunn & Smyth 1996), which
     map each observation through its fitted Beta CDF (one per posterior
     draw) and then through Phi^-1. If the model is well-specified, these
     residuals are marginally Normal(0, 1), so plotting them against fitted
     mu and against a N(0,1) overlay is the direct analog of the existing
     residuals-vs-fitted figure.

  3. Variance components + Beta precision. Per-component posterior summary
     for sigma_question, sigma_transcript, sigma_participant, plus kappa.
     Same "compressed at zero" diagnostic (P(sigma < 0.1)) as the other
     fits, so we can compare apples to apples.

  4. Side-by-side sigma_transcript comparison across all four fits
     (primary / secondary / secondary-t / secondary-beta) when each fit's
     .nc file is present. This is the headline plot for "did the Beta
     refit restore the sigma_transcript signal that Student-t collapsed?"

Outputs:
  output/ppc-secondary-beta.png             24-panel per-transcript PPC, section 4
  output/ppc-secondary-beta-section1.png    24-panel per-transcript PPC, section 1
  output/residuals-secondary-beta.png       randomised-quantile residual diagnostics
  output/sigmas-secondary-beta.png          sigma posteriors + kappa
  output/sigmas-transcript-comparison.png   sigma_transcript across all 4 fits
  output/model-fit-summary.txt              extended with a SECONDARY-BETA section

Usage:
  python3 model_fit_checks_beta.py

When invoked from ``run_all.sh`` the env vars ``ANALYSIS_DATA_CSV`` and
``ANALYSIS_OUT_DIR`` flow through; the beta-long CSV path is derived from
``ANALYSIS_DATA_CSV`` automatically.
"""

import os
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=,mode=NUMBA")

import re
import sys
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cleaned"
OUT_DIR = Path(
    os.environ.get(
        "ANALYSIS_OUT_DIR",
        str(Path(__file__).resolve().parent / "output"),
    )
)
OUT_DIR.mkdir(exist_ok=True)


def _default_beta_long_csv() -> Path:
    ar = os.environ.get("ANALYSIS_DATA_CSV")
    if ar:
        p = Path(ar)
        if "analysis-ready" in p.name:
            return p.parent / p.name.replace("analysis-ready", "beta-long")
    return DATA_DIR / "quant-1-official-study-beta-long.csv"


DATA = Path(os.environ.get("ANALYSIS_BETA_LONG_CSV", str(_default_beta_long_csv())))
FIT = OUT_DIR / "secondary-beta-fit.nc"
SUMMARY_PATH = OUT_DIR / "model-fit-summary.txt"

DOMAIN_VAR = "C(domain, Treatment('brainteasers'))"
DOMAIN_DIM = f"{DOMAIN_VAR}_dim"

RNG = np.random.default_rng(42)

# Subsample the posterior for the PPC and residual draws so the per-obs
# Beta CDF evaluation stays well under a minute on a 2564-row dataset.
PPC_SUBSAMPLE = 1000


def stack(post, name):
    return post[name].stack(sample=("chain", "draw")).values


def hdi95(vals: np.ndarray) -> tuple[float, float]:
    h = az.hdi(vals, hdi_prob=0.95)
    return float(h[0]), float(h[1])


def reconstruct_eta(idata: az.InferenceData, df: pd.DataFrame):
    """Return (eta, kappa) with eta shape (n_obs, S) on logit scale."""
    post = idata.posterior

    intercept = stack(post, "Intercept")
    b_order = stack(post, "order")
    b_section = stack(post, "section_is_4")
    b_dom = stack(post, DOMAIN_VAR)
    u_q = stack(post, "1|question")
    u_t = stack(post, "1|transcript")
    u_t_section = stack(post, "section_is_4|transcript")
    u_p = stack(post, "1|participant_id")
    kappa = stack(post, "kappa")

    p_q = [str(q) for q in post["question__factor_dim"].values]
    p_t = [str(t) for t in post["transcript__factor_dim"].values]
    p_pid = [str(p) for p in post["participant_id__factor_dim"].values]
    p_d = [str(d) for d in post[DOMAIN_DIM].values]

    obs_q = df["question"].astype(str).map(lambda q: p_q.index(q)).values
    obs_v = df["transcript"].astype(str).map(lambda t: p_t.index(t)).values
    obs_pid = df["participant_id"].astype(str).map(lambda x: p_pid.index(x)).values
    obs_order = df["order"].values.astype(float)
    obs_section = df["section_is_4"].values.astype(float)
    obs_d = df["domain"].values

    eta = (
        intercept[None, :]
        + obs_order[:, None] * b_order[None, :]
        + obs_section[:, None] * b_section[None, :]
        + u_q[obs_q, :]
        + u_t[obs_v, :]
        + obs_section[:, None] * u_t_section[obs_v, :]
        + u_p[obs_pid, :]
    )
    if "astrophysics" in p_d:
        mask = obs_d == "astrophysics"
        eta[mask] += b_dom[p_d.index("astrophysics"), :]
    if "coding" in p_d:
        mask = obs_d == "coding"
        eta[mask] += b_dom[p_d.index("coding"), :]
    return eta, kappa


def sample_beta_y_rep(eta: np.ndarray, kappa: np.ndarray, sub_idx: np.ndarray):
    """Sample posterior-predictive Beta draws on the subsample of posterior draws."""
    eta_sub = eta[:, sub_idx]
    kappa_sub = kappa[sub_idx]
    mu_sub = 1.0 / (1.0 + np.exp(-eta_sub))
    alpha = mu_sub * kappa_sub[None, :]
    beta = (1.0 - mu_sub) * kappa_sub[None, :]
    return RNG.beta(alpha, beta)


def randomised_quantile_residuals(eta: np.ndarray, kappa: np.ndarray, y: np.ndarray):
    """Map each observation through its fitted Beta CDF at a randomly chosen draw,
    then through Phi^-1. Under correct specification these are marginally N(0,1).
    """
    n_obs, S = eta.shape
    draw_idx = RNG.integers(0, S, size=n_obs)
    eta_at = eta[np.arange(n_obs), draw_idx]
    kappa_at = kappa[draw_idx]
    mu_at = 1.0 / (1.0 + np.exp(-eta_at))
    alpha = mu_at * kappa_at
    beta = (1.0 - mu_at) * kappa_at
    u = sstats.beta.cdf(y, alpha, beta)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return sstats.norm.ppf(u)


def plot_ppc_per_transcript(out_path: Path, label: str, df_obs: pd.DataFrame,
                            y_rep: np.ndarray, mask: np.ndarray) -> None:
    """6x4 grid of per-transcript histograms of pooled y_rep with observed rug ticks."""
    df_sub = df_obs.loc[mask].copy()
    y_rep_sub = y_rep[mask.values]

    transcripts = sorted(df_sub["transcript"].unique())
    assert len(transcripts) == 24, f"expected 24 transcripts, got {len(transcripts)}"
    q_lookup = (
        df_sub[["transcript", "question"]]
        .drop_duplicates().set_index("transcript")
    )
    transcripts = sorted(
        transcripts, key=lambda t: (q_lookup.loc[t, "question"], t)
    )

    fig, axes = plt.subplots(6, 4, figsize=(14, 16), sharex=True)
    for ax, tid in zip(axes.ravel(), transcripts):
        rows = (df_sub["transcript"] == tid).values
        pool = y_rep_sub[rows].ravel()
        observed = df_sub.loc[rows, "p"].values
        q = df_sub.loc[rows, "question"].iloc[0]
        n = int(rows.sum())
        ax.hist(pool, bins=60, density=True, range=(0.0, 1.0),
                color="#1f77b4", alpha=0.4, edgecolor="none")
        for y in observed:
            ax.axvline(y, color="black", alpha=0.4, lw=0.7)
        ax.set_title(f"t={tid} | {q} (n={n})", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0.0, 1.0)
        ax.set_yticks([])
    fig.suptitle(
        f"Beta PPC by transcript ({label}; bars = pooled y_rep, "
        f"vertical lines = observed p)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_qr_residuals(out_path: Path, mu_mean: np.ndarray, qr: np.ndarray,
                      domains: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette = {
        "brainteasers": "#d62728",
        "astrophysics": "#1f77b4",
        "coding": "#2ca02c",
    }
    colors = [palette[d] for d in domains]
    axes[0].scatter(mu_mean, qr, c=colors, alpha=0.5, s=12)
    axes[0].axhline(0, color="gray", lw=1)
    axes[0].set_xlabel("Posterior-mean fitted mu (probability)")
    axes[0].set_ylabel("Randomised quantile residual")
    axes[0].set_title("Randomised quantile residuals vs fitted (secondary-beta)")
    for d, c in palette.items():
        axes[0].scatter([], [], c=c, label=d, s=12)
    axes[0].legend(fontsize=8, loc="best")

    axes[1].hist(qr, bins=30, density=True, color="#888", alpha=0.7)
    xs = np.linspace(qr.min(), qr.max(), 200)
    axes[1].plot(
        xs, np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi),
        color="black", lw=1.5, label="N(0,1)",
    )
    axes[1].set_xlabel("Randomised quantile residual")
    axes[1].set_title("Residual marginal vs N(0,1)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sigmas(out_path: Path, idata: az.InferenceData) -> None:
    panels = [
        ("sigma_question",          stack(idata.posterior, "1|question_sigma")),
        ("sigma_transcript",        stack(idata.posterior, "1|transcript_sigma")),
        ("sigma_transcript_section", stack(idata.posterior, "section_is_4|transcript_sigma")),
        ("sigma_participant",       stack(idata.posterior, "1|participant_id_sigma")),
        ("kappa (Beta prec.)",      stack(idata.posterior, "kappa")),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (name, vals) in zip(axes, panels):
        ax.hist(vals, bins=50, density=True, color="#1f77b4", alpha=0.7)
        ax.axvline(0, color="gray", lw=1)
        ax.set_title(f"{name} (secondary-beta)", fontsize=10)
        ax.set_xlabel(name)
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sigma_transcript_comparison(out_path: Path,
                                     idata_beta: az.InferenceData) -> list[tuple[str, np.ndarray]]:
    """Overlay sigma_transcript posteriors across all four fits, when present."""
    candidates = [
        ("primary",        OUT_DIR / "primary-fit.nc"),
        ("secondary",      OUT_DIR / "secondary-fit.nc"),
        ("secondary-t",    OUT_DIR / "secondary-t-fit.nc"),
        ("secondary-beta", FIT),
    ]
    # For the Normal-residual and Student-t secondary fits, `1|transcript_sigma`
    # IS the per-transcript section-update SD (because y_secondary differences
    # the sections out by construction). For the Beta fit with the random slope
    # on section per transcript, the strict secondary-analog is
    # `section_is_4|transcript_sigma`. Use that for the Beta column so the
    # comparison is apples-to-apples on "how much per-transcript update SD does
    # each fit identify?".
    present: list[tuple[str, np.ndarray]] = []
    for label, path in candidates:
        if path == FIT:
            vals = stack(idata_beta.posterior, "section_is_4|transcript_sigma")
            present.append((label, vals))
            continue
        if not path.exists():
            continue
        idat = az.from_netcdf(path)
        vals = stack(idat.posterior, "1|transcript_sigma")
        present.append((label, vals))

    if len(present) < 2:
        return present

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, vals in present:
        ax.hist(vals, bins=50, density=True, alpha=0.4, label=label)
    ax.set_xlabel("sigma_transcript posterior")
    ax.set_title("sigma_transcript posterior across fits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return present


def merge_summary_block(new_block: list[str]) -> None:
    """Append the SECONDARY-BETA section to model-fit-summary.txt.

    If a previous SECONDARY-BETA block exists, replace it; if the summary
    file is missing entirely, create it with a minimal header.
    """
    section = "\n".join(new_block).rstrip() + "\n"
    if SUMMARY_PATH.exists():
        existing = SUMMARY_PATH.read_text(encoding="utf-8")
        if "--- SECONDARY-BETA" in existing:
            existing = re.sub(
                r"\n*--- SECONDARY-BETA.*?(?=\n--- |\Z)",
                "\n\n" + section,
                existing,
                flags=re.S,
            )
        else:
            existing = existing.rstrip() + "\n\n" + section
        SUMMARY_PATH.write_text(existing, encoding="utf-8")
    else:
        header = ["Model-fit diagnostics summary", "=" * 36, ""]
        SUMMARY_PATH.write_text("\n".join(header) + "\n" + section, encoding="utf-8")


def main() -> int:
    if not FIT.exists():
        sys.exit(
            f"ERROR: fit not found: {FIT}\n"
            f"Run fit_secondary_beta.py first."
        )
    if not DATA.exists():
        sys.exit(f"ERROR: beta-long CSV not found: {DATA}")

    print(f"Loading fit: {FIT.name}")
    idata = az.from_netcdf(FIT)
    df = pd.read_csv(DATA, encoding="utf-8")
    n_obs = len(df)

    eta, kappa = reconstruct_eta(idata, df)
    S = kappa.shape[0]
    sub_idx = RNG.choice(S, min(S, PPC_SUBSAMPLE), replace=False)
    print(
        f"  S={S} posterior draws total; using {sub_idx.size} for PPC + "
        f"residuals to keep the Beta-CDF pass under a minute."
    )
    y_rep = sample_beta_y_rep(eta, kappa, sub_idx)
    obs_y = df["p"].values

    # PPC plots: separately for section 1 and section 4 so the pile-ups in
    # each section's predictive distribution are visible side by side.
    s4_mask = df["section_is_4"] == 1
    s1_mask = ~s4_mask
    ppc4 = OUT_DIR / "ppc-secondary-beta.png"
    ppc1 = OUT_DIR / "ppc-secondary-beta-section1.png"
    plot_ppc_per_transcript(ppc4, "secondary-beta, section 4", df, y_rep, s4_mask)
    plot_ppc_per_transcript(ppc1, "secondary-beta, section 1", df, y_rep, s1_mask)
    print(f"  wrote {ppc4.name}, {ppc1.name}")

    # Randomised quantile residuals.
    qr = randomised_quantile_residuals(eta, kappa, obs_y)
    mu_mean = (1.0 / (1.0 + np.exp(-eta))).mean(axis=1)
    res_path = OUT_DIR / "residuals-secondary-beta.png"
    plot_qr_residuals(res_path, mu_mean, qr, df["domain"].values)
    print(f"  wrote {res_path.name}")

    # Sigma posteriors + side-by-side comparison.
    sig_path = OUT_DIR / "sigmas-secondary-beta.png"
    plot_sigmas(sig_path, idata)
    print(f"  wrote {sig_path.name}")

    cmp_path = OUT_DIR / "sigmas-transcript-comparison.png"
    comp = plot_sigma_transcript_comparison(cmp_path, idata)
    if cmp_path.exists():
        print(f"  wrote {cmp_path.name}")

    # PPC calibration: pool y_rep per transcript across both sections so the
    # 95% PPC interval is per-transcript (matching the existing fits' check).
    out_of = 0
    total = 0
    for tid in sorted(df["transcript"].unique()):
        rows = (df["transcript"] == tid).values
        pool = y_rep[rows].ravel()
        lo, hi = np.percentile(pool, [2.5, 97.5])
        out_of += int(((obs_y[rows] < lo) | (obs_y[rows] > hi)).sum())
        total += int(rows.sum())
    pct_out = 100.0 * out_of / total

    # Numerical summary block.
    qr_mean = float(qr.mean())
    qr_sd = float(qr.std(ddof=1))
    qr_skew = float(((qr - qr_mean) ** 3).mean() / qr_sd ** 3)
    qr_kurt = float(((qr - qr_mean) ** 4).mean() / qr_sd ** 4 - 3.0)

    lines = [
        f"--- SECONDARY-BETA  (n_obs = {n_obs}) ---",
        "",
        "Randomised quantile residuals (should be N(0,1) if model is well-specified):",
        f"  mean   = {qr_mean:+.3f}",
        f"  sd     = {qr_sd:.3f}",
        f"  min    = {qr.min():+.3f}",
        f"  max    = {qr.max():+.3f}",
        f"  skew   = {qr_skew:+.3f}",
        f"  excess kurt = {qr_kurt:+.3f}",
        "",
    ]
    for sname, var in [
        ("sigma_question",            "1|question_sigma"),
        ("sigma_transcript",          "1|transcript_sigma"),
        ("sigma_transcript_section",  "section_is_4|transcript_sigma"),
        ("sigma_participant",         "1|participant_id_sigma"),
    ]:
        vals = stack(idata.posterior, var)
        lo, hi = hdi95(vals)
        lines += [
            f"{sname} posterior (Beta logit scale):",
            f"  mean   = {vals.mean():.3f}",
            f"  median = {float(np.median(vals)):.3f}",
            f"  95% HDI= [{lo:.3f}, {hi:.3f}]",
            f"  P({sname} < 0.1) = {(vals < 0.1).mean():.3f}"
            f"   (high -> compressed at zero)",
            "",
        ]
    lo_k, hi_k = hdi95(kappa)
    lines += [
        "kappa (Beta precision) posterior:",
        f"  mean   = {kappa.mean():.2f}",
        f"  median = {float(np.median(kappa)):.2f}",
        f"  95% HDI= [{lo_k:.2f}, {hi_k:.2f}]",
        "",
        "Per-transcript PPC calibration (pooled section-1 + section-4 raw p):",
        f"  observations outside per-transcript 95% PPC interval: "
        f"{pct_out:.1f}% (reference ~5% under correct calibration)",
        "",
    ]
    if len(comp) > 1:
        lines.append("Side-by-side sigma_transcript posterior across fits:")
        lines.append(
            f"  {'fit':<16} {'mean':>7} {'HDI lo':>8} {'HDI hi':>8} {'P(<0.1)':>9}"
        )
        for label, vals in comp:
            lo, hi = hdi95(vals)
            lines.append(
                f"  {label:<16} {vals.mean():>7.3f} {lo:>8.3f} {hi:>8.3f} "
                f"{(vals < 0.1).mean():>9.3f}"
            )
        lines.append("")

    merge_summary_block(lines)
    print(f"\nUpdated {SUMMARY_PATH.name}")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
