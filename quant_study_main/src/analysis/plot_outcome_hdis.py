"""Forest plot of per-transcript expected raw outcome (mu_v) + 95% HDI.

For each transcript v, computes the posterior of

    mu_v = Intercept + 0.5 * beta_order
         + beta_domain[d(v)]
         + u_question[q(v)]
         + u_transcript[v]

(The 0.5 * beta_order averages over the two order conditions, so the
per-transcript number is the expected logit credence under a counterfactual
judge facing that transcript with order randomised.)

The plot has two panels (primary | secondary). Within each panel: one row
per transcript, a horizontal line from HDI_lo to HDI_hi with a dot at the
mean, coloured by domain. Sorted by mean mu ascending so the hardest
transcripts (lowest expected outcome) sit at the top -- matching the layout
of tier-hdi-forest.png. A vertical dashed line at mu = 0 marks where
participants on average end at 50% credence in the honest answer (primary)
or do not update at all from part 1 to part 4 (secondary).

Output: output/outcome-hdi-forest.png
"""

import os
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=,mode=NUMBA")

from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cleaned"
OUT_DIR = Path(os.environ.get("ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent / "output")))
DATA = Path(os.environ.get("ANALYSIS_DATA_CSV", str(DATA_DIR / "quant-1-official-study-analysis-ready.csv")))

DOMAIN_VAR = "C(domain, Treatment('brainteasers'))"
DOMAIN_DIM = f"{DOMAIN_VAR}_dim"

DOMAIN_COLORS = {
    "brainteasers": "#d62728",
    "astrophysics": "#1f77b4",
    "coding": "#2ca02c",
}


def compute_per_transcript_mu(idata, lookup):
    """Posterior of mu_v for each transcript, averaged across order."""
    post = idata.posterior
    intercept = post["Intercept"].stack(sample=("chain", "draw")).values
    b_order = post["order"].stack(sample=("chain", "draw")).values
    b_dom = post[DOMAIN_VAR].stack(sample=("chain", "draw")).values
    u_q = post["1|question"].stack(sample=("chain", "draw")).values
    u_t = post["1|transcript"].stack(sample=("chain", "draw")).values
    n_draws = intercept.shape[0]

    p_q = [str(q) for q in post["question__factor_dim"].values]
    p_t = [str(t) for t in post["transcript__factor_dim"].values]
    p_d = [str(d) for d in post[DOMAIN_DIM].values]

    mu = np.empty((n_draws, len(lookup)), dtype=float)
    for vi, row in lookup.iterrows():
        v, q, d = str(row["transcript"]), row["question"], row["domain"]
        d_post = b_dom[p_d.index(d), :] if d in p_d else np.zeros(n_draws)
        mu[:, vi] = (
            intercept + 0.5 * b_order + d_post
            + u_q[p_q.index(q), :] + u_t[p_t.index(v), :]
        )
    return mu


def summarize(mu, lookup):
    hdi = az.hdi(mu, hdi_prob=0.95)
    return pd.DataFrame({
        "transcript": lookup["transcript"].values,
        "question": lookup["question"].values,
        "domain": lookup["domain"].values,
        "mu_mean": mu.mean(axis=0),
        "mu_hdi_lo": hdi[:, 0],
        "mu_hdi_hi": hdi[:, 1],
    }).sort_values("mu_mean").reset_index(drop=True)


QUESTION_DISPLAY = {
    "ORM library": "ORM Library",
    "Database deletion": "DB Deletion",
    "Probability puzzle": "Probability",
    "Graviational Lens Modelling": "Lens Modelling",
    "Large Linear Structure": "Linear Structure",
    "Early Universe Galaxies": "EUGs",
    "Internal Temperature of Stars": "ITS",
}

WITHIN_Q_IDX = {}


def row_label(row):
    q = QUESTION_DISPLAY.get(row["question"], row["question"])
    return f"{q} {WITHIN_Q_IDX[int(row['transcript'])]}"


def plot_forest(prim, sec, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharex=False)
    titles = [
        ("Primary outcome, normal model\nExpected end-of-debate logit credence in honest's answer\n"
         "(>0 = leans correct; hardest at top)",
         "mu  (higher = closer to correct)"),
        ("Secondary outcome, normal model\nExpected logit credence change (part 4 - part 1)\n"
         "(>0 = updates toward correct; hardest at top)",
         "mu  (higher = greater update toward correct)"),
    ]
    for ax, df, (title, xlabel) in zip(axes, [prim, sec], titles):
        n = len(df)
        y = np.arange(n)
        colors = [DOMAIN_COLORS[d] for d in df["domain"]]
        for i in range(n):
            ax.plot(
                [df["mu_hdi_lo"].iloc[i], df["mu_hdi_hi"].iloc[i]],
                [y[i], y[i]],
                color=colors[i], lw=2.2,
                solid_capstyle="round", alpha=0.85,
            )
        ax.scatter(
            df["mu_mean"], y, c=colors, s=36, zorder=3,
            edgecolor="white", linewidth=0.6,
        )
        ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([row_label(r) for _, r in df.iterrows()], fontsize=12)
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_title(title, fontsize=15)
        # Hardest at top: smallest mu (most negative) at top of the plot.
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=12)
    legend_handles = [
        Line2D([0], [0], color=c, lw=2.6, label=d)
        for d, c in DOMAIN_COLORS.items()
    ]
    axes[1].legend(handles=legend_handles, loc="lower left",
                   fontsize=13, framealpha=0.95)
    fig.suptitle(
        "Per-transcript expected raw outcome (posterior mean + 95% HDI)",
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path.name}")


df = pd.read_csv(DATA, encoding="utf-8")
lookup = (
    df[["transcript", "question", "domain"]]
    .drop_duplicates()
    .sort_values("transcript")
    .reset_index(drop=True)
)

_within_q = lookup.sort_values(["question", "transcript"]).copy()
_within_q["within_q_idx"] = _within_q.groupby("question").cumcount() + 1
WITHIN_Q_IDX.update(
    {int(t): int(i) for t, i in zip(_within_q["transcript"], _within_q["within_q_idx"])}
)

mu_prim = compute_per_transcript_mu(
    az.from_netcdf(OUT_DIR / "primary-fit.nc"), lookup
)
mu_sec = compute_per_transcript_mu(
    az.from_netcdf(OUT_DIR / "secondary-fit.nc"), lookup
)

plot_forest(
    summarize(mu_prim, lookup),
    summarize(mu_sec, lookup),
    OUT_DIR / "outcome-hdi-forest.png",
)
