"""Forest plot of per-transcript P(update toward correct) + 95% HDI for the
exploratory Bernoulli sign model (parallels plot_outcome_hdis.py).

For each transcript v, computes the posterior of

    logit(p_v) = Intercept + 0.5 * beta_order
               + beta_domain[d(v)]
               + u_question[q(v)]
               + u_transcript[v]

then transforms to p_v = sigmoid(logit(p_v)) for plotting on the probability
scale. The 0.5 * beta_order averages over the two order conditions, so the
per-transcript number is the expected probability that a counterfactual
judge facing that transcript with order randomised would update toward the
honest answer.

Single panel: one row per transcript, horizontal line from HDI_lo to HDI_hi
with a dot at the mean, coloured by domain, sorted by mean p ascending so
the transcripts that least often push the judge toward correct sit at the
top. Vertical dashed line at p = 0.5.

Output: output/outcome-hdi-forest-sign.png
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_per_transcript_p(idata, lookup):
    """Posterior of P(toward correct) for each transcript, averaged across order."""
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

    logit = np.empty((n_draws, len(lookup)), dtype=float)
    for vi, row in lookup.iterrows():
        v, q, d = str(row["transcript"]), row["question"], row["domain"]
        d_post = b_dom[p_d.index(d), :] if d in p_d else np.zeros(n_draws)
        logit[:, vi] = (
            intercept + 0.5 * b_order + d_post
            + u_q[p_q.index(q), :] + u_t[p_t.index(v), :]
        )
    return sigmoid(logit)


def summarize(p, lookup):
    hdi = az.hdi(p, hdi_prob=0.95)
    return pd.DataFrame({
        "transcript": lookup["transcript"].values,
        "question": lookup["question"].values,
        "domain": lookup["domain"].values,
        "p_mean": p.mean(axis=0),
        "p_hdi_lo": hdi[:, 0],
        "p_hdi_hi": hdi[:, 1],
    }).sort_values("p_mean").reset_index(drop=True)


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


def plot_forest(df, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(9, 10))
    n = len(df)
    y = np.arange(n)
    colors = [DOMAIN_COLORS[d] for d in df["domain"]]
    for i in range(n):
        ax.plot(
            [df["p_hdi_lo"].iloc[i], df["p_hdi_hi"].iloc[i]],
            [y[i], y[i]],
            color=colors[i], lw=2.2,
            solid_capstyle="round", alpha=0.85,
        )
    ax.scatter(
        df["p_mean"], y, c=colors, s=36, zorder=3,
        edgecolor="white", linewidth=0.6,
    )
    ax.axvline(0.5, color="gray", lw=1, ls="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([row_label(r) for _, r in df.iterrows()], fontsize=12)
    ax.set_xlabel("P(update toward correct)", fontsize=15)
    ax.set_title(
        "secondary-sign: per-transcript P(toward correct) + 95% HDI\n"
        "(lowest at top; dashed line at P = 0.5)",
        fontsize=15,
    )
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=12)
    legend_handles = [
        Line2D([0], [0], color=c, lw=2.6, label=d)
        for d, c in DOMAIN_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="lower left",
              fontsize=13, framealpha=0.95)
    fig.suptitle(
        "Exploratory Bernoulli model (sign of credence change)", fontsize=17,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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

p = compute_per_transcript_p(
    az.from_netcdf(OUT_DIR / "secondary-sign-fit.nc"), lookup
)
plot_forest(summarize(p, lookup), OUT_DIR / "outcome-hdi-forest-sign.png")
