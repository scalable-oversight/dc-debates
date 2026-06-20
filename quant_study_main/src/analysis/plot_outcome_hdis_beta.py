"""Forest plot of per-transcript expected outcomes under the Beta fit.

Beta analogue of plot_outcome_hdis.py. Two panels:

  Left (primary-like, average across sections):
    mu_v = Intercept + 0.5 * beta_order + 0.5 * beta_section
         + beta_domain[d(v)]
         + u_question[q(v)] + u_transcript[v] + 0.5 * u_transcript_sec[v]

  Right (secondary-like, per-transcript section-4 - section-1 update):
    delta_v = beta_section + u_transcript_sec[v]

The right panel is the direct Beta analogue of the secondary outcome
y_secondary = logit(p_4) - logit(p_1), per transcript. It is only
non-degenerate because we fit (0 + section_is_4 | transcript) in
fit_secondary_beta.py; with intercept-only transcript random effects the
per-transcript update would collapse to the global beta_section.

Both panels: one row per transcript, horizontal line from HDI_lo to HDI_hi
with a dot at the mean, coloured by domain, sorted by mean ascending so
the hardest transcripts (lowest expected outcome / smallest update) sit at
the top. A vertical dashed line at 0 marks the relevant reference point.

Output: <out_dir>/outcome-hdi-forest-beta.png
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


def compute_per_transcript_mu_avg(idata, lookup):
    """Posterior of mu_v averaged across order AND section."""
    post = idata.posterior
    intercept = post["Intercept"].stack(sample=("chain", "draw")).values
    b_order = post["order"].stack(sample=("chain", "draw")).values
    b_section = post["section_is_4"].stack(sample=("chain", "draw")).values
    b_dom = post[DOMAIN_VAR].stack(sample=("chain", "draw")).values
    u_q = post["1|question"].stack(sample=("chain", "draw")).values
    u_t = post["1|transcript"].stack(sample=("chain", "draw")).values
    u_t_sec = post["section_is_4|transcript"].stack(sample=("chain", "draw")).values
    n_draws = intercept.shape[0]

    p_q = [str(q) for q in post["question__factor_dim"].values]
    p_t = [str(t) for t in post["transcript__factor_dim"].values]
    p_d = [str(d) for d in post[DOMAIN_DIM].values]

    mu = np.empty((n_draws, len(lookup)), dtype=float)
    for vi, row in lookup.iterrows():
        v, q, d = str(row["transcript"]), row["question"], row["domain"]
        d_post = b_dom[p_d.index(d), :] if d in p_d else np.zeros(n_draws)
        mu[:, vi] = (
            intercept + 0.5 * b_order + 0.5 * b_section + d_post
            + u_q[p_q.index(q), :]
            + u_t[p_t.index(v), :]
            + 0.5 * u_t_sec[p_t.index(v), :]
        )
    return mu


def compute_per_transcript_section_update(idata, lookup):
    """Posterior of (beta_section + u_transcript_sec[v]) per transcript."""
    post = idata.posterior
    b_section = post["section_is_4"].stack(sample=("chain", "draw")).values
    u_t_sec = post["section_is_4|transcript"].stack(sample=("chain", "draw")).values
    n_draws = b_section.shape[0]

    p_t = [str(t) for t in post["transcript__factor_dim"].values]

    delta = np.empty((n_draws, len(lookup)), dtype=float)
    for vi, row in lookup.iterrows():
        v = str(row["transcript"])
        delta[:, vi] = b_section + u_t_sec[p_t.index(v), :]
    return delta


def summarize(arr, lookup, value_name):
    hdi = az.hdi(arr, hdi_prob=0.95)
    return pd.DataFrame({
        "transcript": lookup["transcript"].values,
        "question": lookup["question"].values,
        "domain": lookup["domain"].values,
        f"{value_name}_mean": arr.mean(axis=0),
        f"{value_name}_hdi_lo": hdi[:, 0],
        f"{value_name}_hdi_hi": hdi[:, 1],
    }).sort_values(f"{value_name}_mean").reset_index(drop=True)


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


def plot_forest(avg_df, delta_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharex=False)
    panels = [
        (
            axes[0], avg_df, "mu_avg",
            "Beta model, expected logit credence in\n"
            "honest's answer, averaged across sections\n"
            "(>0 = leans correct; hardest at top)",
            "mu  (higher = closer to correct, averaged across sections)",
        ),
        (
            axes[1], delta_df, "delta",
            "Beta model, per-transcript expected\n"
            "logit credence update from section 1 to section 4\n"
            "(>0 = updates toward correct; hardest at top)",
            "delta  (higher = greater update toward correct)",
        ),
    ]
    for ax, df, prefix, title, xlabel in panels:
        n = len(df)
        y = np.arange(n)
        colors = [DOMAIN_COLORS[d] for d in df["domain"]]
        for i in range(n):
            ax.plot(
                [df[f"{prefix}_hdi_lo"].iloc[i], df[f"{prefix}_hdi_hi"].iloc[i]],
                [y[i], y[i]],
                color=colors[i], lw=2.2,
                solid_capstyle="round", alpha=0.85,
            )
        ax.scatter(
            df[f"{prefix}_mean"], y, c=colors, s=36, zorder=3,
            edgecolor="white", linewidth=0.6,
        )
        ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([row_label(r) for _, r in df.iterrows()], fontsize=12)
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_title(title, fontsize=15)
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=12)
    legend_handles = [
        Line2D([0], [0], color=c, lw=2.6, label=d)
        for d, c in DOMAIN_COLORS.items()
    ]
    axes[1].legend(handles=legend_handles, loc="lower left",
                   fontsize=13, framealpha=0.95)
    fig.suptitle(
        "Per-transcript expected outcomes under the Beta fit\n"
        "(posterior mean + 95% HDI, Beta-logit scale)",
        fontsize=18,
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

idata = az.from_netcdf(OUT_DIR / "secondary-beta-fit.nc")
mu_avg = compute_per_transcript_mu_avg(idata, lookup)
delta = compute_per_transcript_section_update(idata, lookup)

plot_forest(
    summarize(mu_avg, lookup, "mu_avg"),
    summarize(delta, lookup, "delta"),
    OUT_DIR / "outcome-hdi-forest-beta.png",
)
