"""Single-panel version of the Beta forest plot: secondary panel only.

Companion to ``plot_outcome_hdis_beta.py``, which produces a two-panel
figure (averaged-across-sections mu_v on the left, per-transcript section
update delta_v on the right). This script produces only the right panel,
sized as a standalone figure for use where the averaged panel is not
needed.

Per-transcript posterior of:
  delta_v = beta_section + u_transcript_sec[v]

Output: <out_dir>/outcome-hdi-forest-beta-secondary-only.png
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

DOMAIN_COLORS = {
    "brainteasers": "#d62728",
    "astrophysics": "#1f77b4",
    "coding": "#2ca02c",
}

QUESTION_DISPLAY = {
    "ORM library": "ORM Library",
    "Database deletion": "DB Deletion",
    "Probability puzzle": "Probability",
    "Graviational Lens Modelling": "Lens Modelling",
    "Large Linear Structure": "Linear Structure",
    "Early Universe Galaxies": "EUGs",
    "Internal Temperature of Stars": "ITS",
}


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


def plot_single_panel(delta_df, within_q_idx, out_path):
    fig, ax = plt.subplots(figsize=(8, 10))
    n = len(delta_df)
    y = np.arange(n)
    colors = [DOMAIN_COLORS[d] for d in delta_df["domain"]]
    for i in range(n):
        ax.plot(
            [delta_df["delta_hdi_lo"].iloc[i], delta_df["delta_hdi_hi"].iloc[i]],
            [y[i], y[i]],
            color=colors[i], lw=2.2,
            solid_capstyle="round", alpha=0.85,
        )
    ax.scatter(
        delta_df["delta_mean"], y, c=colors, s=36, zorder=3,
        edgecolor="white", linewidth=0.6,
    )
    ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.7)
    ax.set_yticks(y)

    def row_label(row):
        q = QUESTION_DISPLAY.get(row["question"], row["question"])
        return f"{q} {within_q_idx[int(row['transcript'])]}"

    ax.set_yticklabels([row_label(r) for _, r in delta_df.iterrows()], fontsize=12)
    ax.set_xlabel(
        "delta  (higher = greater update toward correct)", fontsize=15
    )
    ax.set_title(
        "Beta model, per-transcript expected logit credence update\n"
        "from section 1 to section 4 (posterior mean + 95% HDI,\n"
        "Beta-logit scale; >0 = updates toward correct; hardest at top)",
        fontsize=14,
    )
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=12)

    legend_handles = [
        Line2D([0], [0], color=c, lw=2.6, label=d)
        for d, c in DOMAIN_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="lower left",
              fontsize=13, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path.name}")


def main():
    df = pd.read_csv(DATA, encoding="utf-8")
    lookup = (
        df[["transcript", "question", "domain"]]
        .drop_duplicates()
        .sort_values("transcript")
        .reset_index(drop=True)
    )

    _within_q = lookup.sort_values(["question", "transcript"]).copy()
    _within_q["within_q_idx"] = _within_q.groupby("question").cumcount() + 1
    within_q_idx = {
        int(t): int(i)
        for t, i in zip(_within_q["transcript"], _within_q["within_q_idx"])
    }

    idata = az.from_netcdf(OUT_DIR / "secondary-beta-fit.nc")
    delta = compute_per_transcript_section_update(idata, lookup)
    delta_df = summarize(delta, lookup, "delta")

    plot_single_panel(
        delta_df, within_q_idx,
        OUT_DIR / "outcome-hdi-forest-beta-secondary-only.png",
    )


if __name__ == "__main__":
    main()
