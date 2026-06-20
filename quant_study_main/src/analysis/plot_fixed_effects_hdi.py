"""Forest plot of fixed effects + variance components for primary vs secondary.

Reads output/primary-fit.nc and output/secondary-fit.nc and produces:

  output/fixed-effects-hdi.png
    Two-panel figure: primary (top) and secondary (bottom). For each model,
    a horizontal forest line per parameter showing the posterior 95% HDI,
    with a dot at the posterior mean. The same parameter ordering is used
    on both panels so they can be compared by eye. Parameters shown match
    the "Fixed effects + variance components" table emitted by the fit
    scripts: residual sigma, intercept, order, the two domain contrasts
    (vs the brainteasers reference), sigma_question, sigma_transcript.
"""

import os
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(os.environ.get(
    "ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent / "output")
))

DOMAIN_VAR = "C(domain, Treatment('brainteasers'))"
DOMAIN_DIM = f"{DOMAIN_VAR}_dim"

# Display order (top -> bottom). Variance components grouped at the bottom.
PARAMS = [
    ("Intercept", "Intercept"),
    ("order", "order"),
    ("astrophysics (vs brainteasers)", (DOMAIN_VAR, "astrophysics")),
    ("coding (vs brainteasers)", (DOMAIN_VAR, "coding")),
    ("sigma_question", "1|question_sigma"),
    ("sigma_transcript", "1|transcript_sigma"),
    ("sigma_residual", "sigma"),
]


def extract(post, key):
    """Return a 1D array of posterior draws for the requested parameter."""
    if isinstance(key, tuple):
        var, level = key
        dim_levels = [str(x) for x in post[f"{var}_dim"].values]
        idx = dim_levels.index(level)
        return post[var].isel({f"{var}_dim": idx}).stack(
            sample=("chain", "draw")
        ).values
    return post[key].stack(sample=("chain", "draw")).values


def summarize(post, params):
    rows = []
    for label, key in params:
        vals = extract(post, key)
        hdi = az.hdi(vals, hdi_prob=0.95)
        rows.append({
            "label": label,
            "mean": float(vals.mean()),
            "hdi_lo": float(hdi[0]),
            "hdi_hi": float(hdi[1]),
        })
    return rows


def plot_panel(ax, rows, title, x_min, x_max):
    n = len(rows)
    y = np.arange(n)
    for i, r in enumerate(rows):
        ax.plot(
            [r["hdi_lo"], r["hdi_hi"]], [y[i], y[i]],
            color="#1f77b4", lw=2.4, solid_capstyle="round", alpha=0.85,
        )
    ax.scatter(
        [r["mean"] for r in rows], y,
        color="#1f77b4", s=44, zorder=3, edgecolor="white", linewidth=0.6,
    )
    ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Posterior mean + 95% HDI")
    ax.tick_params(axis="x", labelsize=8)


primary = az.from_netcdf(OUT_DIR / "primary-fit.nc").posterior
secondary = az.from_netcdf(OUT_DIR / "secondary-fit.nc").posterior

prim_rows = summarize(primary, PARAMS)
sec_rows = summarize(secondary, PARAMS)

# Shared x-limits so the two panels are directly comparable by eye.
all_los = [r["hdi_lo"] for r in prim_rows + sec_rows]
all_his = [r["hdi_hi"] for r in prim_rows + sec_rows]
pad = 0.05 * (max(all_his) - min(all_los) or 1.0)
x_min = min(all_los) - pad
x_max = max(all_his) + pad

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
plot_panel(axes[0], prim_rows, "primary", x_min, x_max)
plot_panel(axes[1], sec_rows, "secondary", x_min, x_max)
fig.suptitle(
    "Fixed effects + variance components: 95% HDIs",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_path = OUT_DIR / "fixed-effects-hdi.png"
fig.savefig(out_path, dpi=120)
plt.close(fig)
print(f"Wrote {out_path.name}")
