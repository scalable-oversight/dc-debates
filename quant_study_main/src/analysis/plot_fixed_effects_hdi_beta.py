"""Forest plot of fixed effects + variance components for the secondary-beta fit.

Reads output/secondary-beta-fit.nc and produces:

  output/fixed-effects-hdi-beta.png
    Single-panel figure showing, for each parameter, the posterior 95% HDI
    as a horizontal line with a dot at the posterior mean. Parameters cover
    the Beta-model linear predictor (Intercept, order, section_is_4, the two
    domain contrasts) plus all variance components (sigma_question,
    sigma_transcript, sigma_transcript_section, sigma_participant) and the
    Beta precision (kappa). All quantities except kappa are on the logit
    (mu) scale.

This is the Beta-model analogue of fixed-effects-hdi.png, which compares the
Normal-residual primary and secondary fits side by side.
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

# Display order (top -> bottom). Variance components and kappa at the bottom.
PARAMS = [
    ("Intercept", "Intercept"),
    ("order", "order"),
    ("section_is_4", "section_is_4"),
    ("astrophysics (vs brainteasers)", (DOMAIN_VAR, "astrophysics")),
    ("coding (vs brainteasers)", (DOMAIN_VAR, "coding")),
    ("sigma_question", "1|question_sigma"),
    ("sigma_transcript", "1|transcript_sigma"),
    ("sigma_transcript_section", "section_is_4|transcript_sigma"),
    ("sigma_participant", "1|participant_id_sigma"),
    ("kappa (precision)", "kappa"),
]


def extract(post, key):
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


post = az.from_netcdf(OUT_DIR / "secondary-beta-fit.nc").posterior
rows = summarize(post, PARAMS)

# kappa lives on a different scale (precision, positive) from the logit-mu
# parameters, so we keep it on the same plot but mark it visually. To do that
# we draw it on the same axis with a clear annotation; the x-limits are set
# from all parameters so kappa's larger magnitude expands the right edge.
n = len(rows)
y = np.arange(n)

fig, ax = plt.subplots(figsize=(10, 5.5))
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
ax.invert_yaxis()
ax.set_xlabel("Posterior mean + 95% HDI  (logit scale for all but kappa)")
ax.tick_params(axis="x", labelsize=8)
ax.set_title(
    "secondary-beta: fixed effects + variance components + kappa",
    fontsize=11,
)
fig.suptitle(
    "Beta-model fit: 95% HDIs",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out_path = OUT_DIR / "fixed-effects-hdi-beta.png"
fig.savefig(out_path, dpi=120)
plt.close(fig)
print(f"Wrote {out_path.name}")
