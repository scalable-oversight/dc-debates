"""Regenerate the four energy-transition PNGs from the saved fits.

For each fit, the full ArviZ InferenceData was saved to
``output/<label>-fit.nc`` at fit time, so we can reload and re-plot
without resampling. Matches the layout used by
``model_fit_checks_extra.energy_plot``.

Output: <out_dir>/energy-<label>.png for label in:
  primary, secondary, secondary-t, secondary-beta
"""

import os
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=,mode=NUMBA")

from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(
    os.environ.get("ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent / "output"))
)

LABELS = ["primary", "secondary", "secondary-t", "secondary-beta"]


def replot(label: str) -> None:
    idata = az.from_netcdf(OUT_DIR / f"{label}-fit.nc")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    az.plot_energy(idata, ax=ax)
    bfmi = az.bfmi(idata)
    ax.set_title(
        f"Energy plot ({label}) -- "
        f"BFMI per chain: {', '.join(f'{b:.2f}' for b in bfmi)}"
    )
    fig.tight_layout()
    out = OUT_DIR / f"energy-{label}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out.name}  (BFMI per chain: {[round(float(b), 3) for b in bfmi]})")


if __name__ == "__main__":
    for lbl in LABELS:
        replot(lbl)
