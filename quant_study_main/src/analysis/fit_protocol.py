"""Shared sampling protocol for the Bambi fit scripts.

Implements the preregistered MCMC procedure as translated for PyMC/Bambi.
The preregistration was written with brms/Stan in mind; the only deviations
needed are nominal:

  (i)  Stan's ``adapt_delta`` is PyMC's ``target_accept`` -- same concept,
       same default (0.8).
  (ii) "Confirm brms's non-centered parameterization is in effect for all
       varying-intercept terms" -> confirm Bambi's. Bambi defaults to
       ``Model(noncentered=True)`` and expands ``(1|g)`` as
       ``1|g = offset * sigma`` with ``offset ~ Normal(0, 1)`` for Normal
       group-level priors. We assert ``model.noncentered`` is True before
       each fit.

Divergence handling (preregistration, verbatim ladder):

  1. target_accept = 0.8 (default)
  2. if any divergences  -> target_accept = 0.95
  3. if divergences persist -> target_accept = 0.99
  4. if divergences persist:
     (a) raise max_treedepth from 10 to 12
     (b) confirm non-centered is in effect, and retain it
     (c) if divergences are concentrated in a variance component whose
         posterior is pressed against zero -- diagnosed here by the same
         P(sigma < 0.1) metric used in ``model_fit_checks.summarize_sigma``,
         with trigger ``P(sigma < 0.1) > 0.5`` -- tighten that component's
         HalfNormal(1) prior to HalfNormal(0.5) and refit.
  5. If divergences still persist, the script exits with a warning and the
     non-converged fit is saved for inspection.

NUTS diagnostics (reported on every fit):
  - mean acceptance probability per chain (should approximate target_accept)
  - tree-depth distribution per chain
  - E-BFMI per chain (required > 0.3)
  - energy plot saved to ``<out_dir>/energy-<label>.png``

Tree-depth saturation rule: if > 5% of post-warmup iterations saturate
max_treedepth (default 10), refit with max_treedepth = 12.

R-hat / ESS rule: if R-hat >= 1.01 or ESS bulk/tail < 400 anywhere, refit
with doubled draws.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TARGET_ACCEPT_LADDER = (0.8, 0.95, 0.99)
MAX_TREEDEPTH_DEFAULT = 10
MAX_TREEDEPTH_ESCALATED = 12
EBFMI_FLOOR = 0.3
TREEDEPTH_SAT_FRAC = 0.05
SIGMA_COMPRESSED_THRESHOLD = 0.1
SIGMA_COMPRESSED_PROB = 0.5
RHAT_CEILING = 1.01
ESS_FLOOR = 400


@dataclass
class ProtocolResult:
    idata: object
    summary: object
    target_accept: float
    max_treedepth: int
    tightened_sigmas: dict = field(default_factory=dict)
    convergence_ok: bool = False
    final_divergences: int = 0
    ebfmi_ok: bool = True


def _n_divergences(idata) -> int:
    return int(idata.sample_stats["diverging"].sum().item())


def _convergence(idata):
    summary = az.summary(idata, kind="diagnostics")
    rhat_ok = bool((summary["r_hat"] < RHAT_CEILING).all())
    bulk_ok = bool((summary["ess_bulk"] >= ESS_FLOOR).all())
    tail_ok = bool((summary["ess_tail"] >= ESS_FLOOR).all())
    return rhat_ok, bulk_ok, tail_ok, summary


def _fit(model, *, draws, tune, chains, seed, target_accept, max_treedepth=None):
    kwargs = dict(
        chains=chains,
        tune=tune,
        draws=draws,
        random_seed=seed,
        progressbar=False,
        target_accept=target_accept,
    )
    if max_treedepth is not None:
        kwargs["max_treedepth"] = max_treedepth
    t0 = time.time()
    idata = model.fit(**kwargs)
    print(f"  done in {time.time() - t0:.1f}s")
    return idata


def _assert_noncentered(model) -> None:
    # Step (b) of the preregistration: confirm non-centered parameterization
    # for varying-intercept terms is in effect. Bambi defaults to True; this
    # asserts it hasn't been disabled at the Model() call site.
    if not getattr(model, "noncentered", False):
        raise RuntimeError(
            "Bambi Model has noncentered=False; the preregistration requires "
            "non-centered parameterization for varying-intercept terms."
        )


def _report_nuts(idata, max_treedepth_used: int) -> dict:
    ss = idata.sample_stats
    out: dict = {}

    if "acceptance_rate" in ss:
        per_chain_acc = ss["acceptance_rate"].mean(dim="draw").values
        out["acceptance_per_chain"] = [float(a) for a in per_chain_acc]
        print(
            "  mean acceptance / chain: "
            + ", ".join(f"{a:.3f}" for a in per_chain_acc)
        )

    if "tree_depth" in ss:
        td = ss["tree_depth"].values
        sat_frac = float((td >= max_treedepth_used).mean())
        out["treedepth_saturation_frac"] = sat_frac
        depth_counts = np.bincount(td.flatten().astype(int))
        depth_str = " ".join(
            f"{d}:{c}" for d, c in enumerate(depth_counts) if c > 0
        )
        print(
            f"  tree depth (max={max_treedepth_used}): "
            f"saturated={sat_frac:.2%}  counts: {depth_str}"
        )
    else:
        out["treedepth_saturation_frac"] = None

    bfmi = np.asarray(az.bfmi(idata))
    out["ebfmi_per_chain"] = [float(b) for b in bfmi]
    out["ebfmi_ok"] = bool((bfmi >= EBFMI_FLOOR).all())
    print(
        "  E-BFMI / chain: "
        + ", ".join(f"{b:.3f}" for b in bfmi)
        + f"  (all >= {EBFMI_FLOOR}: {out['ebfmi_ok']})"
    )
    return out


def _save_energy_plot(idata, out_dir: Path, label: str) -> Path:
    ax = az.plot_energy(idata)
    fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
    path = out_dir / f"energy-{label}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def _sigma_compression(idata, sigma_var_names) -> dict:
    out = {}
    for name in sigma_var_names:
        if name in idata.posterior.data_vars:
            vals = idata.posterior[name].values.flatten()
            out[name] = float((vals < SIGMA_COMPRESSED_THRESHOLD).mean())
    return out


def run_protocol(
    *,
    label: str,
    build_model: Callable,
    sigma_var_names,
    chains: int,
    tune: int,
    draws: int,
    seed: int,
    out_dir: Path,
) -> ProtocolResult:
    """Run the full preregistered fit protocol.

    Parameters
    ----------
    label
        Short identifier used in printed output and the energy-plot filename
        (e.g. ``"primary"``, ``"secondary-beta"``).
    build_model
        Callable ``build_model(tighten_sigmas: dict[str, float] | None) -> bmb.Model``.
        When ``tighten_sigmas`` is given, the returned model must use those
        ``HalfNormal(sigma=value)`` priors for the named varying-intercept
        sigmas instead of the default ``HalfNormal(1)``. Used by preregistration
        step (c).
    sigma_var_names
        Posterior variable names of the varying-intercept SDs that step (c)
        may tighten (e.g. ``["1|question_sigma", "1|transcript_sigma"]``).
        Names of the corresponding Bambi prior keys are derived by stripping
        the ``_sigma`` suffix.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    model = build_model(tighten_sigmas=None)
    _assert_noncentered(model)

    target_accept = TARGET_ACCEPT_LADDER[0]
    max_treedepth = None

    print(
        f"\nInitial fit (chains={chains}, tune={tune}, draws={draws}, "
        f"target_accept={target_accept})..."
    )
    idata = _fit(
        model,
        draws=draws, tune=tune, chains=chains, seed=seed,
        target_accept=target_accept,
    )
    nd = _n_divergences(idata)
    print(f"  divergent transitions: {nd}")

    for next_ta in TARGET_ACCEPT_LADDER[1:]:
        if nd == 0:
            break
        target_accept = next_ta
        print(
            f"\nEscalating divergences: refitting at "
            f"target_accept={target_accept}..."
        )
        idata = _fit(
            model,
            draws=draws, tune=tune, chains=chains, seed=seed,
            target_accept=target_accept,
        )
        nd = _n_divergences(idata)
        print(f"  divergent transitions: {nd}")

    tightened: dict = {}
    if nd > 0:
        # Step (a): bump max_treedepth.
        max_treedepth = MAX_TREEDEPTH_ESCALATED
        print(
            f"\nDivergences persist: refitting at "
            f"target_accept={target_accept}, max_treedepth={max_treedepth} "
            f"(preregistration step (a))..."
        )
        idata = _fit(
            model,
            draws=draws, tune=tune, chains=chains, seed=seed,
            target_accept=target_accept, max_treedepth=max_treedepth,
        )
        nd = _n_divergences(idata)
        print(f"  divergent transitions: {nd}")

    if nd > 0:
        # Step (c): identify variance components compressed against zero
        # and tighten the corresponding HalfNormal(1) prior to HalfNormal(0.5).
        comp = _sigma_compression(idata, sigma_var_names)
        if comp:
            print(
                "\n  sigma compression P(sigma<{thr}): ".format(
                    thr=SIGMA_COMPRESSED_THRESHOLD
                )
                + ", ".join(f"{k}={v:.3f}" for k, v in comp.items())
            )
        to_tighten_posterior_names = [
            k for k, v in comp.items() if v > SIGMA_COMPRESSED_PROB
        ]
        if to_tighten_posterior_names:
            # ``"1|question_sigma"`` (posterior name) corresponds to the
            # Bambi prior key ``"1|question"``.
            tightened = {
                name[: -len("_sigma")]: 0.5
                for name in to_tighten_posterior_names
            }
            print(
                f"  preregistration step (c): tightening HalfNormal(1)->"
                f"HalfNormal(0.5) for {list(tightened.keys())}; rebuilding model."
            )
            model = build_model(tighten_sigmas=tightened)
            _assert_noncentered(model)
            idata = _fit(
                model,
                draws=draws, tune=tune, chains=chains, seed=seed,
                target_accept=target_accept, max_treedepth=max_treedepth,
            )
            nd = _n_divergences(idata)
            print(f"  divergent transitions: {nd}")
        else:
            print(
                "  no varying-intercept sigma compressed against zero; "
                "preregistration step (c) does not apply."
            )

    if nd > 0:
        print(
            "\nWARNING: divergent transitions remain after the full "
            "preregistered escalation ladder."
        )

    # NUTS-specific diagnostics on the current fit.
    print("\nNUTS diagnostics:")
    nuts = _report_nuts(idata, max_treedepth or MAX_TREEDEPTH_DEFAULT)
    energy_path = _save_energy_plot(idata, out_dir, label)
    print(f"  energy plot -> {energy_path.name}")

    # Tree-depth saturation rule: only triggers when we haven't already
    # bumped max_treedepth in the divergence cascade.
    if (
        max_treedepth is None
        and nuts["treedepth_saturation_frac"] is not None
        and nuts["treedepth_saturation_frac"] > TREEDEPTH_SAT_FRAC
    ):
        max_treedepth = MAX_TREEDEPTH_ESCALATED
        print(
            f"\n>{TREEDEPTH_SAT_FRAC:.0%} of iterations saturate max_treedepth "
            f"(observed {nuts['treedepth_saturation_frac']:.2%}); "
            f"refitting at max_treedepth={max_treedepth}..."
        )
        idata = _fit(
            model,
            draws=draws, tune=tune, chains=chains, seed=seed,
            target_accept=target_accept, max_treedepth=max_treedepth,
        )
        nd = _n_divergences(idata)
        print(f"  divergent transitions: {nd}")
        print("\nNUTS diagnostics after treedepth refit:")
        nuts = _report_nuts(idata, max_treedepth)
        energy_path = _save_energy_plot(idata, out_dir, label)
        print(f"  energy plot -> {energy_path.name}")

    rhat_ok, bulk_ok, tail_ok, summary = _convergence(idata)
    print(
        f"\nDiagnostics -- R-hat<{RHAT_CEILING}: {rhat_ok}  "
        f"ESS_bulk>={ESS_FLOOR}: {bulk_ok}  ESS_tail>={ESS_FLOOR}: {tail_ok}"
    )

    if not (rhat_ok and bulk_ok and tail_ok):
        print(f"\nConvergence failed; refitting with draws={draws * 2}...")
        idata = _fit(
            model,
            draws=draws * 2, tune=tune, chains=chains, seed=seed,
            target_accept=target_accept, max_treedepth=max_treedepth,
        )
        rhat_ok, bulk_ok, tail_ok, summary = _convergence(idata)
        print(
            f"\nRefit -- R-hat<{RHAT_CEILING}: {rhat_ok}  "
            f"ESS_bulk>={ESS_FLOOR}: {bulk_ok}  ESS_tail>={ESS_FLOOR}: {tail_ok}"
        )
        nd = _n_divergences(idata)
        print("\nNUTS diagnostics after draws refit:")
        nuts = _report_nuts(idata, max_treedepth or MAX_TREEDEPTH_DEFAULT)
        energy_path = _save_energy_plot(idata, out_dir, label)
        print(f"  energy plot -> {energy_path.name}")

    convergence_ok = bool(
        rhat_ok and bulk_ok and tail_ok and nuts["ebfmi_ok"] and nd == 0
    )

    return ProtocolResult(
        idata=idata,
        summary=summary,
        target_accept=target_accept,
        max_treedepth=max_treedepth or MAX_TREEDEPTH_DEFAULT,
        tightened_sigmas=tightened,
        convergence_ok=convergence_ok,
        final_divergences=nd,
        ebfmi_ok=nuts["ebfmi_ok"],
    )
