"""End-to-end sample-size planning with proper pilot-uncertainty propagation.

Pipeline
--------
1. Load pilot xlsx and compute end-state (logit s4) and update (logit s4 - logit s1) DVs.
2. For the novice subset (proxy for the AI-tasker pool):
   a. Fit the planned hierarchical model via a self-contained REML estimator.
   b. Parametric-bootstrap the variance components from this fit.
3. For each bootstrap draw, derive tau_marginal / tau_variant_only / sigma and run
   a small between-subjects ranking-recovery simulation.
4. Aggregate across bootstrap draws and produce a publication-ready summary table.
5. Also report the fixed-scenario simulation that the legacy script would produce.

Outputs (under analysis_dir, using --out as a prefix):
  <out>.summary.txt         human-readable headline table
  <out>.full_results.csv    all metrics at all (N, draw) combinations
  <out>.bootstrap_draws.csv the bootstrap (tau, sigma) distribution
  <out>.fits.txt            raw variance-components fits + bootstrap summary

Usage:
    python -m src.analysis.sample_size_with_uncertainty
    python -m src.analysis.sample_size_with_uncertainty --n-boot 500
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kendalltau

from ._io import ANALYSIS_DIR, INPUT_XLSX, ensure_analysis_dir

DEFAULT_OUT_PREFIX = ANALYSIS_DIR / "sample_size_with_uncertainty"
DEFAULT_N_GRID = [20, 25, 30, 40, 50, 60, 75, 100]

DOMAIN_MAP = {
    "bergs": "physics",
    "switch": "physics",
    "probability_-_expected_number_of_rolls": "physics",
    "early_universe_galaxies": "astro",
    "gravitational_lens_modelling": "astro",
    "internal_temperature_of_stars": "astro",
    "little_red_dots": "astro",
    "large_linear_structure": "astro",
    "malt-public_306440_(rust)": "coding",
    "malt-public_323067_(db_deletion)": "coding",
    "malt-public_338159_(orm_library)": "coding",
    "malt-public_339242_(prefix-sum)": "coding",
}

OUTCOMES = [
    ("end_state", "logit_section_4_credence_in_correct_answer", "End-state (logit s4)"),
    ("update",    "_update",                                   "Update (logit s4 - logit s1)"),
]


def load_pilot(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    needed = {
        "debate_name", "debate_turn_4_link", "participant_id",
        "order", "group",
        "logit_section_1_credence_in_correct_answer",
        "logit_section_4_credence_in_correct_answer",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["variant"] = df["debate_turn_4_link"]
    df["question"] = df["debate_name"]
    df["domain"] = df["debate_name"].map(DOMAIN_MAP).fillna("unknown")
    df["_update"] = (
        df["logit_section_4_credence_in_correct_answer"]
        - df["logit_section_1_credence_in_correct_answer"]
    )
    df["order_coded"] = (df["order"] == "Honest first").astype(int)
    df["participant_id"] = df["participant_id"].astype(str)
    return df


@dataclass
class DesignInfo:
    X: np.ndarray
    Z_list: list[np.ndarray]
    K_list: list[np.ndarray]
    re_names: list[str]
    fe_names: list[str]
    n: int
    re_sizes: list[int]


def build_design(df: pd.DataFrame) -> DesignInfo:
    n = len(df)
    intercept = np.ones(n)
    order = df["order_coded"].to_numpy(dtype=float)
    is_astro = (df["domain"] == "astro").astype(float).to_numpy()
    is_coding = (df["domain"] == "coding").astype(float).to_numpy()
    X = np.column_stack([intercept, order, is_astro, is_coding])
    fe_names = ["intercept", "order", "I(astro)", "I(coding)"]

    Z_list, K_list, re_names, re_sizes = [], [], [], []
    for col, name in [("question", "question"),
                      ("variant", "variant"),
                      ("participant_id", "participant")]:
        levels = df[col].unique()
        idx_map = {lvl: i for i, lvl in enumerate(levels)}
        Z = np.zeros((n, len(levels)))
        for i, lvl in enumerate(df[col].to_numpy()):
            Z[i, idx_map[lvl]] = 1.0
        Z_list.append(Z)
        K_list.append(Z @ Z.T)
        re_names.append(name)
        re_sizes.append(len(levels))

    return DesignInfo(X=X, Z_list=Z_list, K_list=K_list,
                      re_names=re_names, fe_names=fe_names,
                      n=n, re_sizes=re_sizes)


def reml_neg_loglik(
    log_sds: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    K_list: list[np.ndarray],
) -> float:
    sds = np.exp(log_sds)
    n = len(y)
    V = sds[-1] ** 2 * np.eye(n)
    for k, K in enumerate(K_list):
        V = V + sds[k] ** 2 * K
    try:
        L = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        return 1e10
    logdetV = 2.0 * np.log(np.diag(L)).sum()
    Vinv_y = np.linalg.solve(L.T, np.linalg.solve(L, y))
    Vinv_X = np.linalg.solve(L.T, np.linalg.solve(L, X))
    XtVinvX = X.T @ Vinv_X
    try:
        L2 = np.linalg.cholesky(XtVinvX)
    except np.linalg.LinAlgError:
        return 1e10
    logdetXtVinvX = 2.0 * np.log(np.diag(L2)).sum()
    b = np.linalg.solve(L2.T, np.linalg.solve(L2, X.T @ Vinv_y))
    resid = y - X @ b
    quad = float(resid @ np.linalg.solve(L.T, np.linalg.solve(L, resid)))
    return 0.5 * (logdetV + logdetXtVinvX + quad)


def fit_reml(
    y: np.ndarray,
    design: DesignInfo,
    starts: list[np.ndarray] | None = None,
) -> dict:
    n_re = len(design.K_list)
    if starts is None:
        starts = [
            np.log(np.full(n_re + 1, 0.5)),
            np.log(np.full(n_re + 1, 1.0)),
            np.log(np.full(n_re + 1, 0.3)),
            np.concatenate([np.log([0.5] * n_re), [np.log(1.0)]]),
        ]

    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x0 in starts:
            res = minimize(
                reml_neg_loglik, x0, args=(y, design.X, design.K_list),
                method="Nelder-Mead",
                options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 5000},
            )
            if best is None or res.fun < best.fun:
                best = res

    sds = np.exp(best.x)
    var = sds ** 2

    V = var[-1] * np.eye(design.n)
    for k, K in enumerate(design.K_list):
        V = V + var[k] * K
    L = np.linalg.cholesky(V)
    Vinv_y = np.linalg.solve(L.T, np.linalg.solve(L, y))
    Vinv_X = np.linalg.solve(L.T, np.linalg.solve(L, design.X))
    b = np.linalg.solve(design.X.T @ Vinv_X, design.X.T @ Vinv_y)

    out = {"_neg_loglik": float(best.fun), "_converged": bool(best.success)}
    for name, sd_val in zip(design.re_names, sds[:-1]):
        out[f"sd_{name}"] = float(sd_val)
        out[f"var_{name}"] = float(sd_val ** 2)
    out["sd_residual"] = float(sds[-1])
    out["var_residual"] = float(var[-1])
    for name, val in zip(design.fe_names, b):
        out[f"beta_{name}"] = float(val)
    return out


def parametric_bootstrap(
    fit: dict,
    design: DesignInfo,
    n_boot: int,
    seed: int,
    progress_every: int = 50,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fe_vec = np.array([fit[f"beta_{n}"] for n in design.fe_names])
    Xb = design.X @ fe_vec

    n_re = len(design.K_list)
    re_sds = [fit[f"sd_{name}"] for name in design.re_names]
    sd_resid = fit["sd_residual"]

    warm_start = np.log(np.maximum(re_sds + [sd_resid], 1e-3))

    rows = []
    n_failed = 0
    t0 = time.time()
    for b in range(n_boot):
        u_list = [rng.normal(0, sd, design.re_sizes[k])
                  for k, sd in enumerate(re_sds)]
        eps = rng.normal(0, sd_resid, design.n)
        y_sim = Xb + sum(design.Z_list[k] @ u_list[k] for k in range(n_re)) + eps
        try:
            r = fit_reml(y_sim, design, starts=[warm_start])
        except Exception:
            n_failed += 1
            continue
        var_q = r.get("var_question", 0.0)
        var_v = r.get("var_variant", 0.0)
        var_p = r.get("var_participant", 0.0)
        var_r = r["var_residual"]
        r["tau_marginal"] = float(np.sqrt(var_q + var_v))
        r["tau_variant_only"] = float(np.sqrt(var_v))
        r["sigma"] = float(np.sqrt(var_p + var_r))
        r["_iter"] = b
        rows.append(r)

        if (b + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"    ...bootstrap {b + 1}/{n_boot} done "
                  f"({elapsed:.1f}s, {n_failed} failed)", flush=True)

    if n_failed:
        print(f"    NOTE: {n_failed}/{n_boot} bootstrap fits failed and were dropped.")

    return pd.DataFrame(rows)


def simulate_ranking_recovery(
    tau: float,
    sigma: float,
    n_per: int,
    n_debates: int,
    n_inner: int,
    rng: np.random.Generator,
) -> dict:
    se = sigma / np.sqrt(n_per) if sigma > 0 else 0.0
    kend = np.empty(n_inner)
    fc = np.empty(n_inner)
    tier_agree = {k: np.empty(n_inner) for k in (3, 4, 5)}

    for i in range(n_inner):
        truth = rng.normal(0.0, max(tau, 1e-9), n_debates)
        est = truth + rng.normal(0.0, se, n_debates)
        t, _ = kendalltau(truth, est)
        kend[i] = t
        dt = truth[:, None] - truth[None, :]
        de = est[:, None] - est[None, :]
        mask = np.triu(np.ones_like(dt, dtype=bool), k=1)
        fc[i] = (np.sign(dt[mask]) == np.sign(de[mask])).mean()
        for k in (3, 4, 5):
            bins = np.linspace(0, 1, k + 1)[1:-1]
            tt = np.digitize(truth, np.quantile(truth, bins))
            et = np.digitize(est, np.quantile(est, bins))
            tier_agree[k][i] = (tt == et).mean()

    return {
        "kendall": float(np.mean(kend)),
        "frac_correct": float(np.mean(fc)),
        "tier_3_agree": float(np.mean(tier_agree[3])),
        "tier_4_agree": float(np.mean(tier_agree[4])),
        "tier_5_agree": float(np.mean(tier_agree[5])),
    }


def marginal_simulation(
    boot: pd.DataFrame,
    n_grid: list[int],
    n_debates: int,
    n_inner: int,
    seed: int,
    tau_col: str = "tau_marginal",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_grid:
        for i, row in boot.iterrows():
            tau = row[tau_col]
            sigma = row["sigma"]
            m = simulate_ranking_recovery(
                tau=tau, sigma=sigma, n_per=n,
                n_debates=n_debates, n_inner=n_inner, rng=rng,
            )
            m.update({"n_per": n, "draw": int(row["_iter"]),
                      "tau": tau, "sigma": sigma})
            rows.append(m)
    return pd.DataFrame(rows)


def summarise_marginal(
    sim: pd.DataFrame,
    metric: str,
    label: str,
) -> str:
    lines = [
        f"\n{label}",
        f"{'N/cell':>7}  {'mean':>6}  {'median':>7}  {'5%-95%':>15}  "
        f"{'P(>=60%)':>9}  {'P(>=70%)':>9}  {'P(>=80%)':>9}",
        "-" * 76,
    ]
    for n, sub in sim.groupby("n_per"):
        v = sub[metric].to_numpy() * 100
        lines.append(
            f"{n:>7}  {v.mean():>5.1f}%  {np.median(v):>6.1f}%  "
            f"[{np.percentile(v, 5):>4.1f}%, {np.percentile(v, 95):>4.1f}%]  "
            f"{(v >= 60).mean():>8.2f}   {(v >= 70).mean():>8.2f}   "
            f"{(v >= 80).mean():>8.2f}"
        )
    return "\n".join(lines)


def summarise_fit(fit: dict, design: DesignInfo, label: str) -> str:
    lines = [f"\n--- {label} ---"]
    lines.append(f"  Fixed effects (GLS at optimum):")
    for name in design.fe_names:
        lines.append(f"    beta_{name:<12} = {fit[f'beta_{name}']:+.3f}")
    lines.append(f"  Variance components:")
    total = fit["var_residual"] + sum(fit[f"var_{n}"] for n in design.re_names)
    for name in design.re_names:
        sd = fit[f"sd_{name}"]
        v = fit[f"var_{name}"]
        share = 100 * v / total if total > 0 else 0.0
        lines.append(f"    SD({name:<11}) = {sd:5.3f}   var share = {share:5.1f}%")
    lines.append(f"    SD(residual)   = {fit['sd_residual']:5.3f}   "
                 f"var share = {100 * fit['var_residual'] / total:5.1f}%")
    var_q = fit.get("var_question", 0.0)
    var_v = fit.get("var_variant", 0.0)
    var_p = fit.get("var_participant", 0.0)
    var_r = fit["var_residual"]
    tau_marg = float(np.sqrt(var_q + var_v))
    tau_var  = float(np.sqrt(var_v))
    sigma    = float(np.sqrt(var_p + var_r))
    lines.append(f"  Planning quantities:")
    lines.append(f"    tau_marginal      = sqrt(var_q + var_v) = {tau_marg:.3f}")
    lines.append(f"    tau_variant_only  = sqrt(var_v)         = {tau_var:.3f}")
    lines.append(f"    sigma             = sqrt(var_p + var_r) = {sigma:.3f}")
    lines.append(f"    tau_marginal / sigma = {tau_marg / sigma:.3f}  "
                 f"(SNR per single judge)")
    return "\n".join(lines)


def summarise_bootstrap(boot: pd.DataFrame, label: str) -> str:
    cols = [c for c in ["sd_question", "sd_variant", "sd_participant",
                         "sd_residual", "tau_marginal", "tau_variant_only", "sigma"]
            if c in boot.columns]
    lines = [f"\n--- Bootstrap distribution of variance components ({label}, n={len(boot)} draws) ---"]
    lines.append(f"{'parameter':<22} {'mean':>7} {'SE':>7}  "
                 f"{'5%':>7} {'50%':>7} {'95%':>7}")
    lines.append("-" * 65)
    for c in cols:
        v = boot[c].to_numpy()
        lines.append(
            f"{c:<22} {v.mean():>7.3f} {v.std():>7.3f}  "
            f"{np.percentile(v, 5):>7.3f} {np.percentile(v, 50):>7.3f} "
            f"{np.percentile(v, 95):>7.3f}"
        )
    return "\n".join(lines)


def fixed_scenario_table(
    n_grid: list[int],
    n_debates: int,
    n_inner: int,
    seed: int,
    scenarios: list[tuple[str, float, float]],
) -> str:
    rng = np.random.default_rng(seed)
    lines = [
        "\n--- Fixed-scenario simulation (for comparison with the original script) ---",
        f"{'N/cell':>7}  " + "  ".join(f"{lbl:>22}" for lbl, _, _ in scenarios),
        "-" * (9 + 24 * len(scenarios)),
    ]
    for n in n_grid:
        cells = []
        for lbl, tau, sigma in scenarios:
            m = simulate_ranking_recovery(
                tau=tau, sigma=sigma, n_per=n,
                n_debates=n_debates, n_inner=n_inner, rng=rng,
            )
            cells.append(f"{m['tier_4_agree'] * 100:>6.1f}% (4-tier)")
        lines.append(f"{n:>7}  " + "  ".join(f"{c:>22}" for c in cells))
    return "\n".join(lines)


def build_headline_summary(full_results: list[pd.DataFrame], n_grid: list[int]) -> str:
    big = pd.concat(full_results, ignore_index=True)
    lines = [
        "=" * 78,
        "HEADLINE SAMPLE-SIZE TABLE (for the pre-registration)",
        "=" * 78,
        "",
        "Marginalised over pilot estimation uncertainty (parametric bootstrap of",
        "variance components from the planned hierarchical model).  Each row gives",
        "the expected 4-tier agreement (fraction of the 24 transcripts placed in",
        "the correct difficulty quartile), with a 5-95% credibility interval that",
        "reflects how much the result depends on the noisy pilot variance estimates.",
        "",
        "tau_marginal      = sqrt(var_question + var_variant): treats transcripts as",
        "                    drawn from the full hierarchical population.  Use this",
        "                    if you care about ranking across the full set of 24.",
        "tau_variant_only  = sqrt(var_variant) alone: treats ranking as conditional",
        "                    on question.  Use this if you care about within-question",
        "                    discrimination only.",
        "",
    ]
    for outcome_id in big["outcome"].unique():
        outcome_label = {"end_state": "End-state DV (logit s4)",
                         "update": "Update DV (logit s4 - logit s1)"}[outcome_id]
        lines.append(f"--- {outcome_label} ---")
        lines.append(
            f"{'N/cell':>7} | "
            f"{'tau_marginal: 4-tier mean [5%, 95%]':>40} | "
            f"{'tau_variant_only: 4-tier mean [5%, 95%]':>42}"
        )
        lines.append("-" * 96)
        for n in n_grid:
            cells = []
            for tau_def in ("marginal", "variant_only"):
                v = big[(big["outcome"] == outcome_id)
                        & (big["tau_definition"] == tau_def)
                        & (big["n_per"] == n)]["tier_4_agree"].to_numpy() * 100
                cells.append(
                    f"{v.mean():>5.1f}% [{np.percentile(v, 5):>4.1f}%, "
                    f"{np.percentile(v, 95):>4.1f}%]"
                )
            lines.append(
                f"{n:>7} | {cells[0]:>40} | {cells[1]:>42}"
            )
        lines.append("")
    lines.append("How to use this table in the pre-reg:")
    lines.append("  - Quote the mean as the planning estimate.")
    lines.append("  - Quote the 5-95% interval as 'pilot-uncertainty credible interval'.")
    lines.append("  - Use tau_marginal as the headline; mention tau_variant_only as a")
    lines.append("    conservative alternative for within-question discrimination.")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    out_prefix = args.out
    out_summary = out_prefix.with_suffix(".summary.txt")
    out_fits = out_prefix.with_suffix(".fits.txt")
    out_full_csv = out_prefix.with_suffix(".full_results.csv")
    out_boot_csv = out_prefix.with_suffix(".bootstrap_draws.csv")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("sample_size_with_uncertainty.py")
    print("=" * 70)
    print(f"Pilot data: {args.xlsx}")
    print(f"Bootstrap iterations: {args.n_boot}")
    print(f"Inner-sim replicates per draw: {args.n_inner}")
    print(f"Sample-size grid: {args.n_grid}")
    print(f"Output prefix: {out_prefix}")
    print()

    df = load_pilot(args.xlsx)
    print(f"Loaded {len(df)} pilot observations from "
          f"{df['participant_id'].nunique()} participants.")
    if "unknown" in df["domain"].unique():
        unk_qs = df.loc[df["domain"] == "unknown", "question"].unique()
        print(f"  WARNING: {len(unk_qs)} question(s) not in DOMAIN_MAP: "
              f"{list(unk_qs)}")
        print( "  These will be lumped into a single 'unknown' domain.")

    summary_lines = []
    fits_lines = []
    full_results = []
    boot_results = {}

    for outcome_id, outcome_col, outcome_label in OUTCOMES:
        print()
        print("=" * 70)
        print(f"OUTCOME: {outcome_label}")
        print("=" * 70)
        section_summary = [f"\n{'#' * 70}\n# {outcome_label}\n{'#' * 70}"]
        section_fits = [f"\n{'#' * 70}\n# {outcome_label}\n{'#' * 70}"]

        sub = df[df["group"] == "novice"].reset_index(drop=True)
        print(f"  Novice subset: n={len(sub)}, "
              f"variants={sub['variant'].nunique()}, "
              f"questions={sub['question'].nunique()}, "
              f"domains={sub['domain'].nunique()}, "
              f"participants={sub['participant_id'].nunique()}")

        design = build_design(sub)
        y = sub[outcome_col].to_numpy(dtype=float)

        print("  Fitting REML model on observed pilot data...", flush=True)
        t0 = time.time()
        fit = fit_reml(y, design)
        print(f"  ...done in {time.time() - t0:.1f}s "
              f"(neg loglik = {fit['_neg_loglik']:.2f}, "
              f"converged = {fit['_converged']})")
        fit_summary = summarise_fit(fit, design,
                                    f"{outcome_label}: REML fit on observed data (novice subset)")
        print(fit_summary)
        section_fits.append(fit_summary)

        print(f"  Running parametric bootstrap (n_boot={args.n_boot})...", flush=True)
        t0 = time.time()
        boot = parametric_bootstrap(
            fit=fit, design=design, n_boot=args.n_boot, seed=args.seed,
        )
        print(f"  ...done in {time.time() - t0:.1f}s")
        boot_summary = summarise_bootstrap(boot, outcome_label)
        print(boot_summary)
        section_fits.append(boot_summary)
        boot["outcome"] = outcome_id
        boot_results[outcome_id] = boot

        print(f"  Running marginal sample-size simulation "
              f"(tau_marginal = sqrt(var_q + var_v))...", flush=True)
        t0 = time.time()
        sim_marg = marginal_simulation(
            boot=boot, n_grid=args.n_grid, n_debates=args.n_debates,
            n_inner=args.n_inner, seed=args.seed, tau_col="tau_marginal",
        )
        print(f"  ...done in {time.time() - t0:.1f}s")
        sim_marg["outcome"] = outcome_id
        sim_marg["tau_definition"] = "marginal"
        full_results.append(sim_marg)

        marg_summary = summarise_marginal(
            sim_marg, "tier_4_agree",
            f"\n=== MARGINAL SIM, tau_marginal = sqrt(var_q + var_v) "
            f"({outcome_label}) ===\nMetric: 4-tier agreement; "
            f"intervals reflect pilot estimation uncertainty"
        )
        print(marg_summary)
        section_summary.append(marg_summary)

        print(f"  Running variant-only-tau simulation "
              f"(tau = sqrt(var_v) only, conservative)...", flush=True)
        t0 = time.time()
        sim_var = marginal_simulation(
            boot=boot, n_grid=args.n_grid, n_debates=args.n_debates,
            n_inner=args.n_inner, seed=args.seed + 1, tau_col="tau_variant_only",
        )
        print(f"  ...done in {time.time() - t0:.1f}s")
        sim_var["outcome"] = outcome_id
        sim_var["tau_definition"] = "variant_only"
        full_results.append(sim_var)

        var_summary = summarise_marginal(
            sim_var, "tier_4_agree",
            f"\n=== MARGINAL SIM, tau_variant_only = sqrt(var_v) "
            f"({outcome_label}) ===\nMetric: 4-tier agreement; this is the "
            f"more conservative within-question planning quantity"
        )
        print(var_summary)
        section_summary.append(var_summary)

        scenarios = [
            ("pessimistic (tau=0.45)", 0.45, fit["sigma"] if "sigma" in fit else
                float(np.sqrt(fit["var_participant"] + fit["var_residual"]))),
            ("central (tau=tau_marg)",
                float(np.sqrt(fit["var_question"] + fit["var_variant"])),
                float(np.sqrt(fit["var_participant"] + fit["var_residual"]))),
            ("optimistic (tau=1.10)", 1.10,
                float(np.sqrt(fit["var_participant"] + fit["var_residual"]))),
        ]
        legacy = fixed_scenario_table(
            n_grid=args.n_grid, n_debates=args.n_debates,
            n_inner=args.n_inner * 5, seed=args.seed, scenarios=scenarios,
        )
        print(legacy)
        section_summary.append(legacy)

        summary_lines.extend(section_summary)
        fits_lines.extend(section_fits)

    headline = build_headline_summary(full_results, args.n_grid)
    print()
    print(headline)
    summary_lines.insert(0, headline)

    out_summary.write_text("\n".join(summary_lines), encoding="utf-8")
    out_fits.write_text("\n".join(fits_lines), encoding="utf-8")
    pd.concat(full_results, ignore_index=True).to_csv(out_full_csv, index=False)
    pd.concat(boot_results.values(), ignore_index=True).to_csv(out_boot_csv, index=False)

    print()
    print("=" * 70)
    print("OUTPUTS WRITTEN:")
    print(f"  {out_summary}    (headline tables for pre-reg)")
    print(f"  {out_fits}       (raw fits + bootstrap distributions)")
    print(f"  {out_full_csv}   (long-format simulation results)")
    print(f"  {out_boot_csv}   (bootstrap draws of variance components)")
    print("=" * 70)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xlsx", type=Path, default=INPUT_XLSX,
                    help=f"Path to the pilot xlsx file (default: {INPUT_XLSX}).")
    ap.add_argument("--n-boot", type=int, default=300,
                    help="Number of parametric bootstrap iterations (default 300; "
                         "use 500-1000 for the final analysis).")
    ap.add_argument("--n-inner", type=int, default=30,
                    help="Inner-simulation replicates per (bootstrap draw, N) cell "
                         "(default 30; 50-100 for final).")
    ap.add_argument("--n-debates", type=int, default=24,
                    help="Number of debate variants in the main study (default 24).")
    ap.add_argument("--n-grid", type=int, nargs="+", default=DEFAULT_N_GRID,
                    help=f"Sample sizes per cell to evaluate (default {DEFAULT_N_GRID}).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (default 0).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PREFIX,
                    help=f"Output prefix (default {DEFAULT_OUT_PREFIX}).")
    args = ap.parse_args(argv)
    if not args.xlsx.exists():
        print(f"ERROR: pilot file not found: {args.xlsx}", file=sys.stderr)
        sys.exit(1)
    ensure_analysis_dir()
    run(args)


if __name__ == "__main__":
    main()
