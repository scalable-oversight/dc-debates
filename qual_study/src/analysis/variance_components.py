"""Variance-components analysis for debate-difficulty planning.

Estimates variance components relevant to planning the debate-difficulty
study, from the cleaned noids xlsx.

For each of three subsets (all / novices / experts) and two DVs
(end-state logit credence, update = s4 - s1 logit), it fits:

    y_ij = mu + beta * order_ij
         + domain_d[ij]                (random, optional)
         + question_q[ij]              (random, optional)
         + variant_v[ij]               (random)
         + participant_p[ij]           (random)
         + residual_ij

and reports the SDs of each component, their variance-partition shares,
and the implied 'sigma' (within-variant, single-judge SD) that drives
sample-size calculations.

It ALSO checks whether domain- and question-level variance are big
enough to matter as random effects, vs. a fixed-effect treatment, using
(a) the variance-component estimates, (b) a likelihood-ratio test
(random vs no random effect for that level), and (c) a simple rule of
thumb on the number of levels.

Usage:
    python -m src.analysis.variance_components
    python -m src.analysis.variance_components --xlsx path/to/pilot.xlsx --out results.txt
"""

from __future__ import annotations

import argparse
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ._io import ANALYSIS_DIR, INPUT_XLSX, ensure_analysis_dir

DEFAULT_OUT_FILE = ANALYSIS_DIR / "variance_components_results.txt"

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


class Tee:
    def __init__(self):
        self.buf = io.StringIO()

    def write(self, s: str = "") -> None:
        print(s)
        self.buf.write(s + "\n")

    def value(self) -> str:
        return self.buf.getvalue()


def vc_handrolled(df: pd.DataFrame, dv: str, group_cols: list[str]) -> dict:
    y = df[dv].to_numpy(dtype=float)
    order = df["order_coded"].to_numpy(dtype=float)

    X = np.column_stack([np.ones_like(y), order])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid_fe = y - X @ beta

    group_idx = {g: df[g].to_numpy() for g in group_cols}
    group_levels = {g: np.unique(group_idx[g]) for g in group_cols}
    group_eff = {g: {lvl: 0.0 for lvl in group_levels[g]} for g in group_cols}

    for _ in range(300):
        for g in group_cols:
            tmp = resid_fe.copy()
            for gg in group_cols:
                if gg == g:
                    continue
                tmp -= np.array([group_eff[gg][lvl] for lvl in group_idx[gg]])
            new_eff = {}
            for lvl in group_levels[g]:
                mask = group_idx[g] == lvl
                new_eff[lvl] = tmp[mask].mean() if mask.any() else 0.0
            group_eff[g] = new_eff

    pred = X @ beta
    for g in group_cols:
        pred = pred + np.array([group_eff[g][lvl] for lvl in group_idx[g]])
    resid = y - pred

    var_resid = float(np.var(resid, ddof=1))

    out = {
        "beta_intercept": float(beta[0]),
        "beta_order": float(beta[1]),
        "var_resid": var_resid,
        "sd_resid": np.sqrt(var_resid),
        "_source": "handrolled",
    }
    for g in group_cols:
        effs = np.array(list(group_eff[g].values()))
        naive_var = float(np.var(effs, ddof=1)) if len(effs) > 1 else 0.0
        n_per = len(y) / max(1, len(effs))
        corrected = max(0.0, naive_var - var_resid / n_per)
        out[f"var_{g}"] = corrected
        out[f"sd_{g}"] = np.sqrt(corrected)
    return out


def vc_statsmodels(df: pd.DataFrame, dv: str, group_cols: list[str]) -> dict:
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {}
    if not group_cols:
        return {}

    sizes = {g: df[g].nunique() for g in group_cols}
    top = max(sizes, key=sizes.get)
    rest = [g for g in group_cols if g != top]

    vc_formula = {g: f"0 + C({g})" for g in rest}

    df2 = df.copy()
    try:
        md = smf.mixedlm(
            f"{dv} ~ order_coded",
            df2,
            groups=df2[top],
            vc_formula=vc_formula if vc_formula else None,
            re_formula="1",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdf = md.fit(reml=True, method="lbfgs")
    except Exception as e:
        return {"_source": f"statsmodels_failed:{e.__class__.__name__}"}

    out = {
        "beta_intercept": float(mdf.fe_params["Intercept"]),
        "beta_order": float(mdf.fe_params["order_coded"]),
        "var_resid": float(mdf.scale),
        "sd_resid": float(np.sqrt(mdf.scale)),
        "_source": "statsmodels",
        "_loglik": float(mdf.llf),
    }
    try:
        var_top = float(mdf.cov_re.iloc[0, 0])
    except Exception:
        var_top = float(np.asarray(mdf.cov_re).ravel()[0])
    out[f"var_{top}"] = var_top
    out[f"sd_{top}"] = np.sqrt(max(0.0, var_top))

    vcomp_vals = np.asarray(mdf.vcomp).ravel()
    for i, g in enumerate(rest):
        v = float(vcomp_vals[i]) if i < len(vcomp_vals) else 0.0
        out[f"var_{g}"] = v
        out[f"sd_{g}"] = np.sqrt(max(0.0, v))

    return out


def fit_vc(df: pd.DataFrame, dv: str, group_cols: list[str]) -> dict:
    res = vc_statsmodels(df, dv, group_cols)
    if not res or res.get("_source", "").startswith("statsmodels_failed"):
        res = vc_handrolled(df, dv, group_cols)
    return res


def lrt_add_random(
    df: pd.DataFrame,
    dv: str,
    baseline: list[str],
    candidate: str,
) -> dict:
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"error": "statsmodels not available"}

    def ll(groups: list[str]) -> float | None:
        if not groups:
            y = df[dv].to_numpy()
            X = np.column_stack([np.ones_like(y), df["order_coded"].to_numpy()])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            n = len(y)
            sigma2 = (resid @ resid) / n
            return -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)

        top = max(groups, key=lambda g: df[g].nunique())
        rest = [g for g in groups if g != top]
        try:
            md = smf.mixedlm(
                f"{dv} ~ order_coded",
                df,
                groups=df[top],
                vc_formula={g: f"0 + C({g})" for g in rest} if rest else None,
                re_formula="1",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdf = md.fit(reml=False, method="lbfgs")
            return float(mdf.llf)
        except Exception:
            return None

    ll0 = ll(baseline)
    ll1 = ll(baseline + [candidate])
    if ll0 is None or ll1 is None:
        return {"error": "fit failed"}

    lr = 2 * (ll1 - ll0)
    p_naive = 1.0 - stats.chi2.cdf(max(0.0, lr), df=1)
    p_boundary = p_naive / 2.0
    return {"ll0": ll0, "ll1": ll1, "lr": lr, "p": p_boundary, "n_levels": df[candidate].nunique()}


def fmt_vc(tee: Tee, res: dict, group_cols: list[str], label: str, subset_size: int) -> None:
    tee.write(f"\n  DV: {label}   (n={subset_size}, source={res.get('_source', '?')})")
    tee.write(f"    Fixed: intercept = {res['beta_intercept']:+.3f}, "
              f"beta_order (honest - dishonest) = {res['beta_order']:+.3f}")
    total = res["var_resid"] + sum(res.get(f"var_{g}", 0.0) for g in group_cols)
    if total <= 0:
        tee.write("    (total variance <= 0; nothing to partition)")
        return
    for g in group_cols:
        v = res.get(f"var_{g}", 0.0)
        sd = res.get(f"sd_{g}", 0.0)
        tee.write(f"    SD({g:<11}) = {sd:5.3f}   var share = {100 * v / total:5.1f}%")
    tee.write(f"    SD(residual)    = {res['sd_resid']:5.3f}   var share = {100 * res['var_resid'] / total:5.1f}%")

    sigma2 = res["var_resid"]
    for g in group_cols:
        if g != "variant":
            sigma2 += res.get(f"var_{g}", 0.0)
    sigma = np.sqrt(sigma2)
    tau = res.get("sd_variant", 0.0)
    tee.write(f"    --> tau (between-variant SD) = {tau:.3f}")
    tee.write(f"    --> sigma (within-variant single-judge SD, for sample-size calc) = {sigma:.3f}")
    tee.write(f"    --> tau / sigma = {tau / sigma:.3f}" if sigma > 0 else "    --> sigma is zero")


def domain_recommendation(vc_with_domain: dict, lrt: dict, n_domains: int, tee: Tee) -> None:
    sd_domain = vc_with_domain.get("sd_domain", 0.0)
    sd_question = vc_with_domain.get("sd_question", 0.0)
    sd_variant = vc_with_domain.get("sd_variant", 0.0)

    within = np.sqrt(
        vc_with_domain["var_resid"]
        + vc_with_domain.get("var_participant", 0.0)
        + vc_with_domain.get("var_question", 0.0)
    )

    tee.write("")
    tee.write(f"    Domain SD    = {sd_domain:.3f}    (vs within-variant noise SD {within:.3f})")
    tee.write(f"    Question SD  = {sd_question:.3f}")
    tee.write(f"    Variant SD   = {sd_variant:.3f}")

    if lrt.get("error"):
        tee.write(f"    LRT for domain random effect: {lrt['error']}")
        lrt_verdict = "inconclusive"
    else:
        tee.write(f"    LRT domain vs no-domain: LR = {lrt['lr']:.2f}, "
                  f"p (boundary-corrected) = {lrt['p']:.3f}")
        if lrt["p"] < 0.05:
            lrt_verdict = "significant"
        elif lrt["p"] < 0.15:
            lrt_verdict = "marginal"
        else:
            lrt_verdict = "not-significant"

    tee.write("")
    tee.write("    RECOMMENDATION:")
    if n_domains < 5:
        tee.write(f"      With only {n_domains} domains, treat domain as FIXED effect.")
        tee.write( "      You cannot reliably estimate a variance from so few levels, regardless")
        tee.write( "      of what the LRT says.  Use N-1 indicator variables.")
    elif n_domains < 8:
        tee.write(f"      With {n_domains} domains, this is borderline.  Fixed effects are safe;")
        tee.write(f"      random effects are defensible if the LRT is significant ({lrt_verdict}).")
    else:
        tee.write(f"      With {n_domains} domains, random effects are fine.  LRT is {lrt_verdict}.")

    if sd_domain < 0.3 * within:
        tee.write( "      Domain SD is small relative to within-variant noise, so the practical")
        tee.write( "      consequences of the fixed-vs-random choice are minor either way.")
    else:
        tee.write( "      Domain SD is NOT small relative to within-variant noise; the choice")
        tee.write( "      has real consequences for inference and should be decided carefully.")


def run(xlsx_path: Path, out_path: Path) -> None:
    tee = Tee()

    tee.write("=" * 70)
    tee.write("Variance-components analysis for debate-difficulty planning")
    tee.write("=" * 70)
    tee.write(f"Input file:  {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    tee.write(f"Rows: {len(df)};  columns: {len(df.columns)}")

    needed = {
        "debate_name",
        "debate_turn_4_link",
        "participant_id",
        "order",
        "group",
        "logit_section_1_credence_in_correct_answer",
        "logit_section_4_credence_in_correct_answer",
    }
    missing = needed - set(df.columns)
    if missing:
        tee.write(f"\nERROR: missing required columns: {sorted(missing)}")
        out_path.write_text(tee.value(), encoding="utf-8")
        sys.exit(1)

    df["variant"] = df["debate_turn_4_link"]
    df["question"] = df["debate_name"]
    df["domain"] = df["debate_name"].map(DOMAIN_MAP).fillna("unknown")
    df["update"] = (
        df["logit_section_4_credence_in_correct_answer"]
        - df["logit_section_1_credence_in_correct_answer"]
    )
    df["order_coded"] = (df["order"] == "Honest first").astype(int)
    df["participant_id"] = df["participant_id"].astype(str)

    tee.write(f"Unique variants:    {df['variant'].nunique()}")
    tee.write(f"Unique questions:   {df['question'].nunique()}")
    tee.write(f"Unique domains:     {df['domain'].nunique()} ({sorted(df['domain'].unique())})")
    tee.write(f"Unique participants:{df['participant_id'].nunique()}")

    if (df["domain"] == "unknown").any():
        unks = df.loc[df["domain"] == "unknown", "question"].unique()
        tee.write(f"\nNOTE: {len(unks)} question(s) not in DOMAIN_MAP: {list(unks)}")
        tee.write( "       They will be lumped into a single 'unknown' domain.")
        tee.write( "       Edit DOMAIN_MAP at the top of this script to fix.")

    subsets = [
        ("ALL (experts + novices)", df),
        ("NOVICES ONLY (Prolific AI-tasker proxy)", df[df["group"] == "novice"]),
        ("EXPERTS ONLY (for comparison)", df[df["group"] == "expert"]),
    ]

    for label, sub in subsets:
        tee.write("\n" + "=" * 70)
        tee.write(label + f"   n={len(sub)}")
        tee.write("=" * 70)

        for dv, dv_label in [
            ("logit_section_4_credence_in_correct_answer", "end-state (logit section 4)"),
            ("update", "update (logit s4 - logit s1)"),
        ]:
            tee.write(f"\n--- {dv_label} ---")

            full = fit_vc(sub, dv, ["domain", "question", "variant", "participant_id"])
            for k in list(full.keys()):
                if k.endswith("participant_id"):
                    full[k.replace("participant_id", "participant")] = full.pop(k)
            fmt_vc(tee, full, ["domain", "question", "variant", "participant"],
                   dv_label + "  -- full model", len(sub))

            tee.write("\n  [Domain random-effect check]")
            lrt = lrt_add_random(
                sub, dv,
                baseline=["question", "variant", "participant_id"],
                candidate="domain",
            )
            domain_recommendation(full, lrt, sub["domain"].nunique(), tee)

            plan = fit_vc(sub, dv, ["variant", "participant_id"])
            for k in list(plan.keys()):
                if k.endswith("participant_id"):
                    plan[k.replace("participant_id", "participant")] = plan.pop(k)
            tee.write("")
            fmt_vc(tee, plan, ["variant", "participant"],
                   dv_label + "  -- planning model (for sample-size calc)", len(sub))

        r = sub[
            ["logit_section_1_credence_in_correct_answer",
             "logit_section_4_credence_in_correct_answer"]
        ].corr().iloc[0, 1]
        tee.write(f"\n  r(s1, s4) on logit scale = {r:.3f}")

    tee.write("\n" + "=" * 70)
    tee.write("HOW TO READ THIS:")
    tee.write("=" * 70)
    tee.write(
        "  tau   = between-variant SD (signal).  Bigger = debates are more\n"
        "          distinguishable in difficulty, which is good for you.\n"
        "  sigma = within-variant, single-judge SD (noise).  In the main\n"
        "          between-subjects study, participant-level variance collapses\n"
        "          into this, so sigma = sqrt(var_participant + var_residual\n"
        "          + any-other-non-variant-variances).  Smaller = easier to\n"
        "          rank debates precisely.\n"
        "  tau / sigma is the signal-to-noise ratio that governs ranking\n"
        "          precision; your pilot gave roughly 1.0 for novices on both\n"
        "          DVs, which is favourable.\n"
        "\n"
        "  For the domain random-effect decision, the simulation-based rule is:\n"
        "    <5 levels:   fixed effects, full stop.\n"
        "    5-7 levels:  fixed effects unless the LRT is clearly significant.\n"
        "    8+ levels:   random effects usually fine.\n"
        "  Your pilot has 3 (or however you mapped them) domains, so with very\n"
        "  high probability the recommendation will be FIXED effects."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tee.value(), encoding="utf-8")
    tee.write(f"\n(Results also written to {out_path})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xlsx", type=Path, default=INPUT_XLSX,
                    help=f"Path to the pilot xlsx file (default: {INPUT_XLSX})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE,
                    help=f"Where to write the results (default: {DEFAULT_OUT_FILE})")
    args = ap.parse_args(argv)
    if not args.xlsx.exists():
        print(f"ERROR: {args.xlsx} not found", file=sys.stderr)
        sys.exit(1)
    ensure_analysis_dir()
    run(args.xlsx, args.out)


if __name__ == "__main__":
    main()
