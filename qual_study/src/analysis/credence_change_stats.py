"""Compute section 4 minus section 1 credence change statistics and write a report."""

from __future__ import annotations

import warnings

import numpy as np
import statsmodels.formula.api as smf
from scipy import stats as sp_stats

from ._io import ANALYSIS_DIR, ensure_analysis_dir, load_combined

CLUSTERING_CAVEAT = (
    "  NOTE: Observations are not independent — participants may appear in"
    " multiple debates.\n"
    "  p-values and CIs do not account for this clustering and may be"
    " anti-conservative."
)

N_BOOTSTRAP = 10_000

CI_LEVELS = [
    (95, 1.960),
    (90, 1.645),
    (80, 1.282),
    (75, 1.150),
]


def logit_to_prob(x):
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    ensure_analysis_dir()
    df = load_combined()
    df["logit_change"] = (
        df["logit_section_4_credence_in_correct_answer"]
        - df["logit_section_1_credence_in_correct_answer"]
    )
    df["int_change"] = (
        df["section_4_credence_in_correct_answer_as_integer"]
        - df["section_1_credence_in_correct_answer_as_integer"]
    )

    rng = np.random.default_rng(42)
    lines: list[str] = []

    def heading(text):
        lines.append("")
        lines.append("=" * 70)
        lines.append(text)
        lines.append("=" * 70)

    def credible_intervals(values):
        clean = values.dropna()
        n = len(clean)
        if n < 2:
            return n, np.nan, [(level, np.nan, np.nan) for level, _ in CI_LEVELS]
        mean = clean.mean()
        se = clean.std(ddof=1) / np.sqrt(n)
        cis = [(level, mean - z * se, mean + z * se) for level, z in CI_LEVELS]
        return n, mean, cis

    def report_logit_ci(data):
        col = "logit_change"
        for label, subset in [("Overall", data), ("Experts", data[data["group"] == "expert"]),
                               ("Novices", data[data["group"] == "novice"])]:
            n, mean, cis = credible_intervals(subset[col])
            ci_str = "  ".join(
                f"{level}% [{lo:+.3f}, {hi:+.3f}]" for level, lo, hi in cis
            )
            lines.append(
                f"  {label:10s}  n={n:3d}  mean={mean:+.3f}  {ci_str}"
            )

    def report_ttest(data):
        col = "logit_change"
        e = data.loc[data["group"] == "expert", col].dropna()
        n = data.loc[data["group"] == "novice", col].dropna()
        if len(e) < 2 or len(n) < 2:
            lines.append("  Insufficient data for t-test")
            return
        t_stat, p_val = sp_stats.ttest_ind(e, n, equal_var=False)
        lines.append(f"  Expert  n={len(e):3d}  mean={e.mean():+.3f}  sd={e.std(ddof=1):.3f}")
        lines.append(f"  Novice  n={len(n):3d}  mean={n.mean():+.3f}  sd={n.std(ddof=1):.3f}")
        lines.append(f"  Welch's t = {t_stat:+.3f},  p = {p_val:.4f}")

    def report_binomial(data):
        col = "int_change"
        for label, subset in [("Overall", data), ("Experts", data[data["group"] == "expert"]),
                               ("Novices", data[data["group"] == "novice"])]:
            vals = subset[col].dropna()
            decisive = vals[vals != 0]
            n_decisive = len(decisive)
            n_toward = int((decisive > 0).sum())
            n_away = int((decisive < 0).sum())
            n_no_change = int((vals == 0).sum())
            if n_decisive == 0:
                lines.append(f"  {label:10s}  no decisive observations (all unchanged)")
                continue
            result = sp_stats.binomtest(n_toward, n_decisive, 0.5)
            pct_toward = n_toward / n_decisive * 100
            lines.append(
                f"  {label:10s}  n={len(vals):3d}  "
                f"toward correct: {n_toward}  away: {n_away}  unchanged: {n_no_change}  "
                f"({pct_toward:.1f}% toward, excluding unchanged)  "
                f"binom p = {result.pvalue:.4f}"
            )

    def report_variance(data, col="logit_change"):
        valid = data[["participant_id", "group", col]].dropna(subset=[col])
        experts = valid.loc[valid["group"] == "expert"]
        novices = valid.loc[valid["group"] == "novice"]

        expert_pids = experts["participant_id"].unique()
        novice_pids = novices["participant_id"].unique()
        n_e, n_n = len(expert_pids), len(novice_pids)

        if n_e < 2 or n_n < 2:
            lines.append("  Insufficient data for variance comparison")
            return

        all_pids = np.concatenate([expert_pids, novice_pids])
        obs_by_pid = {}
        for pid in all_pids:
            obs_by_pid[pid] = valid.loc[valid["participant_id"] == pid, col].values

        obs_var_e = experts[col].var(ddof=1)
        obs_var_n = novices[col].var(ddof=1)
        obs_ratio = obs_var_e / obs_var_n
        obs_log_ratio = abs(np.log(obs_ratio))

        e_pid_arr = np.array(list(expert_pids))
        n_pid_arr = np.array(list(novice_pids))
        boot_ratios = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            e_sample = rng.choice(e_pid_arr, size=n_e, replace=True)
            n_sample = rng.choice(n_pid_arr, size=n_n, replace=True)
            e_vals = np.concatenate([obs_by_pid[pid] for pid in e_sample])
            n_vals = np.concatenate([obs_by_pid[pid] for pid in n_sample])
            if len(e_vals) < 2 or len(n_vals) < 2:
                boot_ratios[i] = np.nan
                continue
            var_e = e_vals.var(ddof=1)
            var_n = n_vals.var(ddof=1)
            if var_n == 0 or var_e == 0:
                boot_ratios[i] = np.nan
                continue
            boot_ratios[i] = var_e / var_n

        valid_boots = boot_ratios[np.isfinite(boot_ratios)]
        ci_lo, ci_hi = np.percentile(valid_boots, [2.5, 97.5])

        perm_log_ratios = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            shuffled = rng.permutation(all_pids)
            perm_e = shuffled[:n_e]
            perm_n = shuffled[n_e:]
            e_vals = np.concatenate([obs_by_pid[pid] for pid in perm_e])
            n_vals = np.concatenate([obs_by_pid[pid] for pid in perm_n])
            if len(e_vals) < 2 or len(n_vals) < 2:
                perm_log_ratios[i] = 0.0
                continue
            var_e = e_vals.var(ddof=1)
            var_n = n_vals.var(ddof=1)
            if var_n == 0 or var_e == 0:
                perm_log_ratios[i] = 0.0
                continue
            perm_log_ratios[i] = abs(np.log(var_e / var_n))

        p_val = np.mean(perm_log_ratios >= obs_log_ratio)

        sd_e, sd_n = np.sqrt(obs_var_e), np.sqrt(obs_var_n)
        lines.append(f"  Expert  n={n_e:3d} participants  var={obs_var_e:.3f}  sd={sd_e:.3f}")
        lines.append(f"  Novice  n={n_n:3d} participants  var={obs_var_n:.3f}  sd={sd_n:.3f}")
        lines.append(f"  Variance ratio (E/N) = {obs_ratio:.3f}  95% bootstrap CI [{ci_lo:.3f}, {ci_hi:.3f}]")
        lines.append(f"  Permutation p = {p_val:.4f}  ({N_BOOTSTRAP:,} iterations)")

    def report_median_iqr(data):
        col = "int_change"
        for label, subset in [("Overall", data), ("Experts", data[data["group"] == "expert"]),
                               ("Novices", data[data["group"] == "novice"])]:
            valid = subset[["participant_id", col]].dropna(subset=[col])
            if len(valid) == 0:
                lines.append(f"  {label:10s}  n=  0  (no data)")
                continue
            per_participant = valid.groupby("participant_id")[col].mean()
            n_participants = len(per_participant)
            q1, median, q3 = per_participant.quantile([0.25, 0.5, 0.75])
            mean = per_participant.mean()
            lines.append(
                f"  {label:10s}  n={n_participants:3d} participants  "
                f"mean={mean:+5.1f}  median={median:+5.1f}  IQR=[{q1:+.1f}, {q3:+.1f}]"
            )

    def report_median_iqr_obs(data):
        col = "int_change"
        for label, subset in [("Overall", data), ("Experts", data[data["group"] == "expert"]),
                               ("Novices", data[data["group"] == "novice"])]:
            vals = subset[col].dropna()
            if len(vals) == 0:
                lines.append(f"  {label:10s}  n=  0  (no data)")
                continue
            q1, median, q3 = vals.quantile([0.25, 0.5, 0.75])
            mean = vals.mean()
            lines.append(
                f"  {label:10s}  n={len(vals):3d} observations  "
                f"mean={mean:+5.1f}  median={median:+5.1f}  IQR=[{q1:+.1f}, {q3:+.1f}]"
            )

    def report_order_bias(data):
        col = "logit_change"
        valid = data[["participant_id", "order", col]].dropna()
        if len(valid) < 4 or valid["order"].nunique() < 2:
            lines.append("  Insufficient data for order analysis")
            return

        for order_val in sorted(valid["order"].unique()):
            vals = valid.loc[valid["order"] == order_val, col]
            lines.append(
                f"  {order_val:20s}  n={len(vals):3d}  "
                f"mean={vals.mean():+.3f}  sd={vals.std(ddof=1):.3f}"
            )

        n_participants = valid["participant_id"].nunique()
        if n_participants < len(valid):
            valid = valid.copy()
            valid["honest_first"] = (valid["order"] == "Honest first").astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    f"{col} ~ honest_first", data=valid,
                    groups=valid["participant_id"],
                )
                result = model.fit(reml=True)
            coef = result.fe_params["honest_first"]
            se = result.bse["honest_first"]
            p_val = result.pvalues["honest_first"]
            lines.append(
                f"  Mixed-effects model (participant random intercept):"
            )
            lines.append(
                f"    Honest first effect: {coef:+.3f}  SE={se:.3f}  p={p_val:.4f}"
            )
            lines.append(
                f"    (positive = larger credence shift toward correct when honest goes first)"
            )
        else:
            honest = valid.loc[valid["order"] == "Honest first", col]
            dishonest = valid.loc[valid["order"] == "Dishonest first", col]
            if len(honest) < 2 or len(dishonest) < 2:
                lines.append("  Insufficient data per group for t-test")
                return
            t_stat, p_val = sp_stats.ttest_ind(honest, dishonest, equal_var=False)
            diff = honest.mean() - dishonest.mean()
            lines.append(f"  Welch's t = {t_stat:+.3f},  p = {p_val:.4f}")
            lines.append(
                f"  Difference (Honest first - Dishonest first): {diff:+.3f}"
            )

    # ── OVERALL ────────────────────────────────────────────────────────────────

    heading("95% CI for Logit Credence Change (section 4 - section 1)")
    lines.append(CLUSTERING_CAVEAT)
    report_logit_ci(df)

    heading("Independent t-test: Experts vs Novices (logit change)")
    lines.append(CLUSTERING_CAVEAT)
    report_ttest(df)

    heading("Variance comparison: Experts vs Novices (logit change, cluster bootstrap)")
    report_variance(df)

    heading("Binomial test: moved toward correct vs away (excluding unchanged)")
    lines.append(CLUSTERING_CAVEAT)
    report_binomial(df)

    heading("Means, Medians and IQRs of Integer Credence Change (section 4 - section 1, participant-level)")
    report_median_iqr(df)

    heading("Means, Medians and IQRs of Integer Credence Change (section 4 - section 1, observation-level)")
    report_median_iqr_obs(df)

    heading("Order bias: effect of Honest first vs Dishonest first on logit credence change")
    report_order_bias(df)

    # ── PER DEBATE ────────────────────────────────────────────────────────────

    for debate in sorted(df["debate_name"].dropna().unique()):
        deb_df = df[df["debate_name"] == debate]

        heading(f"DEBATE: {debate}")

        lines.append("")
        lines.append("95% CI for Logit Credence Change (section 4 - section 1)")
        report_logit_ci(deb_df)

        lines.append("")
        lines.append("Independent t-test: Experts vs Novices (logit change)")
        report_ttest(deb_df)

        lines.append("")
        lines.append("Variance comparison: Experts vs Novices (logit change, cluster bootstrap)")
        report_variance(deb_df)

        lines.append("")
        lines.append("Binomial test: moved toward correct vs away (excluding unchanged)")
        report_binomial(deb_df)

        lines.append("")
        lines.append("Means, Medians and IQRs of Integer Credence Change (section 4 - section 1, participant-level)")
        report_median_iqr(deb_df)

        lines.append("")
        lines.append("Means, Medians and IQRs of Integer Credence Change (section 4 - section 1, observation-level)")
        report_median_iqr_obs(deb_df)

        lines.append("")
        lines.append("Order bias: effect of Honest first vs Dishonest first on logit credence change")
        report_order_bias(deb_df)

    report = "\n".join(lines).strip() + "\n"
    out = ANALYSIS_DIR / "credence_change_stats.txt"
    out.write_text(report, encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
