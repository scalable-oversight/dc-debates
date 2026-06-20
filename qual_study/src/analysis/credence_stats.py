"""Compute credence statistics and write a text report."""

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


def logit_to_prob(x):
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    ensure_analysis_dir()
    df = load_combined()

    rng = np.random.default_rng(42)
    lines: list[str] = []

    def heading(text):
        lines.append("")
        lines.append("=" * 70)
        lines.append(text)
        lines.append("=" * 70)

    def subheading(text):
        lines.append("")
        lines.append(f"--- {text} ---")

    def credible_interval_95(values):
        clean = values.dropna()
        n = len(clean)
        if n < 2:
            return n, np.nan, np.nan, np.nan
        mean = clean.mean()
        se = clean.std(ddof=1) / np.sqrt(n)
        lo = mean - 1.96 * se
        hi = mean + 1.96 * se
        return n, mean, lo, hi

    def fmt_logit_ci(label, n, mean, lo, hi):
        p_mean = logit_to_prob(mean)
        p_lo = logit_to_prob(lo)
        p_hi = logit_to_prob(hi)
        return (
            f"  {label:10s}  n={n:3d}  "
            f"mean={mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
            f"(prob: {p_mean:.1%}  [{p_lo:.1%}, {p_hi:.1%}])"
        )

    def report_logit_ci(data, section):
        col = f"logit_section_{section}_credence_in_correct_answer"
        subheading(f"Section {section}")
        for label, subset in [("Overall", data), ("Experts", data[data["group"] == "expert"]),
                               ("Novices", data[data["group"] == "novice"])]:
            n, mean, lo, hi = credible_interval_95(subset[col])
            lines.append(fmt_logit_ci(label, n, mean, lo, hi))

    def report_ttest(data, section):
        col = f"logit_section_{section}_credence_in_correct_answer"
        subheading(f"Section {section}")
        e = data.loc[data["group"] == "expert", col].dropna()
        n = data.loc[data["group"] == "novice", col].dropna()
        if len(e) < 2 or len(n) < 2:
            lines.append("  Insufficient data for t-test")
            return
        t_stat, p_val = sp_stats.ttest_ind(e, n, equal_var=False)
        lines.append(f"  Expert  n={len(e):3d}  mean={e.mean():+.3f}  sd={e.std(ddof=1):.3f}")
        lines.append(f"  Novice  n={len(n):3d}  mean={n.mean():+.3f}  sd={n.std(ddof=1):.3f}")
        lines.append(f"  Welch's t = {t_stat:+.3f},  p = {p_val:.4f}")

    def report_binomial(data, section):
        col = f"section_{section}_credence_in_correct_answer_as_integer"
        subheading(f"Section {section}")
        for label, subset in [("Overall", data), ("Experts", data[data["group"] == "expert"]),
                               ("Novices", data[data["group"] == "novice"])]:
            vals = subset[col].dropna()
            decisive = vals[vals != 50]
            n_decisive = len(decisive)
            n_above = int((decisive > 50).sum())
            n_below = int((decisive < 50).sum())
            n_equal = int((vals == 50).sum())
            if n_decisive == 0:
                lines.append(f"  {label:10s}  no decisive observations (all = 50%)")
                continue
            result = sp_stats.binomtest(n_above, n_decisive, 0.5)
            pct_above = n_above / n_decisive * 100
            lines.append(
                f"  {label:10s}  n={len(vals):3d}  "
                f">50%: {n_above}  <50%: {n_below}  =50%: {n_equal}  "
                f"({pct_above:.1f}% above, excluding =50%)  "
                f"binom p = {result.pvalue:.4f}"
            )

    def _cluster_variance_ratio(data, col, n_iter=N_BOOTSTRAP):
        experts = data.loc[data["group"] == "expert"]
        novices = data.loc[data["group"] == "novice"]

        expert_pids = experts["participant_id"].unique()
        novice_pids = novices["participant_id"].unique()
        n_e, n_n = len(expert_pids), len(novice_pids)

        all_pids = np.concatenate([expert_pids, novice_pids])
        obs_by_pid = {}
        for pid in all_pids:
            obs_by_pid[pid] = data.loc[data["participant_id"] == pid, col].dropna().values

        expert_by_pid = {pid: obs_by_pid[pid] for pid in expert_pids}
        novice_by_pid = {pid: obs_by_pid[pid] for pid in novice_pids}

        obs_var_e = experts[col].dropna().var(ddof=1)
        obs_var_n = novices[col].dropna().var(ddof=1)
        obs_ratio = obs_var_e / obs_var_n
        obs_log_ratio = abs(np.log(obs_ratio))

        boot_ratios = np.empty(n_iter)
        e_pid_arr = np.array(list(expert_by_pid.keys()))
        n_pid_arr = np.array(list(novice_by_pid.keys()))

        for i in range(n_iter):
            e_sample = rng.choice(e_pid_arr, size=n_e, replace=True)
            n_sample = rng.choice(n_pid_arr, size=n_n, replace=True)
            e_vals = np.concatenate([expert_by_pid[pid] for pid in e_sample])
            n_vals = np.concatenate([novice_by_pid[pid] for pid in n_sample])
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

        perm_log_ratios = np.empty(n_iter)
        for i in range(n_iter):
            shuffled = rng.permutation(all_pids)
            perm_e_pids = shuffled[:n_e]
            perm_n_pids = shuffled[n_e:]
            e_vals = np.concatenate([obs_by_pid[pid] for pid in perm_e_pids])
            n_vals = np.concatenate([obs_by_pid[pid] for pid in perm_n_pids])
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

        return obs_var_e, obs_var_n, obs_ratio, p_val, ci_lo, ci_hi, n_e, n_n

    def _format_variance_report(obs_var_e, obs_var_n, obs_ratio, p_val, ci_lo, ci_hi, n_e_pids, n_n_pids):
        sd_e, sd_n = np.sqrt(obs_var_e), np.sqrt(obs_var_n)
        result = []
        result.append(f"  Expert  n={n_e_pids:3d} participants  var={obs_var_e:.3f}  sd={sd_e:.3f}")
        result.append(f"  Novice  n={n_n_pids:3d} participants  var={obs_var_n:.3f}  sd={sd_n:.3f}")
        result.append(f"  Variance ratio (E/N) = {obs_ratio:.3f}  95% bootstrap CI [{ci_lo:.3f}, {ci_hi:.3f}]")
        result.append(f"  Permutation p = {p_val:.4f}  ({N_BOOTSTRAP:,} iterations)")
        return result

    def report_variance(data, section):
        col = f"logit_section_{section}_credence_in_correct_answer"
        subheading(f"Section {section}")
        valid = data[["participant_id", "group", col]].dropna(subset=[col])
        e = valid.loc[valid["group"] == "expert"]
        n = valid.loc[valid["group"] == "novice"]
        if len(e["participant_id"].unique()) < 2 or len(n["participant_id"].unique()) < 2:
            lines.append("  Insufficient data for variance comparison")
            return
        results = _cluster_variance_ratio(valid, col)
        lines.extend(_format_variance_report(*results))

    def report_variance_raw(data, section):
        col = f"section_{section}_credence_in_correct_answer_as_integer"
        subheading(f"Section {section}")
        valid = data[["participant_id", "group", col]].dropna(subset=[col])
        e = valid.loc[valid["group"] == "expert"]
        n = valid.loc[valid["group"] == "novice"]
        if len(e["participant_id"].unique()) < 2 or len(n["participant_id"].unique()) < 2:
            lines.append("  Insufficient data for variance comparison")
            return
        results = _cluster_variance_ratio(valid, col)
        lines.extend(_format_variance_report(*results))

    def report_median_iqr(data, section):
        col = f"section_{section}_credence_in_correct_answer_as_integer"
        subheading(f"Section {section}")
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
                f"mean={mean:5.1f}  median={median:5.1f}  IQR=[{q1:.1f}, {q3:.1f}]"
            )

    def report_median_iqr_obs(data, section):
        col = f"section_{section}_credence_in_correct_answer_as_integer"
        subheading(f"Section {section}")
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
                f"mean={mean:5.1f}  median={median:5.1f}  IQR=[{q1:.1f}, {q3:.1f}]"
            )

    def report_order_bias(data, section):
        col = f"logit_section_{section}_credence_in_correct_answer"
        subheading(f"Section {section}")
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
            valid["y"] = valid[col]
            valid["honest_first"] = (valid["order"] == "Honest first").astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    "y ~ honest_first", data=valid,
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
                f"    (positive = higher logit credence in correct answer when honest goes first)"
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

    heading("95% Credible Intervals for Logit Credence (with back-calculated probs)")
    lines.append(CLUSTERING_CAVEAT)
    for section in range(1, 5):
        report_logit_ci(df, section)

    heading("Independent t-tests: Experts vs Novices (logit credence)")
    lines.append(CLUSTERING_CAVEAT)
    for section in range(1, 5):
        report_ttest(df, section)

    heading("Variance comparison: Experts vs Novices (logit credence, cluster bootstrap)")
    for section in range(1, 5):
        report_variance(df, section)

    heading("Variance comparison: Experts vs Novices (raw credence %, cluster bootstrap)")
    for section in range(1, 5):
        report_variance_raw(df, section)

    heading("Binomial test: proportion >50% vs <50% (excluding =50%)")
    lines.append(CLUSTERING_CAVEAT)
    for section in range(1, 5):
        report_binomial(df, section)

    heading("Means, Medians and IQRs of Credence in Correct Answer (integer %, participant-level)")
    for section in range(1, 5):
        report_median_iqr(df, section)

    heading("Means, Medians and IQRs of Credence in Correct Answer (integer %, observation-level)")
    for section in range(1, 5):
        report_median_iqr_obs(df, section)

    heading("Order bias: effect of Honest first vs Dishonest first on logit credence")
    for section in range(1, 5):
        report_order_bias(df, section)

    # ── PER DEBATE ────────────────────────────────────────────────────────────

    for debate in sorted(df["debate_name"].dropna().unique()):
        deb_df = df[df["debate_name"] == debate]

        heading(f"DEBATE: {debate}")

        lines.append("")
        lines.append("95% Credible Intervals for Logit Credence (with back-calculated probs)")
        for section in range(1, 5):
            report_logit_ci(deb_df, section)

        lines.append("")
        lines.append("Independent t-tests: Experts vs Novices (logit credence)")
        for section in range(1, 5):
            report_ttest(deb_df, section)

        lines.append("")
        lines.append("Variance comparison: Experts vs Novices (logit credence, cluster bootstrap)")
        for section in range(1, 5):
            report_variance(deb_df, section)

        lines.append("")
        lines.append("Variance comparison: Experts vs Novices (raw credence %, cluster bootstrap)")
        for section in range(1, 5):
            report_variance_raw(deb_df, section)

        lines.append("")
        lines.append("Binomial test: proportion >50% vs <50% (excluding =50%)")
        for section in range(1, 5):
            report_binomial(deb_df, section)

        lines.append("")
        lines.append("Means, Medians and IQRs of Credence in Correct Answer (integer %, participant-level)")
        for section in range(1, 5):
            report_median_iqr(deb_df, section)

        lines.append("")
        lines.append("Means, Medians and IQRs of Credence in Correct Answer (integer %, observation-level)")
        for section in range(1, 5):
            report_median_iqr_obs(deb_df, section)

        lines.append("")
        lines.append("Order bias: effect of Honest first vs Dishonest first on logit credence")
        for section in range(1, 5):
            report_order_bias(deb_df, section)

    report = "\n".join(lines).strip() + "\n"
    out = ANALYSIS_DIR / "credence_stats.txt"
    out.write_text(report, encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
