"""Derive per-transcript difficulty estimates and the four-tier classification.

Implements the 'Deriving the tier classification' section of CF_analysis_plan.docx.
For each posterior draw and each transcript v, computes:

    e_v = beta_domain[d(v)] + u_question[q(v)] + u_transcript[v]
    delta_v = -e_v                  # higher = harder

(beta_domain = 0 for the reference domain, brainteasers.)

Within each draw, the 24 transcripts are ranked in ascending delta_v and binned:
    Tier 1 (easiest): ranks  1-6
    Tier 2          : ranks  7-12
    Tier 3          : ranks 13-18
    Tier 4 (hardest): ranks 19-24

Outputs (per outcome: primary / secondary):
  data/tier-classification-<outcome>.csv   per-transcript: delta_v mean+HDI,
                                           tier probabilities, modal tier
  data/tier-summary-<outcome>.csv          tier means + adjacent-tier
                                           differences
  data/question-pair-<outcome>.csv         per-question: P(same tier) and
                                           within-question delta_v difference

Usage:  python3 derive_tiers.py [primary|secondary]   (default: primary)
"""

import os
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=,mode=NUMBA")

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

# Analysis-ready CSV lives in ../data/; fits and derived artifacts in ./output/.
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cleaned"
OUT_DIR = Path(os.environ.get("ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent / "output")))
OUT_DIR.mkdir(exist_ok=True)

# Map the chosen outcome label to the fit file produced by fit_*_model.py.
# 'secondary-t' is the Student-t residual refit produced by
# fit_secondary_student_t.py (added per the plan's alternative-residual clause).
# 'secondary-sign' is the exploratory Bernoulli fit on the sign of y_secondary
# produced by fit_secondary_sign.py.
outcome = sys.argv[1] if len(sys.argv) > 1 else "primary"
if outcome not in {"primary", "secondary", "secondary-t", "secondary-sign"}:
    sys.exit(
        f"Unknown outcome '{outcome}'. "
        f"Use 'primary', 'secondary', 'secondary-t', or 'secondary-sign'."
    )

FIT = OUT_DIR / f"{outcome}-fit.nc"
DATA = Path(os.environ.get("ANALYSIS_DATA_CSV", str(DATA_DIR / "quant-1-official-study-analysis-ready.csv")))
OUT_TR = OUT_DIR / f"tier-classification-{outcome}.csv"
OUT_TIER = OUT_DIR / f"tier-summary-{outcome}.csv"
OUT_QP = OUT_DIR / f"question-pair-{outcome}.csv"

DOMAIN_VAR = "C(domain, Treatment('brainteasers'))"
DOMAIN_DIM = f"{DOMAIN_VAR}_dim"

# ---------------------------------------------------------------------------
# 1. Load fit + the transcript -> (question, domain) lookup.
# ---------------------------------------------------------------------------
idata = az.from_netcdf(FIT)
post = idata.posterior

df = pd.read_csv(DATA, encoding="utf-8")
# Each transcript appears once per participant; the (question, domain) is a
# property of the transcript itself, so deduplicate on transcript.
lookup = (
    df[["transcript", "question", "domain"]]
    .drop_duplicates()
    .sort_values("transcript")
    .reset_index(drop=True)
)
assert len(lookup) == 24, f"Expected 24 transcripts, got {len(lookup)}"

# Posterior coordinate orderings (so we can index by name, not by position).
post_transcripts = [str(t) for t in post["transcript__factor_dim"].values]
post_questions = [str(q) for q in post["question__factor_dim"].values]
post_domains = [str(d) for d in post[DOMAIN_DIM].values]

# ---------------------------------------------------------------------------
# 2. Build delta_v as an (n_draws, 24) array.
# ---------------------------------------------------------------------------
# Stack chain*draw -> a single sample axis so ranks-per-draw are easy.
u_transcript = post["1|transcript"].stack(sample=("chain", "draw")).values  # (24, n_draws)
u_question = post["1|question"].stack(sample=("chain", "draw")).values      # (12, n_draws)
beta_domain = post[DOMAIN_VAR].stack(sample=("chain", "draw")).values       # (2, n_draws)

n_draws = u_transcript.shape[1]
delta = np.empty((n_draws, 24), dtype=float)  # (draw, transcript_v)
for vi, row in lookup.iterrows():
    v = str(row["transcript"])
    q = row["question"]
    d = row["domain"]

    u_t = u_transcript[post_transcripts.index(v)]
    u_q = u_question[post_questions.index(q)]
    if d in post_domains:
        b_d = beta_domain[post_domains.index(d)]
    else:
        # brainteasers is the reference level and has no posterior coefficient
        b_d = np.zeros(n_draws)
    e_v = b_d + u_q + u_t
    delta[:, vi] = -e_v

print(f"Computed delta_v for {outcome} model: {n_draws} draws x 24 transcripts.")

# ---------------------------------------------------------------------------
# 3. Tier assignment per draw, then posterior tier-membership matrix.
# ---------------------------------------------------------------------------
# Ascending rank in each row -> 0-23. Floor-divide by 6 -> 0..3 -> tier 1..4.
order = np.argsort(delta, axis=1)
ranks = np.empty_like(order)
np.put_along_axis(
    ranks,
    order,
    np.broadcast_to(np.arange(24), order.shape),
    axis=1,
)
tier_per_draw = ranks // 6 + 1  # (n_draws, 24) values in {1,2,3,4}

# 24x4 probability matrix.
tier_prob = np.zeros((24, 4), dtype=float)
for t in range(4):
    tier_prob[:, t] = (tier_per_draw == (t + 1)).mean(axis=0)
modal_tier = tier_prob.argmax(axis=1) + 1

# ---------------------------------------------------------------------------
# 4. Per-transcript summary table.
# ---------------------------------------------------------------------------
delta_mean = delta.mean(axis=0)
hdi = az.hdi(delta, hdi_prob=0.95)  # shape (24, 2)

per_transcript = pd.DataFrame({
    "transcript": lookup["transcript"].values,
    "question": lookup["question"].values,
    "domain": lookup["domain"].values,
    "delta_mean": delta_mean,
    "delta_hdi_lo": hdi[:, 0],
    "delta_hdi_hi": hdi[:, 1],
    "P_tier1": tier_prob[:, 0],
    "P_tier2": tier_prob[:, 1],
    "P_tier3": tier_prob[:, 2],
    "P_tier4": tier_prob[:, 3],
    "modal_tier": modal_tier,
}).sort_values("delta_mean").reset_index(drop=True)
per_transcript.to_csv(OUT_TR, index=False, encoding="utf-8")

# ---------------------------------------------------------------------------
# 5. Tier means + adjacent-tier differences (computed per draw, then summarised).
# ---------------------------------------------------------------------------
# For each draw, mean delta over the 6 transcripts assigned to each rank tier.
tier_mean_per_draw = np.empty((n_draws, 4), dtype=float)
sorted_delta = np.take_along_axis(delta, order, axis=1)  # (n_draws, 24), ascending
for t in range(4):
    tier_mean_per_draw[:, t] = sorted_delta[:, t * 6:(t + 1) * 6].mean(axis=1)

tier_mean_summary = pd.DataFrame({
    "tier": [1, 2, 3, 4],
    "mean": tier_mean_per_draw.mean(axis=0),
    "hdi_lo": az.hdi(tier_mean_per_draw, hdi_prob=0.95)[:, 0],
    "hdi_hi": az.hdi(tier_mean_per_draw, hdi_prob=0.95)[:, 1],
})

adj = np.column_stack([
    tier_mean_per_draw[:, 1] - tier_mean_per_draw[:, 0],
    tier_mean_per_draw[:, 2] - tier_mean_per_draw[:, 1],
    tier_mean_per_draw[:, 3] - tier_mean_per_draw[:, 2],
])
adj_summary = pd.DataFrame({
    "tier_pair": ["2-1", "3-2", "4-3"],
    "diff_mean": adj.mean(axis=0),
    "diff_hdi_lo": az.hdi(adj, hdi_prob=0.95)[:, 0],
    "diff_hdi_hi": az.hdi(adj, hdi_prob=0.95)[:, 1],
})

with open(OUT_TIER, "w", encoding="utf-8", newline="") as f:
    f.write("# Tier means\n")
    tier_mean_summary.to_csv(f, index=False)
    f.write("\n# Adjacent-tier differences\n")
    adj_summary.to_csv(f, index=False)

# ---------------------------------------------------------------------------
# 6. Per-question: P(same tier) and within-question delta_v difference.
# ---------------------------------------------------------------------------
# For paired transcripts a and b in the same question:
#   delta_a - delta_b reduces to -(u_t[a] - u_t[b])  (shared domain & question).
question_rows = []
for q, sub in lookup.groupby("question", sort=False):
    if len(sub) != 2:
        sys.exit(
            f"ERROR: question {q!r} has {len(sub)} transcripts; expected 2."
        )
    a_idx, b_idx = sub.index.tolist()
    # Same-tier probability across draws.
    p_same = (tier_per_draw[:, a_idx] == tier_per_draw[:, b_idx]).mean()
    diff = delta[:, a_idx] - delta[:, b_idx]
    diff_hdi = az.hdi(diff, hdi_prob=0.95)
    question_rows.append({
        "question": q,
        "transcript_a": int(sub.iloc[0]["transcript"]),
        "transcript_b": int(sub.iloc[1]["transcript"]),
        "P_same_tier": p_same,
        "within_q_diff_mean": diff.mean(),
        "within_q_diff_hdi_lo": diff_hdi[0],
        "within_q_diff_hdi_hi": diff_hdi[1],
    })
question_summary = (
    pd.DataFrame(question_rows)
    .sort_values("P_same_tier", ascending=False)
    .reset_index(drop=True)
)
question_summary.to_csv(OUT_QP, index=False, encoding="utf-8")

# ---------------------------------------------------------------------------
# 7. Console summary.
# ---------------------------------------------------------------------------
print(f"\nPer-transcript classification (sorted easiest -> hardest):")
display = per_transcript[[
    "transcript", "question", "domain", "delta_mean",
    "delta_hdi_lo", "delta_hdi_hi", "modal_tier",
    "P_tier1", "P_tier2", "P_tier3", "P_tier4",
]].copy()
for col in ["delta_mean", "delta_hdi_lo", "delta_hdi_hi",
            "P_tier1", "P_tier2", "P_tier3", "P_tier4"]:
    display[col] = display[col].round(3)
print(display.to_string(index=False))

print(f"\nTier means:")
print(tier_mean_summary.round(3).to_string(index=False))
print(f"\nAdjacent-tier differences:")
print(adj_summary.round(3).to_string(index=False))

print(f"\nPer-question pairing (sorted by P(same tier) desc):")
qd = question_summary.copy()
for col in ["P_same_tier", "within_q_diff_mean",
            "within_q_diff_hdi_lo", "within_q_diff_hdi_hi"]:
    qd[col] = qd[col].round(3)
print(qd.to_string(index=False))

print(f"\nWrote: {OUT_TR.name}, {OUT_TIER.name}, {OUT_QP.name}")
