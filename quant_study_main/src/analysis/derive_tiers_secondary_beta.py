"""Derive per-transcript tier classification from the secondary-beta fit.

The secondary-beta fit (fit_secondary_beta.py) has no single transcript
random intercept in the secondary-y sense. The strict analogue of the
secondary model's transcript random effect u_t is the Beta model's
per-transcript section slope u_{v,sec} (the random effect "section_is_4
| transcript" in the brms-style formula). Specifically, the per-transcript
expected logit-credence shift from section 1 to section 4 is
    beta_sec + u_{v,sec}[v],
and u_{v,sec}[v] is the per-transcript deviation from the global mean
update beta_sec. So we define

    delta_v = -u_{v,sec}[v]

mirroring derive_tiers.py's convention that "higher delta = harder
transcript" -- more positive delta means the transcript moves a typical
participant LESS toward the correct answer than the global mean.

(The global mean beta_sec is a constant shift across all 24 transcripts
and so does not affect the rank/tier assignment; we omit it so that the
delta_v vector is directly comparable to the secondary model's u_t-driven
delta_v in scale and centring.)

Within each draw, the 24 transcripts are ranked in ascending delta_v and
binned into 4 tiers of 6 (and, separately, into 3 tiers of 8 for the
exploratory 3-tier classification).

Outputs:
  output/tier-classification-secondary-beta.csv
  output/tier-summary-secondary-beta.csv
  output/question-pair-secondary-beta.csv
  output/tier3-classification-secondary-beta.csv
  output/tier3-summary-secondary-beta.csv
  output/tier3-question-pair-secondary-beta.csv
"""

import os
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=,mode=NUMBA")

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cleaned"
OUT_DIR = Path(os.environ.get(
    "ANALYSIS_OUT_DIR", str(Path(__file__).resolve().parent / "output")
))
OUT_DIR.mkdir(exist_ok=True)

FIT = OUT_DIR / "secondary-beta-fit.nc"
DATA = Path(os.environ.get(
    "ANALYSIS_DATA_CSV",
    str(DATA_DIR / "quant-1-official-study-analysis-ready.csv"),
))

N_TRANSCRIPTS = 24

# ---------------------------------------------------------------------------
# 1. Load posterior + transcript -> (question, domain) lookup.
# ---------------------------------------------------------------------------
post = az.from_netcdf(FIT).posterior

df = pd.read_csv(DATA, encoding="utf-8")
lookup = (
    df[["transcript", "question", "domain"]]
    .drop_duplicates()
    .sort_values("transcript")
    .reset_index(drop=True)
)
assert len(lookup) == N_TRANSCRIPTS, f"Expected 24 transcripts, got {len(lookup)}"

post_transcripts = [str(t) for t in post["transcript__factor_dim"].values]

# ---------------------------------------------------------------------------
# 2. Build delta_v = -u_{v,sec}[v] as an (n_draws, 24) array.
# ---------------------------------------------------------------------------
u_v_sec = post["section_is_4|transcript"].stack(sample=("chain", "draw")).values
# shape (24, n_draws)
n_draws = u_v_sec.shape[1]

delta = np.empty((n_draws, N_TRANSCRIPTS), dtype=float)
for vi, row in lookup.iterrows():
    v = str(row["transcript"])
    delta[:, vi] = -u_v_sec[post_transcripts.index(v)]

print(
    f"Computed delta_v for secondary-beta model: "
    f"{n_draws} draws x {N_TRANSCRIPTS} transcripts."
)


def write_tier_outputs(n_tiers, out_tr, out_tier, out_qp, header_suffix):
    tier_size = N_TRANSCRIPTS // n_tiers
    assert tier_size * n_tiers == N_TRANSCRIPTS

    # Rank per draw.
    order = np.argsort(delta, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks, order,
        np.broadcast_to(np.arange(N_TRANSCRIPTS), order.shape),
        axis=1,
    )
    tier_per_draw = ranks // tier_size + 1  # values in {1, ..., n_tiers}

    tier_prob = np.zeros((N_TRANSCRIPTS, n_tiers), dtype=float)
    for t in range(n_tiers):
        tier_prob[:, t] = (tier_per_draw == (t + 1)).mean(axis=0)
    modal_tier = tier_prob.argmax(axis=1) + 1

    delta_mean = delta.mean(axis=0)
    hdi = az.hdi(delta, hdi_prob=0.95)

    per_transcript_cols = {
        "transcript": lookup["transcript"].values,
        "question": lookup["question"].values,
        "domain": lookup["domain"].values,
        "delta_mean": delta_mean,
        "delta_hdi_lo": hdi[:, 0],
        "delta_hdi_hi": hdi[:, 1],
    }
    for t in range(n_tiers):
        per_transcript_cols[f"P_tier{t + 1}"] = tier_prob[:, t]
    per_transcript_cols["modal_tier"] = modal_tier
    per_transcript = (
        pd.DataFrame(per_transcript_cols)
        .sort_values("delta_mean")
        .reset_index(drop=True)
    )
    per_transcript.to_csv(out_tr, index=False, encoding="utf-8")

    tier_mean_per_draw = np.empty((n_draws, n_tiers), dtype=float)
    sorted_delta = np.take_along_axis(delta, order, axis=1)
    for t in range(n_tiers):
        tier_mean_per_draw[:, t] = sorted_delta[
            :, t * tier_size:(t + 1) * tier_size
        ].mean(axis=1)

    tier_mean_summary = pd.DataFrame({
        "tier": list(range(1, n_tiers + 1)),
        "mean": tier_mean_per_draw.mean(axis=0),
        "hdi_lo": az.hdi(tier_mean_per_draw, hdi_prob=0.95)[:, 0],
        "hdi_hi": az.hdi(tier_mean_per_draw, hdi_prob=0.95)[:, 1],
    })

    adj = np.column_stack([
        tier_mean_per_draw[:, t + 1] - tier_mean_per_draw[:, t]
        for t in range(n_tiers - 1)
    ])
    adj_summary = pd.DataFrame({
        "tier_pair": [f"{t + 1}-{t}" for t in range(1, n_tiers)],
        "diff_mean": adj.mean(axis=0),
        "diff_hdi_lo": az.hdi(adj, hdi_prob=0.95)[:, 0],
        "diff_hdi_hi": az.hdi(adj, hdi_prob=0.95)[:, 1],
    })

    with open(out_tier, "w", encoding="utf-8", newline="") as f:
        f.write("# Tier means\n")
        tier_mean_summary.to_csv(f, index=False)
        f.write("\n# Adjacent-tier differences\n")
        adj_summary.to_csv(f, index=False)

    question_rows = []
    for q, sub in lookup.groupby("question", sort=False):
        if len(sub) != 2:
            sys.exit(
                f"ERROR: question {q!r} has {len(sub)} transcripts; expected 2."
            )
        a_idx, b_idx = sub.index.tolist()
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
    question_summary.to_csv(out_qp, index=False, encoding="utf-8")

    print(f"\n[{header_suffix}] Per-transcript classification "
          f"(sorted easiest -> hardest):")
    prob_cols = [f"P_tier{t + 1}" for t in range(n_tiers)]
    display = per_transcript[
        ["transcript", "question", "domain", "delta_mean",
         "delta_hdi_lo", "delta_hdi_hi", "modal_tier", *prob_cols]
    ].copy()
    for col in ["delta_mean", "delta_hdi_lo", "delta_hdi_hi", *prob_cols]:
        display[col] = display[col].round(3)
    print(display.to_string(index=False))

    print(f"\n[{header_suffix}] Tier means:")
    print(tier_mean_summary.round(3).to_string(index=False))
    print(f"\n[{header_suffix}] Adjacent-tier differences:")
    print(adj_summary.round(3).to_string(index=False))

    print(f"\n[{header_suffix}] Per-question pairing "
          f"(sorted by P(same tier) desc):")
    qd = question_summary.copy()
    for col in ["P_same_tier", "within_q_diff_mean",
                "within_q_diff_hdi_lo", "within_q_diff_hdi_hi"]:
        qd[col] = qd[col].round(3)
    print(qd.to_string(index=False))

    print(f"\n[{header_suffix}] Wrote: "
          f"{out_tr.name}, {out_tier.name}, {out_qp.name}")


# 4-tier (preregistered).
write_tier_outputs(
    n_tiers=4,
    out_tr=OUT_DIR / "tier-classification-secondary-beta.csv",
    out_tier=OUT_DIR / "tier-summary-secondary-beta.csv",
    out_qp=OUT_DIR / "question-pair-secondary-beta.csv",
    header_suffix="4-tier",
)

# 3-tier (exploratory).
write_tier_outputs(
    n_tiers=3,
    out_tr=OUT_DIR / "tier3-classification-secondary-beta.csv",
    out_tier=OUT_DIR / "tier3-summary-secondary-beta.csv",
    out_qp=OUT_DIR / "tier3-question-pair-secondary-beta.csv",
    header_suffix="3-tier",
)
