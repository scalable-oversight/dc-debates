"""Swarm plots of logit credence change (section 4 - section 1) by expertise group.

Produces:
  - An overall plot in data/cleaned/analysis/
  - One plot per debate in data/cleaned/analysis/by_debate/
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from .._io import ANALYSIS_DIR, BY_DEBATE_DIR, ensure_analysis_dir, load_combined, safe_filename

GROUP_LABELS = {"expert": "Some domain expertise", "novice": "No domain expertise"}
HUE_ORDER = ["Some domain expertise", "No domain expertise"]
PALETTE = {"Some domain expertise": "tab:blue", "No domain expertise": "tab:orange"}


def make_swarmplot(data, title, out_path, label_points=False):
    fig, ax = plt.subplots(figsize=(8, 6))

    subset = data[["group_label", "participant_id", "logit_change"]].dropna(subset=["logit_change"])
    sns.swarmplot(
        data=subset, x="group_label", y="logit_change", ax=ax,
        order=HUE_ORDER, hue="group_label", hue_order=HUE_ORDER,
        palette=PALETTE, alpha=0.6, size=4,
    )

    if label_points:
        collections = [c for c in ax.collections
                       if hasattr(c, "get_offsets") and len(c.get_offsets())]
        for coll, hue_val in zip(collections, HUE_ORDER):
            group_rows = subset[subset["group_label"] == hue_val]
            offsets = coll.get_offsets()
            for (xy, (_, row)) in zip(offsets, group_rows.iterrows()):
                pid = int(row["participant_id"])
                ax.annotate(
                    str(pid), xy, fontsize=5,
                    xytext=(3, 1), textcoords="offset points",
                    alpha=0.8,
                )

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("")
    ax.set_ylabel("Logit credence change (section 4 − section 1)")
    ax.set_title(title)
    legend = ax.get_legend()
    if legend:
        legend.remove()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")


def main() -> None:
    ensure_analysis_dir(by_debate=True)
    df = load_combined()
    df["logit_change"] = (
        df["logit_section_4_credence_in_correct_answer"]
        - df["logit_section_1_credence_in_correct_answer"]
    )
    df["group_label"] = df["group"].map(GROUP_LABELS)

    make_swarmplot(df, "Logit Credence Change (Section 4 − Section 1) by Expertise",
                   ANALYSIS_DIR / "credence_change_swarmplots.png")

    for debate_name in sorted(df["debate_name"].dropna().unique()):
        deb_df = df[df["debate_name"] == debate_name]
        make_swarmplot(deb_df,
                       f"Logit Credence Change (Section 4 − Section 1) — {debate_name}",
                       BY_DEBATE_DIR / f"credence_change_swarmplots_{safe_filename(debate_name)}.png",
                       label_points=True)


if __name__ == "__main__":
    main()
