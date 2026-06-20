"""Overlapping histograms of credence by expertise group, one subplot per section."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .._io import ANALYSIS_DIR, ensure_analysis_dir, load_combined


def main() -> None:
    ensure_analysis_dir()
    df = load_combined()

    bins = range(0, 110, 10)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)

    for section, ax in zip(range(1, 5), axes.flat):
        col = f"section_{section}_credence_in_correct_answer_as_integer"
        experts = df.loc[df["group"] == "expert", col].dropna()
        novices = df.loc[df["group"] == "novice", col].dropna()

        ax.hist(experts, bins=bins, alpha=0.5, label="Some domain expertise", edgecolor="black")
        ax.hist(novices, bins=bins, alpha=0.5, label="No domain expertise", edgecolor="black")
        ax.set_xlabel("Credence in correct answer (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"Section {section}")
        ax.legend()

    fig.suptitle("Credence in Correct Answer by Expertise and Section", fontsize=14)
    plt.tight_layout()

    out = ANALYSIS_DIR / "credence_histograms.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
