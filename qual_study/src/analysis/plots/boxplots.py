"""Boxplots of credence by expertise group, one subplot per section."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .._io import ANALYSIS_DIR, ensure_analysis_dir, load_combined


def main() -> None:
    ensure_analysis_dir()
    df = load_combined()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)

    for section, ax in zip(range(1, 5), axes.flat):
        col = f"section_{section}_credence_in_correct_answer_as_integer"
        experts = df.loc[df["group"] == "expert", col].dropna()
        novices = df.loc[df["group"] == "novice", col].dropna()

        bp = ax.boxplot(
            [experts, novices],
            tick_labels=["Some domain expertise", "No domain expertise"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("tab:blue")
        bp["boxes"][0].set_alpha(0.5)
        bp["boxes"][1].set_facecolor("tab:orange")
        bp["boxes"][1].set_alpha(0.5)

        ax.set_ylabel("Credence in correct answer (%)")
        ax.set_title(f"Section {section}")

    fig.suptitle("Credence in Correct Answer by Expertise and Section", fontsize=14)
    plt.tight_layout()

    out = ANALYSIS_DIR / "credence_boxplots.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
