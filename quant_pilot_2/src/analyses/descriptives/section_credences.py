"""Swarmplot showing section_1–4_credence_in_correct_answer across sections,
plus descriptives with 95% CIs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True,
                   help="Path to cleaned quant-1-pilot-2-no-ids.xlsx")
    p.add_argument("--mapping", type=Path, required=False, default=None,
                   help="(Unused — accepted for pipeline uniformity)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory for output files")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    OUT = args.out_dir

    df = pd.read_excel(args.input, engine="openpyxl")

    CRED_COLS = [f"section_{s}_credence_in_correct_answer" for s in range(1, 5)]
    SECTION_LABELS = [f"Section {s}" for s in range(1, 5)]

    report = []
    for col, label in zip(CRED_COLS, SECTION_LABELS):
        s = df[col].dropna()
        n = len(s)
        m = s.mean()
        se = s.std() / np.sqrt(n)
        ci_lo, ci_hi = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
        report.append(
            f"{label} ({col})\n"
            f"  n       = {n}\n"
            f"  mean    = {m:.4f}\n"
            f"  95% CI  = [{ci_lo:.4f}, {ci_hi:.4f}]\n"
            f"  median  = {s.median():.4f}\n"
            f"  sd      = {s.std():.4f}\n"
            f"  min     = {s.min():.4f}\n"
            f"  max     = {s.max():.4f}"
        )

    txt = OUT / "section_credences_descriptives.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(report) + "\n")
    print(f"Wrote {txt}")

    long = df[CRED_COLS].melt(var_name="Section", value_name="Credence")
    long["Section"] = long["Section"].map(dict(zip(CRED_COLS, SECTION_LABELS)))
    long = long.dropna(subset=["Credence"])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.swarmplot(data=long, x="Section", y="Credence", ax=ax, alpha=0.6, size=4)

    for i, (col, label) in enumerate(zip(CRED_COLS, SECTION_LABELS)):
        s = df[col].dropna()
        m = s.mean()
        se = s.std() / np.sqrt(len(s))
        ci_lo, ci_hi = stats.t.interval(0.95, df=len(s) - 1, loc=m, scale=se)
        ax.plot(i, m, "D", color="red", markersize=7, zorder=5)
        ax.vlines(i, ci_lo, ci_hi, color="red", linewidth=2, zorder=5)

    ax.set_ylabel("Credence in Correct Answer")
    ax.set_title("Credence in Correct Answer by Section")
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="chance (0.5)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "section_credences_swarmplot.png", dpi=150)
    print(f"Saved section_credences_swarmplot.png")
    plt.close()


if __name__ == "__main__":
    main()
