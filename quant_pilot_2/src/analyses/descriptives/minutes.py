"""Histogram, swarmplot & descriptives of Minutes Spent and
total_minutes_evaluating_debates.
"""

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


COLS = ["Minutes Spent", "total_minutes_evaluating_debates"]
LABELS = {
    "Minutes Spent": "Minutes Spent (full survey)",
    "total_minutes_evaluating_debates": "Total Minutes Evaluating Debates",
}


def descriptives(series, label):
    s = series.dropna()
    lines = [
        f"  n       = {len(s)}",
        f"  mean    = {s.mean():.2f}",
        f"  median  = {s.median():.2f}",
        f"  sd      = {s.std():.2f}",
        f"  min     = {s.min():.2f}",
        f"  max     = {s.max():.2f}",
    ]
    return f"{label}\n" + "\n".join(lines)


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

    report = []
    for col in COLS:
        report.append(descriptives(df[col], LABELS[col]))

    txt = OUT / "minutes_descriptives.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(report) + "\n")
    print(f"Wrote {txt}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col in zip(axes, COLS):
        s = df[col].dropna()
        ax.hist(s, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(s.mean(), color="red", linestyle="--", label=f"mean = {s.mean():.1f}")
        ax.axvline(s.median(), color="blue", linestyle=":", label=f"median = {s.median():.1f}")
        ax.set_xlabel(LABELS[col])
        ax.set_ylabel("Count")
        ax.set_title(LABELS[col])
        ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "minutes_histograms.png", dpi=150)
    print(f"Saved minutes_histograms.png")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    for ax, col in zip(axes, COLS):
        sns.swarmplot(y=df[col].dropna(), ax=ax, alpha=0.6, size=4)
        ax.set_ylabel(LABELS[col])
        ax.set_title(LABELS[col])
    plt.tight_layout()
    fig.savefig(OUT / "minutes_swarmplots.png", dpi=150)
    print(f"Saved minutes_swarmplots.png")
    plt.close()


if __name__ == "__main__":
    main()
