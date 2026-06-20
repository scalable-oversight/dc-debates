"""Run the analysis stage: stats reports and plots from the combined noids xlsx.

Reads:
    data/cleaned/qual-1-all-debates-for-grading_noids.xlsx
Writes (under data/cleaned/analysis/):
    credence_stats.txt
    credence_change_stats.txt
    variance_components_results.txt
    sample_size_with_uncertainty.{summary,fits}.txt
    sample_size_with_uncertainty.{full_results,bootstrap_draws}.csv
    credence_boxplots.png
    credence_histograms.png
    credence_swarmplots.png  credence_swarmplots_t1_t4.png
    credence_change_swarmplots.png
    by_debate/*.png

Run `run_pipeline.py` first to produce the noids xlsx.

Usage:
    python run_analysis.py
"""

from src.analysis.credence_change_stats import main as credence_change_stats_main
from src.analysis.credence_stats import main as credence_stats_main
from src.analysis.plots.boxplots import main as boxplots_main
from src.analysis.plots.change_swarmplots import main as change_swarmplots_main
from src.analysis.plots.histograms import main as histograms_main
from src.analysis.plots.swarmplots import main as swarmplots_main
from src.analysis.sample_size_with_uncertainty import main as sample_size_main
from src.analysis.variance_components import main as variance_components_main


def main() -> None:
    print("\n=== Plots: boxplots ===")
    boxplots_main()

    print("\n=== Plots: histograms ===")
    histograms_main()

    print("\n=== Plots: swarmplots ===")
    swarmplots_main()

    print("\n=== Plots: change swarmplots ===")
    change_swarmplots_main()

    print("\n=== Stats: credence ===")
    credence_stats_main()

    print("\n=== Stats: credence change ===")
    credence_change_stats_main()

    print("\n=== Variance components ===")
    variance_components_main(argv=[])

    print("\n=== Sample size with uncertainty ===")
    sample_size_main(argv=[])


if __name__ == "__main__":
    main()
