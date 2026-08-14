"""Plot row-weighted distribution diagnostics for all recorded flight cycles."""

from __future__ import annotations

import numpy as np

from plotting_common import (
    DARK_BLUE,
    LIGHT_BLUE,
    descriptive_table,
    load_dataset,
    make_parser,
    save_csv,
    save_figure,
    selected_channels,
    style_axis,
)
import matplotlib.pyplot as plt


def main() -> None:
    parser = make_parser(
        "Plot row-level telemetry statistics; every recorded cycle has equal weight.",
        "row_level_statistics",
    )
    args = parser.parse_args()
    channels = selected_channels(args)
    data = load_dataset(args.train_csv, channels)
    summary = descriptive_table(data, channels)

    nonzero_sd = summary["sd"].replace(0, np.nan)
    summary["mean_median_gap_sd"] = (
        (summary["mean"] - summary["median"]) / nonzero_sd
    ).fillna(0.0)
    summary["iqr_to_sd"] = (summary["iqr"] / nonzero_sd).fillna(0.0)
    summary.loc[
        summary["effectively_constant"], ["mean_median_gap_sd", "iqr_to_sd"]
    ] = 0.0
    order = summary["mean_median_gap_sd"].abs().sort_values().index
    plotted = summary.loc[order]

    figure, axes = plt.subplots(
        1, 2, figsize=(13, 9), sharey=True, constrained_layout=True
    )
    y = np.arange(len(plotted))
    gap_colors = np.where(plotted["mean_median_gap_sd"] >= 0, DARK_BLUE, LIGHT_BLUE)
    axes[0].barh(y, plotted["mean_median_gap_sd"], color=gap_colors)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set(
        yticks=y,
        yticklabels=plotted.index,
        xlabel="(mean − median) / SD",
        title="Standardized mean–median gap",
    )
    axes[1].barh(y, plotted["iqr_to_sd"], color=LIGHT_BLUE)
    axes[1].set(xlabel="IQR / SD", title="Robust spread relative to SD")
    for axis in axes:
        style_axis(axis)

    figure.suptitle(
        "Row-level telemetry diagnostics (longer UAV histories contribute more rows)",
        fontsize=14,
    )
    csv_path = save_csv(summary, args.output_dir, "row_level_statistics.csv")
    figure_path = save_figure(
        figure, args.output_dir, "row_level_statistics.png", args.dpi
    )
    print(f"Saved {figure_path}\nSaved {csv_path}")


if __name__ == "__main__":
    main()
