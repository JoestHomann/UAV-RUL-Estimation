"""Detect and plot constant or low-variation telemetry channels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from plotting_common import (
    DARK_BLUE,
    RED,
    is_effectively_constant,
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
        "Plot unique-value counts and numeric ranges to identify constant channels.",
        "constant_features",
    )
    args = parser.parse_args()
    channels = selected_channels(args)
    data = load_dataset(args.train_csv, channels)

    records = []
    for channel in channels:
        values = data[channel]
        value_range = float(values.max() - values.min())
        unique_count = int(values.nunique(dropna=False))
        exact_constant = unique_count <= 1 or value_range == 0
        records.append(
            {
                "channel": channel,
                "unique_count": unique_count,
                "range": value_range,
                "sd": float(values.std(ddof=1)),
                "constant": exact_constant,
                "effectively_constant": is_effectively_constant(values),
            }
        )
    summary = pd.DataFrame.from_records(records).set_index("channel")
    plotted = summary.sort_values(
        ["effectively_constant", "unique_count"], ascending=[False, True]
    )
    colors = [RED if value else DARK_BLUE for value in plotted["effectively_constant"]]
    y = np.arange(len(plotted))

    figure, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True, constrained_layout=True)
    axes[0].barh(y, plotted["unique_count"], color=colors)
    axes[0].set_xscale("log")
    axes[0].set(
        yticks=y,
        yticklabels=plotted.index,
        xlabel="Number of distinct values (log scale)",
        title="Unique-value count",
    )

    axes[1].barh(y, plotted["range"], color=colors)
    max_range = max(float(plotted["range"].max()), 1.0)
    axes[1].set_xscale("symlog", linthresh=max_range * 1e-8)
    axes[1].set(xlabel="Max − min (symlog scale)", title="Numeric range")
    for position, (_, row) in enumerate(plotted.iterrows()):
        if row["effectively_constant"]:
            axes[1].scatter(0, position, marker="x", color=RED, zorder=3)
            axes[1].annotate(
                "effectively constant", (0, position), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7, color=RED
            )
    for axis in axes:
        style_axis(axis)
    figure.suptitle("Constant-feature detection", fontsize=14)

    csv_path = save_csv(summary, args.output_dir, "constant_features.csv")
    figure_path = save_figure(
        figure, args.output_dir, "constant_features.png", args.dpi
    )
    print(f"Saved {figure_path}\nSaved {csv_path}")


if __name__ == "__main__":
    main()
