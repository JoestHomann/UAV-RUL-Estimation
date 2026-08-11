"""Plot a histogram and compact box plot for every telemetry channel."""

from __future__ import annotations

import numpy as np

from plotting_common import (
    DARK_BLUE,
    LIGHT_BLUE,
    hide_unused_axes,
    is_effectively_constant,
    load_dataset,
    make_parser,
    save_figure,
    selected_channels,
    style_axis,
    subplot_grid,
)


def histogram_edges(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    if is_effectively_constant(values):
        padding = max(abs(minimum) * 0.01, 0.5)
        return np.array([minimum - padding, maximum + padding])

    edges = np.histogram_bin_edges(values, bins="fd")
    if len(edges) - 1 > 60:
        return np.linspace(minimum, maximum, 61)
    if len(edges) - 1 < 10:
        return np.linspace(minimum, maximum, 11)
    return edges


def main() -> None:
    parser = make_parser(
        "Plot row-level histograms and box plots for every telemetry channel.",
        "histograms_boxplots",
    )
    args = parser.parse_args()
    channels = selected_channels(args)
    data = load_dataset(args.train_csv, channels)

    figure, axes = subplot_grid(len(channels), height=3.1)
    for axis, channel in zip(axes.flat, channels):
        values = data[channel].to_numpy(dtype=float)
        axis.hist(values, bins=histogram_edges(values), color=DARK_BLUE, alpha=0.78)
        axis.set_title(channel, fontsize=10)
        axis.set_ylabel("Rows", fontsize=8)
        style_axis(axis)
        if is_effectively_constant(values):
            center = float(values.mean())
            axis.text(
                0.5,
                0.60,
                "effectively constant",
                transform=axis.transAxes,
                ha="center",
                fontsize=7,
            )
            axis.set_xticks([center], [f"{center:.7g}"])
        else:
            axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4), useOffset=False)

        inset = axis.inset_axes([0.08, 0.74, 0.84, 0.18])
        inset.boxplot(
            values,
            vert=False,
            widths=0.55,
            showfliers=True,
            patch_artist=True,
            flierprops={"marker": ".", "markersize": 1, "alpha": 0.25},
            boxprops={"facecolor": LIGHT_BLUE, "edgecolor": DARK_BLUE},
            medianprops={"color": DARK_BLUE, "linewidth": 1.5},
            whiskerprops={"color": DARK_BLUE},
            capprops={"color": DARK_BLUE},
        )
        inset.set_xticks([])
        inset.set_yticks([])
        inset.patch.set_alpha(0.78)

    hide_unused_axes(axes, len(channels))
    figure.suptitle("Row-level telemetry histograms with box-plot insets", fontsize=14)
    figure_path = save_figure(
        figure, args.output_dir, "histograms_boxplots.png", args.dpi
    )
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
