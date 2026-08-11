"""Plot the distribution of per-UAV telemetry means with equal UAV weight."""

from __future__ import annotations

import numpy as np

from plotting_common import (
    DARK_BLUE,
    ID_COLUMN,
    LIGHT_BLUE,
    descriptive_table,
    hide_unused_axes,
    load_dataset,
    make_parser,
    save_csv,
    save_figure,
    selected_channels,
    style_axis,
    subplot_grid,
)


def main() -> None:
    parser = make_parser(
        "Plot per-UAV telemetry summaries so every UAV has equal weight.",
        "uav_level_statistics",
    )
    args = parser.parse_args()
    channels = selected_channels(args)
    data = load_dataset(args.train_csv, channels)
    uav_means = data.groupby(ID_COLUMN, sort=True)[channels].mean()
    summary = descriptive_table(uav_means, channels)

    random = np.random.default_rng(42)
    figure, axes = subplot_grid(len(channels))
    for axis, channel in zip(axes.flat, channels):
        values = uav_means[channel].to_numpy()
        is_constant = bool(summary.loc[channel, "effectively_constant"])
        jitter = random.normal(1.0, 0.035, size=len(values))
        axis.scatter(values, jitter, s=8, color=LIGHT_BLUE, alpha=0.45, zorder=1)
        axis.boxplot(
            values,
            vert=False,
            widths=0.35,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "none", "edgecolor": DARK_BLUE},
            medianprops={"color": DARK_BLUE, "linewidth": 2},
            whiskerprops={"color": DARK_BLUE},
            capprops={"color": DARK_BLUE},
        )
        axis.scatter(values.mean(), 1, marker="D", s=20, color="black", zorder=3)
        axis.set_title(channel, fontsize=10)
        axis.set_yticks([])
        axis.text(
            0.02,
            0.90,
            f"{len(values)} UAVs",
            transform=axis.transAxes,
            fontsize=7,
            va="top",
        )
        style_axis(axis)
        if is_constant:
            center = float(values.mean())
            padding = max(abs(center) * 0.01, 1.0)
            axis.set_xlim(center - padding, center + padding)
            axis.set_xticks([center], [f"{center:.7g}"])
            axis.text(
                0.98,
                0.90,
                "effectively constant",
                transform=axis.transAxes,
                fontsize=7,
                ha="right",
                va="top",
            )
        else:
            axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4), useOffset=False)

    hide_unused_axes(axes, len(channels))
    figure.suptitle("Distribution of per-UAV mean telemetry (equal UAV weight)", fontsize=14)
    means_path = save_csv(uav_means, args.output_dir, "per_uav_means.csv")
    summary_path = save_csv(summary, args.output_dir, "uav_level_statistics.csv")
    figure_path = save_figure(
        figure, args.output_dir, "uav_level_statistics.png", args.dpi
    )
    print(f"Saved {figure_path}\nSaved {means_path}\nSaved {summary_path}")


if __name__ == "__main__":
    main()
