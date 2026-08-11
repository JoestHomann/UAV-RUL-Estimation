"""Plot count, location, spread, quantiles, and extremes for each channel."""

from __future__ import annotations

from matplotlib.lines import Line2D

from plotting_common import (
    DARK_BLUE,
    GRAY,
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
        "Plot full descriptive statistics for each telemetry channel.",
        "descriptive_statistics",
    )
    args = parser.parse_args()
    channels = selected_channels(args)
    data = load_dataset(args.train_csv, channels)
    summary = descriptive_table(data, channels)

    figure, axes = subplot_grid(len(channels))
    for axis, channel in zip(axes.flat, channels):
        row = summary.loc[channel]
        if row["effectively_constant"]:
            center = float(row["mean"])
            padding = max(abs(center) * 0.01, 1.0)
            axis.axvline(center, color=DARK_BLUE, linewidth=2)
            axis.scatter(center, 0, marker="D", s=22, color="black", zorder=4)
            axis.set_xlim(center - padding, center + padding)
            axis.set_xticks([center], [f"{center:.7g}"])
            axis.text(
                0.98,
                0.88,
                "effectively constant",
                transform=axis.transAxes,
                fontsize=7,
                ha="right",
                va="top",
                color=GRAY,
            )
        else:
            axis.hlines(0, row["min"], row["max"], color=GRAY, linewidth=0.8)
            axis.hlines(0, row["q05"], row["q95"], color=LIGHT_BLUE, linewidth=4)
            axis.hlines(0, row["q25"], row["q75"], color=DARK_BLUE, linewidth=10)
            axis.scatter(row["median"], 0, marker="|", s=140, color="white", zorder=3)
            axis.scatter(row["mean"], 0, marker="D", s=22, color="black", zorder=4)
            axis.scatter(
                [row["min"], row["max"]], [0, 0], marker="x", s=22, color=GRAY, zorder=4
            )
        axis.set_title(channel, fontsize=10)
        axis.set_yticks([])
        axis.text(
            0.02,
            0.88,
            f"n={int(row['count']):,}  SD={row['sd']:.4g}  IQR={row['iqr']:.4g}",
            transform=axis.transAxes,
            fontsize=7,
            va="top",
        )
        style_axis(axis)
        if not row["effectively_constant"]:
            axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4), useOffset=False)

    hide_unused_axes(axes, len(channels))
    legend = [
        Line2D([0], [0], color=GRAY, lw=1, marker="x", label="Min–max"),
        Line2D([0], [0], color=LIGHT_BLUE, lw=4, label="P05–P95"),
        Line2D([0], [0], color=DARK_BLUE, lw=8, label="IQR (P25–P75)"),
        Line2D([0], [0], color="white", marker="|", markeredgecolor="black", lw=0, label="Median"),
        Line2D([0], [0], color="black", marker="D", lw=0, label="Mean"),
    ]
    figure.legend(legend, [item.get_label() for item in legend], loc="outside lower center", ncol=5, frameon=False)
    figure.suptitle("Telemetry descriptive statistics", fontsize=14)

    csv_path = save_csv(summary, args.output_dir, "descriptive_statistics.csv")
    figure_path = save_figure(
        figure, args.output_dir, "descriptive_statistics.png", args.dpi
    )
    print(f"Saved {figure_path}\nSaved {csv_path}")


if __name__ == "__main__":
    main()
