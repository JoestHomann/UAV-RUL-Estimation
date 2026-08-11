"""Plot telemetry location and spread across predefined flight-cycle bands."""

from __future__ import annotations

import numpy as np
import pandas as pd

from plotting_common import (
    CYCLE_COLUMN,
    DARK_BLUE,
    LIGHT_BLUE,
    describe_values,
    hide_unused_axes,
    load_dataset,
    is_effectively_constant,
    make_parser,
    save_csv,
    save_figure,
    selected_channels,
    style_axis,
    subplot_grid,
)


AGE_LABELS = ["1–50", "51–100", "101–200", ">200"]
AGE_BINS = [0, 50, 100, 200, np.inf]


def main() -> None:
    parser = make_parser(
        "Plot telemetry statistics for the 1–50, 51–100, 101–200, and >200 age bands.",
        "age_band_statistics",
    )
    args = parser.parse_args()
    channels = selected_channels(args)
    data = load_dataset(args.train_csv, channels).copy()
    data["age_band"] = pd.cut(
        data[CYCLE_COLUMN], bins=AGE_BINS, labels=AGE_LABELS, right=True
    )

    records: list[dict[str, float | int | str]] = []
    for channel in channels:
        for band in AGE_LABELS:
            values = data.loc[data["age_band"] == band, channel]
            records.append(
                {"channel": channel, "age_band": band, **describe_values(values)}
            )
    summary = pd.DataFrame.from_records(records).set_index(["channel", "age_band"])
    band_counts = data["age_band"].value_counts(sort=False)
    tick_labels = [f"{band}\n(n={int(band_counts[band]):,})" for band in AGE_LABELS]

    x = np.arange(len(AGE_LABELS))
    figure, axes = subplot_grid(len(channels))
    for axis, channel in zip(axes.flat, channels):
        channel_summary = summary.loc[channel].reindex(AGE_LABELS)
        axis.fill_between(
            x,
            channel_summary["q25"].to_numpy(),
            channel_summary["q75"].to_numpy(),
            color=LIGHT_BLUE,
            alpha=0.22,
            label="IQR",
        )
        axis.plot(
            x,
            channel_summary["median"],
            color=DARK_BLUE,
            linewidth=2.0,
            marker="o",
            markersize=3,
            label="Median",
        )
        axis.plot(
            x,
            channel_summary["mean"],
            color=LIGHT_BLUE,
            linewidth=1.3,
            marker="o",
            markersize=2.5,
            label="Mean",
        )
        axis.set_title(channel, fontsize=10)
        axis.set_xticks(x, tick_labels, fontsize=6)
        if is_effectively_constant(data[channel]):
            center = float(data[channel].mean())
            padding = max(abs(center) * 0.01, 1.0)
            axis.set_ylim(center - padding, center + padding)
            axis.set_yticks([center], [f"{center:.7g}"])
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
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
        style_axis(axis)

    hide_unused_axes(axes, len(channels))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="outside lower center", ncol=3, frameon=False
    )
    figure.suptitle("Telemetry statistics by flight-cycle age band", fontsize=14)
    csv_path = save_csv(summary, args.output_dir, "age_band_statistics.csv")
    figure_path = save_figure(
        figure, args.output_dir, "age_band_statistics.png", args.dpi
    )
    print(f"Saved {figure_path}\nSaved {csv_path}")


if __name__ == "__main__":
    main()
