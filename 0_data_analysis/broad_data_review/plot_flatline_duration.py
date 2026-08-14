"""Measure and plot exact consecutive flatline runs within UAV histories."""

from __future__ import annotations

import numpy as np
import pandas as pd

from plotting_common import (
    DARK_BLUE,
    ID_COLUMN,
    LIGHT_BLUE,
    ORANGE,
    YELLOW,
    load_dataset,
    make_parser,
    save_csv,
    save_figure,
    selected_channels,
    style_axis,
)
import matplotlib.pyplot as plt


def consecutive_run_lengths(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=int)
    boundaries = np.flatnonzero(
        np.concatenate(([True], values[1:] != values[:-1], [True]))
    )
    return np.diff(boundaries)


def main() -> None:
    parser = make_parser(
        "Plot exact flatline durations within each UAV history.",
        "flatline_duration",
    )
    parser.add_argument(
        "--minimum-run",
        type=int,
        default=5,
        help="Minimum equal-value run considered a flatline (default: 5 cycles).",
    )
    args = parser.parse_args()
    if args.minimum_run < 2:
        parser.error("--minimum-run must be at least 2")

    channels = selected_channels(args)
    data = load_dataset(args.train_csv, channels)
    records: list[dict[str, int | str]] = []
    for uav_id, history in data.groupby(ID_COLUMN, sort=True):
        for channel in channels:
            runs = consecutive_run_lengths(history[channel].to_numpy())
            flatline_rows = int(runs[runs >= args.minimum_run].sum())
            records.append(
                {
                    "uav_id": uav_id,
                    "channel": channel,
                    "longest_run": int(runs.max()),
                    "flatline_rows": flatline_rows,
                    "row_count": int(len(history)),
                }
            )
    by_uav = pd.DataFrame.from_records(records)

    summary_records = []
    for channel, channel_rows in by_uav.groupby("channel", sort=False):
        summary_records.append(
            {
                "channel": channel,
                "minimum_run": args.minimum_run,
                "median_longest_run": float(channel_rows["longest_run"].median()),
                "maximum_run": int(channel_rows["longest_run"].max()),
                "uavs_with_flatline_percent": float(
                    100 * (channel_rows["longest_run"] >= args.minimum_run).mean()
                ),
                "rows_in_flatlines_percent": float(
                    100
                    * channel_rows["flatline_rows"].sum()
                    / channel_rows["row_count"].sum()
                ),
            }
        )
    summary = pd.DataFrame.from_records(summary_records).set_index("channel")
    plotted = summary.sort_values("rows_in_flatlines_percent")
    y = np.arange(len(plotted))
    offset = 0.19

    figure, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=True, constrained_layout=True)
    axes[0].barh(
        y - offset,
        plotted["median_longest_run"],
        height=0.36,
        color=LIGHT_BLUE,
        label="Median per-UAV maximum",
    )
    axes[0].barh(
        y + offset,
        plotted["maximum_run"],
        height=0.36,
        color=DARK_BLUE,
        label="Overall maximum",
    )
    axes[0].set_xscale("log")
    axes[0].set(
        yticks=y,
        yticklabels=plotted.index,
        xlabel="Consecutive identical cycles (log scale)",
        title="Flatline duration",
    )
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].barh(
        y - offset,
        plotted["uavs_with_flatline_percent"],
        height=0.36,
        color=YELLOW,
        label=f"UAVs with run ≥ {args.minimum_run}",
    )
    axes[1].barh(
        y + offset,
        plotted["rows_in_flatlines_percent"],
        height=0.36,
        color=ORANGE,
        label=f"Rows inside runs ≥ {args.minimum_run}",
    )
    axes[1].set(
        xlim=(0, 100),
        xlabel="Percent (%)",
        title="Flatline prevalence",
    )
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        style_axis(axis)
    figure.suptitle(
        f"Exact-value flatlines within UAV histories (minimum run: {args.minimum_run})",
        fontsize=14,
    )

    by_uav_path = save_csv(
        by_uav.set_index(["channel", "uav_id"]),
        args.output_dir,
        "flatline_by_uav.csv",
    )
    summary_path = save_csv(summary, args.output_dir, "flatline_summary.csv")
    figure_path = save_figure(
        figure, args.output_dir, "flatline_duration.png", args.dpi
    )
    print(f"Saved {figure_path}\nSaved {summary_path}\nSaved {by_uav_path}")


if __name__ == "__main__":
    main()
