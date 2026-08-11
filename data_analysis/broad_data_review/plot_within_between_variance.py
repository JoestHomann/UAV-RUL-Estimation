"""Decompose each telemetry channel into within- and between-UAV variance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from plotting_common import (
    DARK_BLUE,
    GRAY,
    ID_COLUMN,
    LIGHT_BLUE,
    is_effectively_constant,
    load_dataset,
    make_parser,
    save_csv,
    save_figure,
    selected_channels,
    style_axis,
)
import matplotlib.pyplot as plt


def variance_decomposition(
    data: pd.DataFrame, channels: list[str]
) -> pd.DataFrame:
    records: list[dict[str, float | str | bool]] = []
    for channel in channels:
        values = data[channel].to_numpy(dtype=float)
        group_means = data.groupby(ID_COLUMN)[channel].transform("mean").to_numpy()
        group_sizes = data.groupby(ID_COLUMN)[channel].size()
        means_by_uav = data.groupby(ID_COLUMN)[channel].mean()
        grand_mean = float(values.mean())

        within_ss = float(np.square(values - group_means).sum())
        between_ss = float(
            (group_sizes * np.square(means_by_uav - grand_mean)).sum()
        )
        total_ss = within_ss + between_ss
        denominator = max(len(values) - 1, 1)
        is_constant = is_effectively_constant(values)
        within_share = within_ss / total_ss if total_ss > 0 and not is_constant else 0.0
        between_share = between_ss / total_ss if total_ss > 0 and not is_constant else 0.0
        records.append(
            {
                "channel": channel,
                "total_variance": total_ss / denominator,
                "within_uav_variance": within_ss / denominator,
                "between_uav_variance": between_ss / denominator,
                "within_share": within_share,
                "between_share": between_share,
                "constant": is_constant,
            }
        )
    return pd.DataFrame.from_records(records).set_index("channel")


def main() -> None:
    parser = make_parser(
        "Plot within-UAV and between-UAV telemetry variance contributions.",
        "within_between_variance",
    )
    args = parser.parse_args()
    channels = selected_channels(args)
    data = load_dataset(args.train_csv, channels)
    summary = variance_decomposition(data, channels)
    plotted = summary.sort_values(["constant", "within_share"])

    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    y = np.arange(len(plotted))
    within_percent = plotted["within_share"].to_numpy() * 100
    between_percent = plotted["between_share"].to_numpy() * 100
    axis.barh(y, within_percent, color=DARK_BLUE, label="Within UAV")
    axis.barh(
        y,
        between_percent,
        left=within_percent,
        color=LIGHT_BLUE,
        label="Between UAVs",
    )
    for position, (_, row) in enumerate(plotted.iterrows()):
        if row["constant"]:
            axis.text(1, position, "constant", va="center", fontsize=8, color=GRAY)
    axis.set(
        yticks=y,
        yticklabels=plotted.index,
        xlim=(0, 100),
        xlabel="Share of total variance (%)",
        title="Telemetry variance decomposition",
    )
    axis.legend(loc="lower right", frameon=False)
    style_axis(axis)

    csv_path = save_csv(summary, args.output_dir, "within_between_variance.csv")
    figure_path = save_figure(
        figure, args.output_dir, "within_between_variance.png", args.dpi
    )
    print(f"Saved {figure_path}\nSaved {csv_path}")


if __name__ == "__main__":
    main()
