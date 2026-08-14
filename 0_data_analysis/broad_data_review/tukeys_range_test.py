"""Investigate extreme telemetry readings in train and test UAV histories.

Robust Tukey bounds are learned from training rows only and then applied
unchanged to both splits. The reports distinguish isolated one-cycle spikes
from sustained extreme runs and record cross-channel co-occurrence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from plotting_common import (
    CYCLE_COLUMN,
    DARK_BLUE,
    GRAY,
    ID_COLUMN,
    LIGHT_BLUE,
    LIGHT_GRAY,
    ORANGE,
    RED,
    hide_unused_axes,
    load_dataset,
    make_parser,
    save_csv,
    save_figure,
    selected_channels,
    style_axis,
    subplot_grid,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_CHANNELS = [
    "telemetry_01",
    "telemetry_04",
    "telemetry_10",
    "telemetry_11",
    "telemetry_18",
    "telemetry_24",
    "telemetry_26",
]


@dataclass
class SplitAnalysis:
    data: pd.DataFrame
    flags: pd.DataFrame
    detail: pd.DataFrame
    summary: pd.DataFrame
    by_uav: pd.DataFrame
    cooccurrence: pd.DataFrame


def calculate_train_bounds(
    train: pd.DataFrame,
    channels: list[str],
    multiplier: float,
) -> pd.DataFrame:
    records = []
    for channel in channels:
        q1 = float(train[channel].quantile(0.25))
        q3 = float(train[channel].quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            raise ValueError(
                f"{channel} has zero IQR and cannot use Tukey extreme bounds"
            )
        records.append(
            {
                "channel": channel,
                "q25": q1,
                "q75": q3,
                "iqr": iqr,
                "iqr_multiplier": multiplier,
                "lower_bound": q1 - multiplier * iqr,
                "upper_bound": q3 + multiplier * iqr,
            }
        )
    return pd.DataFrame.from_records(records).set_index("channel")


def flagged_run_lengths(flags: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    """Return the extreme-run length at flagged positions and zero elsewhere."""
    result = np.zeros(len(flags), dtype=int)
    if len(flags) == 0:
        return result

    new_run = np.concatenate(
        (
            [True],
            (flags[1:] != flags[:-1]) | (cycles[1:] != cycles[:-1] + 1),
            [True],
        )
    )
    boundaries = np.flatnonzero(new_run)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if flags[start]:
            result[start:stop] = stop - start
    return result


def make_cooccurrence_table(flags: pd.DataFrame, split: str) -> pd.DataFrame:
    matrix = flags.astype(np.int64).T.dot(flags.astype(np.int64))
    records = []
    for row_index, channel_a in enumerate(flags.columns):
        for channel_b in flags.columns[row_index:]:
            records.append(
                {
                    "split": split,
                    "channel_a": channel_a,
                    "channel_b": channel_b,
                    "cooccurring_rows": int(matrix.loc[channel_a, channel_b]),
                }
            )
    return pd.DataFrame.from_records(records)


def analyse_split(
    data: pd.DataFrame,
    split: str,
    channels: list[str],
    bounds: pd.DataFrame,
) -> SplitAnalysis:
    data = data.reset_index(drop=True).copy()
    flags = pd.DataFrame(index=data.index)
    for channel in channels:
        flags[channel] = (data[channel] < bounds.loc[channel, "lower_bound"]) | (
            data[channel] > bounds.loc[channel, "upper_bound"]
        )

    cooccurring_count = flags.sum(axis=1).astype(int)
    cooccurring_names = flags.apply(
        lambda row: ";".join(row.index[row.to_numpy()]), axis=1
    )
    detail_parts = []
    summary_records = []
    uav_count = int(data[ID_COLUMN].nunique())

    for channel in channels:
        channel_flags = flags[channel].to_numpy(dtype=bool)
        run_length = np.zeros(len(data), dtype=int)
        for _, positions in data.groupby(ID_COLUMN, sort=False).indices.items():
            ordered_positions = np.asarray(positions, dtype=int)
            run_length[ordered_positions] = flagged_run_lengths(
                channel_flags[ordered_positions],
                data.loc[ordered_positions, CYCLE_COLUMN].to_numpy(dtype=int),
            )

        selected_index = np.flatnonzero(channel_flags)
        selected = data.loc[selected_index]
        lower = float(bounds.loc[channel, "lower_bound"])
        upper = float(bounds.loc[channel, "upper_bound"])
        iqr = float(bounds.loc[channel, "iqr"])
        values = selected[channel].to_numpy(dtype=float)
        direction = np.where(values < lower, "low", "high")
        distance = np.where(
            values < lower,
            (lower - values) / iqr,
            (values - upper) / iqr,
        )
        previous_value = data.groupby(ID_COLUMN, sort=False)[channel].shift(1)
        next_value = data.groupby(ID_COLUMN, sort=False)[channel].shift(-1)

        detail = pd.DataFrame(
            {
                "split": split,
                ID_COLUMN: selected[ID_COLUMN].to_numpy(),
                CYCLE_COLUMN: selected[CYCLE_COLUMN].to_numpy(),
                "RUL": selected["RUL"].to_numpy()
                if "RUL" in selected.columns
                else np.nan,
                "channel": channel,
                "value": values,
                "previous_value": previous_value.loc[selected_index].to_numpy(),
                "next_value": next_value.loc[selected_index].to_numpy(),
                "direction": direction,
                "lower_bound": lower,
                "upper_bound": upper,
                "distance_beyond_bound_iqr": distance,
                "extreme_run_length": run_length[selected_index],
                "isolated_spike": run_length[selected_index] == 1,
                "cooccurring_channel_count": cooccurring_count.loc[
                    selected_index
                ].to_numpy(),
                "cooccurring_channels": cooccurring_names.loc[
                    selected_index
                ].to_numpy(),
            }
        )
        detail["change_from_previous"] = detail["value"] - detail["previous_value"]
        detail["change_to_next"] = detail["next_value"] - detail["value"]
        detail_parts.append(detail)

        extreme_rows = len(detail)
        affected_uavs = int(detail[ID_COLUMN].nunique()) if extreme_rows else 0
        summary_records.append(
            {
                "split": split,
                "channel": channel,
                "total_rows": len(data),
                "extreme_rows": extreme_rows,
                "extreme_rows_percent": 100 * extreme_rows / len(data),
                "affected_uavs": affected_uavs,
                "total_uavs": uav_count,
                "affected_uavs_percent": 100 * affected_uavs / uav_count,
                "low_extremes": int((detail["direction"] == "low").sum()),
                "high_extremes": int((detail["direction"] == "high").sum()),
                "isolated_extremes": int(detail["isolated_spike"].sum()),
                "isolated_extremes_percent": (
                    100 * detail["isolated_spike"].mean() if extreme_rows else 0.0
                ),
                "maximum_extreme_run": int(detail["extreme_run_length"].max())
                if extreme_rows
                else 0,
                "cooccurring_extremes_percent": (
                    100 * (detail["cooccurring_channel_count"] > 1).mean()
                    if extreme_rows
                    else 0.0
                ),
                "earliest_extreme_cycle": int(detail[CYCLE_COLUMN].min())
                if extreme_rows
                else np.nan,
                "median_extreme_cycle": float(detail[CYCLE_COLUMN].median())
                if extreme_rows
                else np.nan,
                "latest_extreme_cycle": int(detail[CYCLE_COLUMN].max())
                if extreme_rows
                else np.nan,
            }
        )

    all_detail = pd.concat(detail_parts, ignore_index=True)
    if all_detail.empty:
        by_uav = pd.DataFrame(
            columns=[
                "split",
                "channel",
                ID_COLUMN,
                "extreme_rows",
                "maximum_extreme_run",
                "isolated_extremes_percent",
            ]
        )
    else:
        by_uav = (
            all_detail.groupby(["split", "channel", ID_COLUMN], sort=True)
            .agg(
                extreme_rows=(CYCLE_COLUMN, "size"),
                maximum_extreme_run=("extreme_run_length", "max"),
                isolated_extremes_percent=("isolated_spike", lambda x: 100 * x.mean()),
                first_extreme_cycle=(CYCLE_COLUMN, "min"),
                last_extreme_cycle=(CYCLE_COLUMN, "max"),
            )
            .reset_index()
        )

    return SplitAnalysis(
        data=data,
        flags=flags,
        detail=all_detail,
        summary=pd.DataFrame.from_records(summary_records),
        by_uav=by_uav,
        cooccurrence=make_cooccurrence_table(flags, split),
    )


def save_overview(
    train_analysis: SplitAnalysis,
    test_analysis: SplitAnalysis,
    bounds: pd.DataFrame,
    channels: list[str],
    output_dir,
    dpi: int,
) -> object:
    figure, axes = subplot_grid(len(channels), columns=4, height=3.2)
    train_summary = train_analysis.summary.set_index("channel")
    test_summary = test_analysis.summary.set_index("channel")

    for axis, channel in zip(axes.flat, channels):
        train_flag = train_analysis.flags[channel].to_numpy(dtype=bool)
        test_flag = test_analysis.flags[channel].to_numpy(dtype=bool)

        axis.scatter(
            train_analysis.data.loc[~train_flag, CYCLE_COLUMN],
            train_analysis.data.loc[~train_flag, channel],
            s=2,
            color=LIGHT_GRAY,
            alpha=0.22,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            test_analysis.data.loc[~test_flag, CYCLE_COLUMN],
            test_analysis.data.loc[~test_flag, channel],
            s=2,
            color=LIGHT_BLUE,
            alpha=0.18,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            train_analysis.data.loc[train_flag, CYCLE_COLUMN],
            train_analysis.data.loc[train_flag, channel],
            s=9,
            color=RED,
            alpha=0.72,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            test_analysis.data.loc[test_flag, CYCLE_COLUMN],
            test_analysis.data.loc[test_flag, channel],
            s=9,
            color=ORANGE,
            alpha=0.72,
            linewidths=0,
            rasterized=True,
        )
        axis.axhline(
            bounds.loc[channel, "lower_bound"],
            color=GRAY,
            linewidth=0.8,
            linestyle="--",
        )
        axis.axhline(
            bounds.loc[channel, "upper_bound"],
            color=GRAY,
            linewidth=0.8,
            linestyle="--",
        )
        axis.set_title(
            f"{channel}\n"
            f"train {train_summary.loc[channel, 'extreme_rows_percent']:.2f}% | "
            f"test {test_summary.loc[channel, 'extreme_rows_percent']:.2f}%",
            fontsize=9,
        )
        axis.set_xlabel("Flight cycle", fontsize=8)
        axis.ticklabel_format(
            axis="y", style="sci", scilimits=(-3, 4), useOffset=False
        )
        style_axis(axis)

    hide_unused_axes(axes, len(channels))
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LIGHT_GRAY, label="Train rows", markersize=5),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LIGHT_BLUE, label="Test rows", markersize=5),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, label="Train extremes", markersize=5),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, label="Test extremes", markersize=5),
        Line2D([0], [0], color=GRAY, linestyle="--", label="Train-derived bounds"),
    ]
    figure.legend(
        handles=legend,
        loc="outside lower center",
        ncol=len(legend),
        frameon=False,
    )
    figure.suptitle(
        "Extreme telemetry readings by flight cycle\n"
        "Bounds learned from training data and applied unchanged to test data",
        fontsize=14,
    )
    return save_figure(
        figure, output_dir, "Tukey_extreme_readings_overview.png", dpi
    )


def main() -> None:
    parser = make_parser(
        "Investigate extreme telemetry readings in train and test histories.",
        "Tukey_extreme_readings_investigation",
        include_test=True,
    )
    parser.set_defaults(channels=DEFAULT_CHANNELS)
    for action in parser._actions:
        if action.dest == "channels":
            action.help = (
                "Telemetry channels to investigate "
                f"(default: {' '.join(DEFAULT_CHANNELS)})."
            )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=3.0,
        help="Tukey-bound IQR multiplier (default: 3.0 for extreme outliers).",
    )
    args = parser.parse_args()
    if args.iqr_multiplier <= 0:
        parser.error("--iqr-multiplier must be greater than zero")

    channels = selected_channels(args)
    train = load_dataset(args.train_csv, channels)
    test = load_dataset(args.test_csv, channels)
    bounds = calculate_train_bounds(train, channels, args.iqr_multiplier)
    train_analysis = analyse_split(train, "train", channels, bounds)
    test_analysis = analyse_split(test, "test", channels, bounds)

    summary = pd.concat(
        [train_analysis.summary, test_analysis.summary], ignore_index=True
    ).set_index(["split", "channel"])
    detail = pd.concat(
        [train_analysis.detail, test_analysis.detail], ignore_index=True
    ).sort_values(["split", "channel", ID_COLUMN, CYCLE_COLUMN])
    by_uav = pd.concat(
        [train_analysis.by_uav, test_analysis.by_uav], ignore_index=True
    ).set_index(["split", "channel", ID_COLUMN])
    cooccurrence = pd.concat(
        [train_analysis.cooccurrence, test_analysis.cooccurrence], ignore_index=True
    ).set_index(["split", "channel_a", "channel_b"])

    figure_path = save_overview(
        train_analysis,
        test_analysis,
        bounds,
        channels,
        args.output_dir,
        args.dpi,
    )
    bounds_path = save_csv(bounds, args.output_dir, "extreme_bounds.csv")
    summary_path = save_csv(summary, args.output_dir, "extreme_summary.csv")
    detail_path = save_csv(
        detail.set_index(["split", "channel", ID_COLUMN, CYCLE_COLUMN]),
        args.output_dir,
        "extreme_rows.csv",
    )
    by_uav_path = save_csv(by_uav, args.output_dir, "extreme_by_uav.csv")
    cooccurrence_path = save_csv(
        cooccurrence, args.output_dir, "extreme_cooccurrence.csv"
    )

    print(summary[["extreme_rows", "extreme_rows_percent", "affected_uavs", "isolated_extremes_percent", "maximum_extreme_run"]].to_string())
    print(
        f"\nSaved {figure_path}\n"
        f"Saved {bounds_path}\n"
        f"Saved {summary_path}\n"
        f"Saved {detail_path}\n"
        f"Saved {by_uav_path}\n"
        f"Saved {cooccurrence_path}"
    )


if __name__ == "__main__":
    main()
