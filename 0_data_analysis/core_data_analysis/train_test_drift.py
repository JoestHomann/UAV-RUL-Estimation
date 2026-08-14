"""Quantify train/test telemetry drift, including matched-age endpoint drift."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_common import (
    CYCLE_COLUMN,
    DARK_BLUE,
    GRAY,
    ID_COLUMN,
    LIGHT_BLUE,
    ORANGE,
    is_effectively_constant,
    load_dataset,
    make_parser,
    save_figure,
    save_table,
    selected_channels,
    sort_telemetry_channels,
    style_axis,
)


AGE_LABELS = ["1-50", "51-100", "101-200", ">200"]
AGE_BINS = [0, 50, 100, 200, np.inf]


def distribution_stats(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    series = pd.Series(values, dtype=float)
    quantiles = series.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "sd": float(series.std(ddof=1)),
        "q05": float(quantiles.loc[0.05]),
        "q25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.50]),
        "q75": float(quantiles.loc[0.75]),
        "q95": float(quantiles.loc[0.95]),
        "min": float(series.min()),
        "max": float(series.max()),
        "iqr": float(quantiles.loc[0.75] - quantiles.loc[0.25]),
    }


def compare_values(
    train_values: pd.Series | np.ndarray,
    test_values: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    train_stats = distribution_stats(train_values)
    test_stats = distribution_stats(test_values)
    test_array = np.asarray(test_values, dtype=float)
    outside = np.mean(
        (test_array < train_stats["min"]) | (test_array > train_stats["max"])
    )
    train_constant = is_effectively_constant(train_values)
    return {
        **{f"train_{key}": value for key, value in train_stats.items()},
        **{f"test_{key}": value for key, value in test_stats.items()},
        "train_effectively_constant": train_constant,
        "standardized_mean_shift": (
            (test_stats["mean"] - train_stats["mean"]) / train_stats["sd"]
            if train_stats["sd"] > 0 and not train_constant
            else np.nan
        ),
        "median_iqr_shift": (
            (test_stats["median"] - train_stats["median"]) / train_stats["iqr"]
            if train_stats["iqr"] > 0 and not train_constant
            else np.nan
        ),
        "test_outside_train_range_percent": (
            100.0 * float(outside) if not train_constant else np.nan
        ),
    }


def raw_and_uav_drift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    channels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_records = []
    train_uav = train.groupby(ID_COLUMN, sort=True)[channels].mean()
    test_uav = test.groupby(ID_COLUMN, sort=True)[channels].mean()
    uav_records = []
    for channel in channels:
        raw_records.append(
            {"channel": channel, **compare_values(train[channel], test[channel])}
        )
        uav_records.append(
            {
                "channel": channel,
                **compare_values(train_uav[channel], test_uav[channel]),
            }
        )
    return (
        pd.DataFrame.from_records(raw_records),
        pd.DataFrame.from_records(uav_records),
    )


def age_band_drift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    channels: list[str],
) -> pd.DataFrame:
    train = train.copy()
    test = test.copy()
    train["age_band"] = pd.cut(
        train[CYCLE_COLUMN], AGE_BINS, labels=AGE_LABELS, right=True
    )
    test["age_band"] = pd.cut(
        test[CYCLE_COLUMN], AGE_BINS, labels=AGE_LABELS, right=True
    )
    records = []
    for age_band in AGE_LABELS:
        train_band = train.loc[train["age_band"] == age_band]
        test_band = test.loc[test["age_band"] == age_band]
        for channel in channels:
            records.append(
                {
                    "age_band": age_band,
                    "channel": channel,
                    **compare_values(train_band[channel], test_band[channel]),
                }
            )
    return pd.DataFrame.from_records(records)


def matched_endpoint_drift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    channels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_at_cycle = {
        int(cycle): group.set_index(ID_COLUMN)
        for cycle, group in train.groupby(CYCLE_COLUMN, sort=True)
    }
    test_endpoints = test.loc[
        test.groupby(ID_COLUMN)[CYCLE_COLUMN].idxmax()
    ].sort_values(ID_COLUMN)
    records = []
    for endpoint in test_endpoints.itertuples(index=False):
        cycle = int(getattr(endpoint, CYCLE_COLUMN))
        reference = train_at_cycle.get(cycle)
        if reference is None or reference.empty:
            continue
        for channel in channels:
            train_values = reference[channel].to_numpy(dtype=float)
            test_value = float(getattr(endpoint, channel))
            train_mean = float(train_values.mean())
            train_sd = float(train_values.std(ddof=1)) if len(train_values) > 1 else np.nan
            q25, median, q75 = np.quantile(train_values, [0.25, 0.50, 0.75])
            iqr = float(q75 - q25)
            records.append(
                {
                    "test_uav_id": getattr(endpoint, ID_COLUMN),
                    "observed_cycle": cycle,
                    "channel": channel,
                    "test_value": test_value,
                    "train_uavs_at_same_age": len(train_values),
                    "train_age_mean": train_mean,
                    "train_age_sd": train_sd,
                    "train_age_median": float(median),
                    "train_age_iqr": iqr,
                    "train_age_min": float(train_values.min()),
                    "train_age_max": float(train_values.max()),
                    "standardized_mean_shift": (
                        (test_value - train_mean) / train_sd
                        if pd.notna(train_sd) and train_sd > 0
                        else np.nan
                    ),
                    "median_iqr_shift": (
                        (test_value - median) / iqr if iqr > 0 else np.nan
                    ),
                    "outside_train_age_range": bool(
                        test_value < train_values.min() or test_value > train_values.max()
                    ),
                }
            )
    details = pd.DataFrame.from_records(records)
    summary_records = []
    for channel, group in details.groupby("channel", sort=False):
        summary_records.append(
            {
                "channel": channel,
                "test_uavs_compared": int(group["test_uav_id"].nunique()),
                "minimum_train_uavs_at_same_age": int(
                    group["train_uavs_at_same_age"].min()
                ),
                "median_standardized_mean_shift": group[
                    "standardized_mean_shift"
                ].median(),
                "median_absolute_standardized_mean_shift": group[
                    "standardized_mean_shift"
                ].abs().median(),
                "median_iqr_shift": group["median_iqr_shift"].median(),
                "median_absolute_iqr_shift": group["median_iqr_shift"].abs().median(),
                "outside_train_age_range_percent": 100.0
                * group["outside_train_age_range"].mean(),
            }
        )
    return details, pd.DataFrame.from_records(summary_records)


def plot_drift(
    raw: pd.DataFrame,
    uav: pd.DataFrame,
    matched: pd.DataFrame,
    output_dir,
    dpi: int,
) -> object:
    nonconstant = raw.loc[~raw["train_effectively_constant"], "channel"]
    matched = matched.loc[matched["channel"].isin(nonconstant)]
    order = sort_telemetry_channels(matched["channel"].tolist())
    plotted = matched.set_index("channel").loc[order].reset_index()
    raw_indexed = raw.set_index("channel").loc[order]
    uav_indexed = uav.set_index("channel").loc[order]
    plotted = plotted.set_index("channel").loc[order]
    y = np.arange(len(order))

    figure, axes = plt.subplots(1, 4, figsize=(19, 10), sharey=True, constrained_layout=True)
    axes[0].barh(y, raw_indexed["standardized_mean_shift"].fillna(0), color=DARK_BLUE)
    axes[0].set_title("All-row mean shift")
    axes[0].set_xlabel("Train SD")
    axes[1].barh(y, uav_indexed["standardized_mean_shift"].fillna(0), color=LIGHT_BLUE)
    axes[1].set_title("Equal-weight UAV mean shift")
    axes[1].set_xlabel("Train UAV-level SD")
    axes[2].barh(y, plotted["median_iqr_shift"].fillna(0), color=ORANGE)
    axes[2].set_title("Matched-age endpoint shift")
    axes[2].set_xlabel("Train same-age IQR")
    axes[3].barh(
        y,
        plotted["outside_train_age_range_percent"],
        color=ORANGE,
    )
    axes[3].set_title("Matched endpoints outside\ntrain same-age range")
    axes[3].set_xlabel("Test UAVs (%)")
    for axis in axes[:3]:
        axis.axvline(0, color=GRAY, linewidth=0.8)
    for axis in axes:
        style_axis(axis)
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    figure.suptitle(
        "Train/test telemetry drift\n"
        "Matched-age results compare each test endpoint with training UAVs observed at the same cycle",
        fontsize=14,
    )
    return save_figure(figure, output_dir, "train_test_drift.png", dpi)


def main() -> None:
    parser = make_parser(
        "Quantify raw, UAV-level, age-band, and matched-endpoint train/test drift.",
        "train_test_drift",
        include_test=True,
    )
    args = parser.parse_args()
    channels = selected_channels(args)
    train = load_dataset(args.train_csv, channels, require_rul=True)
    test = load_dataset(args.test_csv, channels, require_rul=False)

    raw, uav = raw_and_uav_drift(train, test, channels)
    age_band = age_band_drift(train, test, channels)
    matched_details, matched_summary = matched_endpoint_drift(
        train, test, channels
    )
    paths = [
        save_table(raw, args.output_dir, "row_level_drift.csv"),
        save_table(uav, args.output_dir, "uav_level_drift.csv"),
        save_table(age_band, args.output_dir, "age_band_drift.csv"),
        save_table(
            matched_details,
            args.output_dir,
            "matched_endpoint_drift_by_test_uav.csv",
        ),
        save_table(
            matched_summary,
            args.output_dir,
            "matched_endpoint_drift_summary.csv",
        ),
        plot_drift(raw, uav, matched_summary, args.output_dir, args.dpi),
    ]
    print("\n".join(f"Saved {path}" for path in paths))


if __name__ == "__main__":
    main()
