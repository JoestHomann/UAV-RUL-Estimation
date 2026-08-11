"""Measure pooled and within-UAV temporal relationships with cycle and RUL."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from core_common import (
    CYCLE_COLUMN,
    DARK_BLUE,
    GRAY,
    ID_COLUMN,
    LIGHT_BLUE,
    ORANGE,
    TARGET_COLUMN,
    is_effectively_constant,
    linear_slope,
    load_dataset,
    make_parser,
    residualize,
    safe_correlation,
    save_figure,
    save_table,
    selected_channels,
    style_axis,
)


def robust_theil_slope(
    cycles: np.ndarray,
    values: np.ndarray,
    maximum_points: int,
) -> float:
    if is_effectively_constant(values):
        return 0.0
    if len(values) > maximum_points:
        positions = np.unique(
            np.linspace(0, len(values) - 1, maximum_points).round().astype(int)
        )
        cycles = cycles[positions]
        values = values[positions]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.theilslopes(values, cycles).slope)


def calculate_pooled(data: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    cycle = data[CYCLE_COLUMN].to_numpy(dtype=float)
    rul = data[TARGET_COLUMN].to_numpy(dtype=float)
    records = []
    for channel in channels:
        values = data[channel].to_numpy(dtype=float)
        residual_values = residualize(values, cycle, rank=False)
        residual_rul = residualize(rul, cycle, rank=False)
        rank_residual_values = residualize(values, cycle, rank=True)
        rank_residual_rul = residualize(rul, cycle, rank=True)
        records.append(
            {
                "channel": channel,
                "pooled_pearson_rul": safe_correlation(
                    values, rul, method="pearson"
                ),
                "pooled_spearman_rul": safe_correlation(
                    values, rul, method="spearman"
                ),
                "pooled_pearson_cycle": safe_correlation(
                    values, cycle, method="pearson"
                ),
                "pooled_spearman_cycle": safe_correlation(
                    values, cycle, method="spearman"
                ),
                "partial_pearson_rul_controlling_cycle": safe_correlation(
                    residual_values, residual_rul, method="pearson"
                ),
                "partial_spearman_rul_controlling_cycle": safe_correlation(
                    rank_residual_values, rank_residual_rul, method="pearson"
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def calculate_per_uav(
    data: pd.DataFrame,
    channels: list[str],
    *,
    early_fraction: float,
    robust_maximum_points: int,
) -> pd.DataFrame:
    records = []
    for uav_id, history in data.groupby(ID_COLUMN, sort=True):
        cycles = history[CYCLE_COLUMN].to_numpy(dtype=float)
        rul = history[TARGET_COLUMN].to_numpy(dtype=float)
        segment_size = max(5, int(np.ceil(len(history) * early_fraction)))
        segment_size = min(segment_size, len(history) // 2)
        for channel in channels:
            values = history[channel].to_numpy(dtype=float)
            early_mean = float(values[:segment_size].mean())
            late_mean = float(values[-segment_size:].mean())
            change = late_mean - early_mean
            sd = float(np.std(values, ddof=1))
            records.append(
                {
                    ID_COLUMN: uav_id,
                    "channel": channel,
                    "n_cycles": len(history),
                    "pearson_cycle": safe_correlation(
                        values, cycles, method="pearson"
                    ),
                    "spearman_cycle": safe_correlation(
                        values, cycles, method="spearman"
                    ),
                    "pearson_rul": safe_correlation(values, rul, method="pearson"),
                    "spearman_rul": safe_correlation(
                        values, rul, method="spearman"
                    ),
                    "linear_slope_per_cycle": linear_slope(cycles, values),
                    "theil_sen_slope_per_cycle": robust_theil_slope(
                        cycles, values, robust_maximum_points
                    ),
                    "first_value": float(values[0]),
                    "last_value": float(values[-1]),
                    "last_minus_first": float(values[-1] - values[0]),
                    "early_mean": early_mean,
                    "late_mean": late_mean,
                    "early_to_late_change": change,
                    "early_to_late_effect_size": change / sd if sd > 0 else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_per_uav(per_uav: pd.DataFrame) -> pd.DataFrame:
    records = []
    for channel, group in per_uav.groupby("channel", sort=False):
        slopes = group["linear_slope_per_cycle"].to_numpy(dtype=float)
        nonzero = slopes[~np.isclose(slopes, 0.0, rtol=1e-10, atol=1e-12)]
        positive = 100.0 * float(np.mean(nonzero > 0)) if len(nonzero) else np.nan
        negative = 100.0 * float(np.mean(nonzero < 0)) if len(nonzero) else np.nan
        if len(nonzero):
            dominant_direction = "increasing" if positive >= negative else "decreasing"
            dominant_percent = max(positive, negative)
        else:
            dominant_direction = "flat"
            dominant_percent = np.nan
        records.append(
            {
                "channel": channel,
                "valid_uavs": int(group["spearman_cycle"].notna().sum()),
                "median_within_pearson_cycle": group["pearson_cycle"].median(),
                "median_within_spearman_cycle": group["spearman_cycle"].median(),
                "median_within_pearson_rul": group["pearson_rul"].median(),
                "median_within_spearman_rul": group["spearman_rul"].median(),
                "median_linear_slope_per_cycle": group[
                    "linear_slope_per_cycle"
                ].median(),
                "median_theil_sen_slope_per_cycle": group[
                    "theil_sen_slope_per_cycle"
                ].median(),
                "positive_slope_uavs_percent": positive,
                "negative_slope_uavs_percent": negative,
                "dominant_trend_direction": dominant_direction,
                "dominant_trend_uavs_percent": dominant_percent,
                "median_last_minus_first": group["last_minus_first"].median(),
                "median_early_to_late_change": group[
                    "early_to_late_change"
                ].median(),
                "median_early_to_late_effect_size": group[
                    "early_to_late_effect_size"
                ].median(),
            }
        )
    return pd.DataFrame.from_records(records)


def calculate_endpoint_rolling_features(
    data: pd.DataFrame,
    channels: list[str],
    windows: list[int],
) -> pd.DataFrame:
    records = []
    for uav_id, history in data.groupby(ID_COLUMN, sort=True):
        for channel in channels:
            full_values = history[channel].to_numpy(dtype=float)
            full_cycles = history[CYCLE_COLUMN].to_numpy(dtype=float)
            for window in windows:
                values = full_values[-window:]
                cycles = full_cycles[-window:]
                records.append(
                    {
                        ID_COLUMN: uav_id,
                        "channel": channel,
                        "window": window,
                        "available_points": len(values),
                        "endpoint_cycle": int(full_cycles[-1]),
                        "rolling_mean": float(values.mean()),
                        "rolling_sd": float(np.std(values, ddof=1))
                        if len(values) > 1
                        else 0.0,
                        "window_delta": float(values[-1] - values[0]),
                        "window_slope": linear_slope(cycles, values),
                    }
                )
    return pd.DataFrame.from_records(records)


def plot_summary(summary: pd.DataFrame, output_dir, dpi: int) -> object:
    plotted = summary.sort_values(
        "median_within_spearman_rul", na_position="first"
    ).reset_index(drop=True)
    y = np.arange(len(plotted))
    figure, axes = plt.subplots(1, 4, figsize=(19, 10), sharey=True, constrained_layout=True)

    axes[0].barh(
        y - 0.18,
        plotted["pooled_spearman_rul"].fillna(0),
        height=0.35,
        color=DARK_BLUE,
        label="RUL",
    )
    axes[0].barh(
        y + 0.18,
        plotted["pooled_spearman_cycle"].fillna(0),
        height=0.35,
        color=LIGHT_BLUE,
        label="Cycle",
    )
    axes[0].set_title("Pooled Spearman correlation")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].barh(
        y,
        plotted["partial_spearman_rul_controlling_cycle"].fillna(0),
        color=ORANGE,
    )
    axes[1].set_title("RUL association\nafter controlling for cycle")

    axes[2].barh(
        y,
        plotted["median_within_spearman_rul"].fillna(0),
        color=DARK_BLUE,
    )
    axes[2].set_title("Median within-UAV\nSpearman correlation with RUL")

    signed_consistency = plotted["dominant_trend_uavs_percent"].fillna(0).to_numpy(copy=True)
    signed_consistency *= np.where(
        plotted["dominant_trend_direction"].eq("increasing"), 1.0, -1.0
    )
    colors = np.where(signed_consistency >= 0, LIGHT_BLUE, ORANGE)
    axes[3].barh(y, signed_consistency, color=colors)
    axes[3].set_title("UAV trend consistency\n(+ increasing, - decreasing)")
    axes[3].set_xlim(-100, 100)

    for axis in axes:
        axis.axvline(0, color=GRAY, linewidth=0.8)
        style_axis(axis)
    axes[0].set_yticks(y, plotted["channel"])
    axes[0].set_xlabel("Correlation")
    axes[1].set_xlabel("Partial correlation")
    axes[2].set_xlabel("Correlation")
    axes[3].set_xlabel("Dominant-direction UAVs (%)")
    figure.suptitle("Temporal and RUL telemetry evidence", fontsize=15)
    return save_figure(figure, output_dir, "temporal_rul_summary.png", dpi)


def main() -> None:
    parser = make_parser(
        "Analyse pooled and within-UAV telemetry relationships with cycle and RUL.",
        "temporal_rul",
    )
    parser.add_argument(
        "--early-fraction",
        type=float,
        default=0.20,
        help="Fraction used for early/late segment means (default: 0.20).",
    )
    parser.add_argument(
        "--rolling-windows",
        nargs="+",
        type=int,
        default=[5, 20, 50],
    )
    parser.add_argument(
        "--robust-maximum-points",
        type=int,
        default=80,
        help="Maximum evenly spaced points used by each Theil-Sen fit.",
    )
    args = parser.parse_args()
    if not 0 < args.early_fraction <= 0.5:
        parser.error("--early-fraction must be in (0, 0.5]")
    if any(window < 2 for window in args.rolling_windows):
        parser.error("Every rolling window must be at least 2")
    if args.robust_maximum_points < 3:
        parser.error("--robust-maximum-points must be at least 3")

    channels = selected_channels(args)
    data = load_dataset(args.train_csv, channels, require_rul=True)
    pooled = calculate_pooled(data, channels)
    per_uav = calculate_per_uav(
        data,
        channels,
        early_fraction=args.early_fraction,
        robust_maximum_points=args.robust_maximum_points,
    )
    per_uav_summary = summarize_per_uav(per_uav)
    summary = pooled.merge(per_uav_summary, on="channel", how="left")
    rolling = calculate_endpoint_rolling_features(
        data, channels, sorted(set(args.rolling_windows))
    )

    paths = [
        save_table(pooled, args.output_dir, "pooled_correlations.csv"),
        save_table(per_uav, args.output_dir, "per_uav_temporal_metrics.csv"),
        save_table(summary, args.output_dir, "temporal_channel_summary.csv"),
        save_table(rolling, args.output_dir, "endpoint_rolling_features.csv"),
        plot_summary(summary, args.output_dir, args.dpi),
    ]
    print("\n".join(f"Saved {path}" for path in paths))


if __name__ == "__main__":
    main()
