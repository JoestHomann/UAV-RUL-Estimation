"""Combine core analysis evidence into a transparent channel-screening table."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from core_common import (
    CYCLE_COLUMN,
    DARK_BLUE,
    DEFAULT_OUTPUT_ROOT,
    ID_COLUMN,
    LIGHT_GRAY,
    is_effectively_constant,
    load_dataset,
    make_parser,
    save_figure,
    save_table,
    selected_channels,
)


def within_between_share(
    data: pd.DataFrame, channels: list[str]
) -> pd.DataFrame:
    records = []
    for channel in channels:
        values = data[channel].to_numpy(dtype=float)
        group_means = data.groupby(ID_COLUMN)[channel].transform("mean").to_numpy()
        means = data.groupby(ID_COLUMN)[channel].mean()
        sizes = data.groupby(ID_COLUMN)[channel].size()
        grand_mean = float(values.mean())
        within = float(np.square(values - group_means).sum())
        between = float((sizes * np.square(means - grand_mean)).sum())
        total = within + between
        records.append(
            {
                "channel": channel,
                "within_uav_variance_share": within / total if total > 0 else 0.0,
                "between_uav_variance_share": between / total if total > 0 else 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def flatline_coverage(
    data: pd.DataFrame,
    channels: list[str],
    minimum_run: int,
) -> pd.DataFrame:
    records = []
    for channel in channels:
        flatline_rows = 0
        uavs_with_flatline = 0
        for _, history in data.groupby(ID_COLUMN, sort=False):
            values = history[channel].to_numpy()
            if len(values) == 0:
                continue
            boundaries = np.concatenate(
                ([0], np.flatnonzero(values[1:] != values[:-1]) + 1, [len(values)])
            )
            lengths = np.diff(boundaries)
            qualifying = lengths >= minimum_run
            if qualifying.any():
                uavs_with_flatline += 1
                flatline_rows += int(lengths[qualifying].sum())
        records.append(
            {
                "channel": channel,
                "unique_values": int(data[channel].nunique(dropna=False)),
                "flatline_rows_percent": 100.0 * flatline_rows / len(data),
                "uavs_with_flatline_percent": 100.0
                * uavs_with_flatline
                / data[ID_COLUMN].nunique(),
            }
        )
    return pd.DataFrame.from_records(records)


def require_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required analysis output not found: {path}. Run run_all.py first."
        )
    return pd.read_csv(path)


def classify(summary: pd.DataFrame, args) -> pd.DataFrame:
    summary = summary.copy()
    summary["degradation_candidate"] = (
        summary["median_within_spearman_rul"].abs()
        >= args.temporal_correlation_threshold
    ) & (
        summary["dominant_trend_uavs_percent"]
        >= args.trend_consistency_threshold
    ) & ~summary["effectively_constant"]
    summary["context_candidate"] = (
        summary["between_uav_variance_share"]
        >= args.between_variance_threshold
    ) & ~summary["effectively_constant"]
    summary["state_or_regime_candidate"] = (
        summary["flatline_rows_percent"] >= args.flatline_threshold
    ) & ~summary["effectively_constant"]
    summary["anomaly_review"] = (
        (summary["robust_extreme_rows_percent"] >= args.anomaly_row_threshold)
        | (summary["jump_rows_percent"] >= args.anomaly_row_threshold)
        | (
            summary["permanent_shift_uavs_percent"]
            >= args.shift_uav_threshold
        )
    ) & ~summary["effectively_constant"]
    summary["redundancy_candidate"] = (
        summary["strongly_correlated_peer_count"].fillna(0) > 0
    ) & ~summary["effectively_constant"]
    summary["train_test_drift_warning"] = (
        summary["median_absolute_iqr_shift"].fillna(0)
        >= args.drift_iqr_threshold
    ) | (
        summary["outside_train_age_range_percent"].fillna(0)
        >= args.drift_outside_threshold
    ) & ~summary["effectively_constant"]
    summary["removal_candidate"] = summary["effectively_constant"]

    primary_roles = []
    evidence_notes = []
    for row in summary.itertuples(index=False):
        if row.effectively_constant:
            primary = "removal candidate"
        elif row.degradation_candidate and row.context_candidate:
            primary = "mixed degradation/context candidate"
        elif row.degradation_candidate:
            primary = "degradation candidate"
        elif row.context_candidate:
            primary = "operating-context candidate"
        elif row.state_or_regime_candidate:
            primary = "state/regime candidate"
        else:
            primary = "unclassified or weak evidence"
        notes = []
        if row.anomaly_review:
            notes.append("review anomaly flags")
        if row.redundancy_candidate:
            notes.append("check correlated peers")
        if row.train_test_drift_warning:
            notes.append("train/test drift warning")
        if not notes:
            notes.append("no secondary warning at current thresholds")
        primary_roles.append(primary)
        evidence_notes.append("; ".join(notes))
    summary["primary_screening_role"] = primary_roles
    summary["secondary_evidence"] = evidence_notes
    return summary


def plot_classification(table: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    flag_columns = [
        "removal_candidate",
        "degradation_candidate",
        "context_candidate",
        "state_or_regime_candidate",
        "anomaly_review",
        "redundancy_candidate",
        "train_test_drift_warning",
    ]
    labels = [
        "Removal",
        "Degradation",
        "Context",
        "State/regime",
        "Anomaly review",
        "Redundancy",
        "Drift warning",
    ]
    plotted = table.sort_values(
        ["removal_candidate", "degradation_candidate", "context_candidate", "channel"],
        ascending=[False, False, False, True],
    )
    matrix = plotted[flag_columns].astype(int).to_numpy()
    figure, axis = plt.subplots(figsize=(11, 10), constrained_layout=True)
    axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([LIGHT_GRAY, DARK_BLUE]),
        vmin=0,
        vmax=1,
    )
    axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(plotted)), plotted["channel"])
    axis.set_title(
        "Initial telemetry channel classification\n"
        "Blue cells meet the documented screening threshold"
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                "x" if matrix[row_index, column_index] else "",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    return save_figure(
        figure, output_dir, "channel_classification.png", dpi
    )


def main() -> None:
    parser = make_parser(
        "Combine core analysis outputs into an initial channel classification.",
        "channel_classification",
    )
    parser.add_argument(
        "--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument("--minimum-flatline-run", type=int, default=5)
    parser.add_argument("--temporal-correlation-threshold", type=float, default=0.30)
    parser.add_argument("--trend-consistency-threshold", type=float, default=70.0)
    parser.add_argument("--between-variance-threshold", type=float, default=0.50)
    parser.add_argument("--flatline-threshold", type=float, default=5.0)
    parser.add_argument("--anomaly-row-threshold", type=float, default=1.0)
    parser.add_argument("--shift-uav-threshold", type=float, default=10.0)
    parser.add_argument("--drift-iqr-threshold", type=float, default=0.50)
    parser.add_argument("--drift-outside-threshold", type=float, default=5.0)
    args = parser.parse_args()
    channels = selected_channels(args)
    train = load_dataset(args.train_csv, channels, require_rul=True)

    temporal = require_csv(
        args.input_root / "temporal_rul" / "temporal_channel_summary.csv"
    )
    redundancy = require_csv(
        args.input_root / "feature_redundancy" / "correlation_clusters.csv"
    )
    redundancy = redundancy.drop(
        columns=["effectively_constant"], errors="ignore"
    )
    drift = require_csv(
        args.input_root
        / "train_test_drift"
        / "matched_endpoint_drift_summary.csv"
    )
    anomaly = require_csv(
        args.input_root / "anomalies" / "anomaly_channel_summary.csv"
    )
    anomaly = anomaly.loc[anomaly["split"] == "train"].drop(columns="split")

    base = pd.DataFrame(
        {
            "channel": channels,
            "effectively_constant": [
                is_effectively_constant(train[channel]) for channel in channels
            ],
        }
    )
    variance = within_between_share(train, channels)
    flatlines = flatline_coverage(train, channels, args.minimum_flatline_run)
    table = base
    for evidence in [temporal, variance, flatlines, redundancy, drift, anomaly]:
        table = table.merge(evidence, on="channel", how="left")
    table = classify(table, args)

    paths = [
        save_table(table, args.output_dir, "channel_classification.csv"),
        plot_classification(table, args.output_dir, args.dpi),
    ]
    print("\n".join(f"Saved {path}" for path in paths))


if __name__ == "__main__":
    main()
