"""Plot an overview of UAV telemetry channels against flight cycle.

The script creates one overview PNG containing the cycle-wise median and mean
for every telemetry channel. Each subplot also reports the variance across all
raw rows for that channel. By default only training data is plotted; pass
``--include-test`` to overlay the available test histories and variance.

Example
-------
python plot_telemetry_over_cycles.py --include-test
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib

# Saving figures must also work on machines without a graphical display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ID_COLUMN = "uav_id"
CYCLE_COLUMN = "flight_cycle"
TELEMETRY_COLUMNS = [f"telemetry_{number:02d}" for number in range(1, 29)]
MEDIAN_COLOR = "#0072B2"
MEAN_COLOR = "#56B4E9"
TRAIN_LINESTYLE = "-"
TEST_LINESTYLE = "--"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "test.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures" / "telemetry_over_cycles"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a telemetry overview over flight cycle."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAIN_CSV,
        help="Path to the training CSV (default: project data/train.csv).",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=DEFAULT_TEST_CSV,
        help="Path to the test CSV (default: project data/test.csv).",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Overlay test histories and their cycle-wise trend.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: figures beside this script).",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=TELEMETRY_COLUMNS,
        default=TELEMETRY_COLUMNS,
        help="Telemetry channels to plot (default: all 28).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output resolution in dots per inch (default: 160).",
    )
    return parser.parse_args()


def load_and_validate_csv(path: Path, channels: Sequence[str]) -> pd.DataFrame:
    """Load a CSV and reject data that would make the plots misleading."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    data = pd.read_csv(path)
    required = [ID_COLUMN, CYCLE_COLUMN, *channels]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    if data[required].isna().any().any():
        raise ValueError(f"{path} contains missing values in plotting columns")

    numeric_columns = [CYCLE_COLUMN, *channels]
    try:
        numeric_values = data[numeric_columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains non-numeric plotting values") from exc
    if not np.isfinite(numeric_values).all():
        raise ValueError(f"{path} contains infinite plotting values")

    duplicate_keys = data.duplicated([ID_COLUMN, CYCLE_COLUMN])
    if duplicate_keys.any():
        count = int(duplicate_keys.sum())
        raise ValueError(f"{path} contains {count} duplicate UAV/cycle keys")

    return data.sort_values([ID_COLUMN, CYCLE_COLUMN], kind="stable")


def cycle_summary(data: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Return the cycle-wise mean and median for one channel."""
    return (
        data.groupby(CYCLE_COLUMN, sort=True)[channel]
        .agg(
            mean="mean",
            median="median",
        )
        .reset_index()
    )


def format_variance(data: pd.DataFrame, channel: str, label: str) -> str:
    """Format the population variance across all rows for a subplot title."""
    values = data[channel].to_numpy(dtype=float)
    variance = 0.0 if values.max() == values.min() else np.var(values, ddof=0)
    return f"Var({label.lower()})={variance:.3e}"


def draw_summary(
    axis: plt.Axes,
    summary: pd.DataFrame,
    label: str,
    linestyle: str,
) -> None:
    cycles = summary[CYCLE_COLUMN].to_numpy()
    axis.plot(
        cycles,
        summary["median"].to_numpy(),
        color=MEDIAN_COLOR,
        linewidth=2.2,
        linestyle=linestyle,
        label=f"{label} median",
    )
    axis.plot(
        cycles,
        summary["mean"].to_numpy(),
        color=MEAN_COLOR,
        linewidth=1.4,
        linestyle=linestyle,
        label=f"{label} mean",
    )


def save_overview_plot(
    channels: Sequence[str],
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    output_dir: Path,
    dpi: int,
) -> None:
    columns = min(4, len(channels))
    rows = math.ceil(len(channels) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.2 * columns, 2.9 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    for axis, channel in zip(axes.flat, channels):
        draw_summary(
            axis,
            cycle_summary(train, channel),
            "Train",
            TRAIN_LINESTYLE,
        )
        if test is not None:
            draw_summary(
                axis,
                cycle_summary(test, channel),
                "Test",
                TEST_LINESTYLE,
            )
        variance_labels = [format_variance(train, channel, "Train")]
        if test is not None:
            variance_labels.append(format_variance(test, channel, "Test"))
        axis.set_title(
            f"{channel}\n{' | '.join(variance_labels)}",
            fontsize=9,
        )
        axis.set_xlabel("Cycle", fontsize=8)
        axis.grid(True, alpha=0.2)
        axis.tick_params(labelsize=8)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)

    for axis in axes.flat[len(channels) :]:
        axis.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
    )
    fig.suptitle("Cycle-wise median and mean telemetry trends", fontsize=14)
    fig.savefig(output_dir / "telemetry_overview.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    channels = list(dict.fromkeys(args.channels))
    train = load_and_validate_csv(args.train_csv, channels)
    test = load_and_validate_csv(args.test_csv, channels) if args.include_test else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_overview_plot(channels, train, test, args.output_dir, args.dpi)

    print(f"Saved telemetry overview to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
