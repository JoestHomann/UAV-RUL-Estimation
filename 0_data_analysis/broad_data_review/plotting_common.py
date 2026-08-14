"""Shared paths, validation, statistics, and styling for data-review plots."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ID_COLUMN = "uav_id"
CYCLE_COLUMN = "flight_cycle"
TELEMETRY_COLUMNS = [f"telemetry_{number:02d}" for number in range(1, 29)]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "test.csv"
FIGURES_ROOT = SCRIPT_DIR / "figures"

DARK_BLUE = "#0072B2"
LIGHT_BLUE = "#56B4E9"
ORANGE = "#D55E00"
GREEN = "#009E73"
YELLOW = "#E69F00"
GRAY = "#7A7A7A"
LIGHT_GRAY = "#D9D9D9"
RED = "#C44E52"


def is_effectively_constant(
    values: pd.Series | np.ndarray,
    *,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
) -> bool:
    """Treat machine-scale numeric jitter around one value as constant."""
    array = np.asarray(values, dtype=float)
    minimum = float(array.min())
    maximum = float(array.max())
    scale = max(abs(minimum), abs(maximum), 1.0)
    return bool(
        maximum - minimum <= absolute_tolerance + relative_tolerance * scale
    )


def make_parser(
    description: str,
    output_subdirectory: str,
    *,
    include_test: bool = False,
    include_channels: bool = True,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAIN_CSV,
        help="Training CSV (default: project data/train.csv).",
    )
    if include_test:
        parser.add_argument(
            "--test-csv",
            type=Path,
            default=DEFAULT_TEST_CSV,
            help="Test CSV (default: project data/test.csv).",
        )
    if include_channels:
        parser.add_argument(
            "--channels",
            nargs="+",
            choices=TELEMETRY_COLUMNS,
            default=TELEMETRY_COLUMNS,
            help="Telemetry channels to include (default: all 28).",
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_ROOT / output_subdirectory,
        help=(
            "Output directory "
            f"(default: 0_data_analysis/broad_data_review/figures/{output_subdirectory})."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure resolution in dots per inch (default: 160).",
    )
    return parser


def selected_channels(args: argparse.Namespace) -> list[str]:
    """Return requested channels once each, preserving their input order."""
    return list(dict.fromkeys(args.channels))


def load_dataset(path: Path, channels: Sequence[str]) -> pd.DataFrame:
    """Load and validate the identifier, cycle, and requested telemetry fields."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    data = pd.read_csv(path)
    required = [ID_COLUMN, CYCLE_COLUMN, *channels]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    if data[required].isna().any().any():
        raise ValueError(f"{path} contains missing values in required columns")

    numeric_columns = [CYCLE_COLUMN, *channels]
    try:
        numeric_values = data[numeric_columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains non-numeric telemetry values") from exc
    if not np.isfinite(numeric_values).all():
        raise ValueError(f"{path} contains non-finite telemetry values")

    duplicate_keys = data.duplicated([ID_COLUMN, CYCLE_COLUMN])
    if duplicate_keys.any():
        raise ValueError(
            f"{path} contains {int(duplicate_keys.sum())} duplicate UAV/cycle keys"
        )

    return data.sort_values([ID_COLUMN, CYCLE_COLUMN], kind="stable")


def describe_values(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    series = pd.Series(values, dtype=float)
    quantiles = series.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(quantiles.loc[0.50]),
        "sd": float(series.std(ddof=1)),
        "min": float(series.min()),
        "q05": float(quantiles.loc[0.05]),
        "q25": float(quantiles.loc[0.25]),
        "q75": float(quantiles.loc[0.75]),
        "q95": float(quantiles.loc[0.95]),
        "max": float(series.max()),
        "iqr": float(quantiles.loc[0.75] - quantiles.loc[0.25]),
        "effectively_constant": is_effectively_constant(series),
    }


def descriptive_table(data: pd.DataFrame, channels: Sequence[str]) -> pd.DataFrame:
    records = [
        {"channel": channel, **describe_values(data[channel])}
        for channel in channels
    ]
    return pd.DataFrame.from_records(records).set_index("channel")


def subplot_grid(
    item_count: int,
    *,
    columns: int = 4,
    width: float = 4.2,
    height: float = 2.8,
) -> tuple[plt.Figure, np.ndarray]:
    used_columns = min(columns, item_count)
    rows = math.ceil(item_count / used_columns)
    return plt.subplots(
        rows,
        used_columns,
        figsize=(width * used_columns, height * rows),
        squeeze=False,
        constrained_layout=True,
    )


def hide_unused_axes(axes: np.ndarray, used_count: int) -> None:
    for axis in axes.flat[used_count:]:
        axis.set_visible(False)


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, alpha=0.2)
    axis.tick_params(labelsize=8)


def prepare_output(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_csv(table: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    path = prepare_output(output_dir) / filename
    table.to_csv(path)
    return path


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    filename: str,
    dpi: int,
) -> Path:
    path = prepare_output(output_dir) / filename
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path
