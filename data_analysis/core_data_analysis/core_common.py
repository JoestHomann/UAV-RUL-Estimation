"""Shared loading, validation, statistics, paths, and plotting helpers."""

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
from scipy import stats


ID_COLUMN = "uav_id"
CYCLE_COLUMN = "flight_cycle"
TARGET_COLUMN = "RUL"
TELEMETRY_COLUMNS = [f"telemetry_{number:02d}" for number in range(1, 29)]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "test.csv"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "figures"

DARK_BLUE = "#0072B2"
LIGHT_BLUE = "#56B4E9"
ORANGE = "#D55E00"
GREEN = "#009E73"
YELLOW = "#E69F00"
GRAY = "#777777"
LIGHT_GRAY = "#D9D9D9"
RED = "#C44E52"
PURPLE = "#7B61A8"


def make_parser(
    description: str,
    output_subdirectory: str,
    *,
    include_test: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    if include_test:
        parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=TELEMETRY_COLUMNS,
        default=TELEMETRY_COLUMNS,
        help="Telemetry channels to analyse (default: all 28).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / output_subdirectory,
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def selected_channels(args: argparse.Namespace) -> list[str]:
    return sort_telemetry_channels(args.channels)


def sort_telemetry_channels(channels: Sequence[str]) -> list[str]:
    """Return unique telemetry names in numeric channel order."""
    unique = dict.fromkeys(channels)
    return sorted(unique, key=lambda channel: int(channel.rsplit("_", 1)[1]))


def load_dataset(
    path: Path,
    channels: Sequence[str],
    *,
    require_rul: bool,
) -> pd.DataFrame:
    """Load a dataset and enforce the assumptions needed by these analyses."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    data = pd.read_csv(path)
    required = [ID_COLUMN, CYCLE_COLUMN, *channels]
    if require_rul:
        required.append(TARGET_COLUMN)
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if not require_rul and TARGET_COLUMN in data.columns:
        raise ValueError(f"Test-like dataset unexpectedly contains {TARGET_COLUMN}")
    if data[required].isna().any().any():
        raise ValueError(f"{path} contains missing values in required columns")

    numeric_columns = [CYCLE_COLUMN, *channels]
    if require_rul:
        numeric_columns.append(TARGET_COLUMN)
    try:
        numeric = data[numeric_columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains non-numeric analysis values") from exc
    if not np.isfinite(numeric).all():
        raise ValueError(f"{path} contains non-finite analysis values")
    if data.duplicated([ID_COLUMN, CYCLE_COLUMN]).any():
        count = int(data.duplicated([ID_COLUMN, CYCLE_COLUMN]).sum())
        raise ValueError(f"{path} contains {count} duplicate UAV/cycle keys")

    return data.sort_values([ID_COLUMN, CYCLE_COLUMN], kind="stable").reset_index(
        drop=True
    )


def is_effectively_constant(
    values: pd.Series | np.ndarray,
    *,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
) -> bool:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return True
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    scale = max(abs(minimum), abs(maximum), 1.0)
    return bool(
        maximum - minimum <= absolute_tolerance + relative_tolerance * scale
    )


def safe_correlation(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    method: str,
) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid]
    y_array = y_array[valid]
    if (
        len(x_array) < 3
        or is_effectively_constant(x_array)
        or is_effectively_constant(y_array)
    ):
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(x_array, y_array)[0, 1])
    if method == "spearman":
        result = stats.spearmanr(x_array, y_array)
        return float(result.statistic)
    raise ValueError(f"Unknown correlation method: {method}")


def residualize(values: np.ndarray, control: np.ndarray, *, rank: bool) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    x = np.asarray(control, dtype=float)
    if rank:
        y = stats.rankdata(y)
        x = stats.rankdata(x)
    design = np.column_stack((np.ones(len(x)), x))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coefficients


def robust_scale(values: pd.Series | np.ndarray) -> tuple[float, float]:
    """Return median and a robust SD-like scale, with safe fallbacks."""
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = 1.4826 * mad
    if scale <= 1e-15:
        q25, q75 = np.quantile(array, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
    if scale <= 1e-15:
        scale = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return median, scale


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    x_centered = x_array - x_array.mean()
    denominator = float(np.square(x_centered).sum())
    if denominator <= 0 or is_effectively_constant(y_array):
        return 0.0
    return float(np.dot(x_centered, y_array - y_array.mean()) / denominator)


def subplot_grid(
    item_count: int,
    *,
    columns: int = 4,
    width: float = 4.2,
    height: float = 2.8,
) -> tuple[plt.Figure, np.ndarray]:
    used_columns = min(columns, max(item_count, 1))
    rows = math.ceil(item_count / used_columns)
    return plt.subplots(
        rows,
        used_columns,
        figsize=(width * used_columns, height * rows),
        squeeze=False,
        constrained_layout=True,
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, alpha=0.2)
    axis.tick_params(labelsize=8)


def save_table(table: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    table.to_csv(path, index=False)
    return path


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    filename: str,
    dpi: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path
