"""Shared constants and structural checks for Phase 1."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_TRAIN_CSV = REPOSITORY_ROOT / "data" / "train.csv"
DEFAULT_TEST_CSV = REPOSITORY_ROOT / "data" / "test.csv"
STEP_1_ARTIFACT_DIR = SCRIPT_DIR / "1_structural_data_audit" / "artifacts"
STEP_2_ARTIFACT_DIR = SCRIPT_DIR / "2_UAV_grouped_validation_folds" / "artifacts"
STEP_3_ARTIFACT_DIR = SCRIPT_DIR / "3_test_like_validation_scenarios" / "artifacts"
STEP_4_ARTIFACT_DIR = SCRIPT_DIR / "4_training_prefixes" / "artifacts"
STEP_5_ARTIFACT_DIR = SCRIPT_DIR / "5_prefix_feature_engineering" / "artifacts"
STEP_6_ARTIFACT_DIR = SCRIPT_DIR / "6_feature_sets" / "artifacts"
STEP_7_ARTIFACT_DIR = SCRIPT_DIR / "7_fold_fitted_preprocessing" / "artifacts"
STEP_8_ARTIFACT_DIR = SCRIPT_DIR / "8_validation_metrics" / "artifacts"
STEP_9_ARTIFACT_DIR = SCRIPT_DIR / "9_cycle_only_baseline" / "artifacts"
STEP_10_ARTIFACT_DIR = SCRIPT_DIR / "10_automated_leakage_checks" / "artifacts"

ID_COLUMN = "uav_id"
CYCLE_COLUMN = "flight_cycle"
TARGET_COLUMN = "RUL"
TELEMETRY_COLUMNS = [f"telemetry_{number:02d}" for number in range(1, 29)]
REMOVAL_CHANNELS = [
    "telemetry_03",
    "telemetry_08",
    "telemetry_14",
    "telemetry_17",
    "telemetry_20",
    "telemetry_27",
]
DEGRADATION_CHANNELS = [
    "telemetry_07",
    "telemetry_13",
    "telemetry_15",
    "telemetry_16",
    "telemetry_19",
    "telemetry_21",
    "telemetry_22",
    "telemetry_23",
    "telemetry_25",
    "telemetry_28",
]
CONTEXT_CHANNELS = [
    "telemetry_01",
    "telemetry_06",
    "telemetry_18",
    "telemetry_26",
]
STATE_CHANNELS = ["telemetry_07", "telemetry_16"]
WEAK_CHANNELS = [
    channel
    for channel in TELEMETRY_COLUMNS
    if channel not in REMOVAL_CHANNELS + DEGRADATION_CHANNELS + CONTEXT_CHANNELS
]
MODEL_CHANNELS = [
    channel for channel in TELEMETRY_COLUMNS if channel not in REMOVAL_CHANNELS
]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def save_csv(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def load_dataset(path: Path, *, require_target: bool) -> pd.DataFrame:
    data = pd.read_csv(path)
    expected = [ID_COLUMN, CYCLE_COLUMN, *TELEMETRY_COLUMNS]
    if require_target:
        expected.append(TARGET_COLUMN)
    missing = [column for column in expected if column not in data.columns]
    unexpected = [column for column in data.columns if column not in expected]
    if missing or unexpected:
        raise ValueError(
            f"Schema mismatch in {path}: missing={missing}, unexpected={unexpected}"
        )
    if not require_target and TARGET_COLUMN in data.columns:
        raise ValueError(f"Test data unexpectedly contains {TARGET_COLUMN}")
    if data[expected].isna().any().any():
        raise ValueError(f"{path} contains missing values")
    numeric_columns = [CYCLE_COLUMN, *TELEMETRY_COLUMNS]
    if require_target:
        numeric_columns.append(TARGET_COLUMN)
    numeric = data[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{path} contains non-finite values")
    if data.duplicated().any():
        raise ValueError(f"{path} contains duplicate rows")
    if data.duplicated([ID_COLUMN, CYCLE_COLUMN]).any():
        raise ValueError(f"{path} contains duplicate UAV/cycle keys")
    return data.sort_values([ID_COLUMN, CYCLE_COLUMN], kind="stable").reset_index(
        drop=True
    )


def summarize_histories(data: pd.DataFrame, *, require_target: bool) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for uav_id, history in data.groupby(ID_COLUMN, sort=True):
        cycles = history[CYCLE_COLUMN].to_numpy(dtype=int)
        if cycles[0] != 1:
            raise ValueError(f"{uav_id} starts at cycle {cycles[0]}, not cycle 1")
        if len(cycles) > 1 and not np.all(np.diff(cycles) == 1):
            raise ValueError(f"{uav_id} has unordered or non-consecutive cycles")
        record: dict[str, Any] = {
            ID_COLUMN: str(uav_id),
            "row_count": int(len(history)),
            "final_cycle": int(cycles[-1]),
        }
        if require_target:
            lifetimes = (
                history[TARGET_COLUMN].to_numpy(dtype=float) + cycles.astype(float)
            )
            if not np.allclose(lifetimes, lifetimes[0], rtol=0.0, atol=1e-10):
                raise ValueError(f"{uav_id} violates RUL + flight_cycle identity")
            if not np.isclose(history[TARGET_COLUMN].iloc[-1], 0.0):
                raise ValueError(f"{uav_id} does not terminate at RUL 0")
            record["terminal_lifetime"] = float(lifetimes[0])
        records.append(record)
    return pd.DataFrame.from_records(records)


def age_band(cycles: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        cycles,
        bins=[0, 50, 100, 200, np.inf],
        labels=["1-50", "51-100", "101-200", ">200"],
        include_lowest=True,
        right=True,
    )


def channel_role(channel: str) -> str:
    if channel in REMOVAL_CHANNELS:
        return "removal"
    if channel in DEGRADATION_CHANNELS:
        return "degradation"
    if channel in CONTEXT_CHANNELS:
        return "context"
    return "weak"
