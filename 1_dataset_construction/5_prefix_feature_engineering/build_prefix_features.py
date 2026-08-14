"""Build leakage-safe features from UAV history prefixes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PHASE_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from common import (
    CYCLE_COLUMN,
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
    ID_COLUMN,
    MODEL_CHANNELS,
    STATE_CHANNELS,
    STEP_3_ARTIFACT_DIR,
    STEP_4_ARTIFACT_DIR,
    STEP_5_ARTIFACT_DIR,
    load_dataset,
)


FEATURE_PREFIX = "feature__"
DEFAULT_WINDOWS = (5, 20, 50)
GLOBAL_STATISTICS = (
    "last",
    "first",
    "baseline_mean",
    "baseline_delta",
    "history_mean",
    "history_sd",
    "history_min",
    "history_max",
    "history_slope",
    "last_delta",
    "mean_abs_delta",
    "max_abs_delta",
)
WINDOW_STATISTICS = ("mean", "sd", "slope", "delta", "last_minus_mean")
STATE_STATISTICS = (
    "unique_values",
    "transition_count",
    "transition_rate",
    "current_run_length",
    "time_since_last_change",
)


def linear_slope(cycles: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    centered_x = cycles.astype(float) - float(np.mean(cycles))
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0:
        return 0.0
    centered_y = values - float(np.mean(values))
    return float(np.dot(centered_x, centered_y) / denominator)


def sample_sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def current_run_length(values: np.ndarray) -> int:
    if len(values) == 0:
        return 0
    last = values[-1]
    length = 1
    for value in values[-2::-1]:
        if value != last:
            break
        length += 1
    return length


def extract_prefix_features(
    history: pd.DataFrame,
    cutoff: int,
    *,
    channels: list[str] = MODEL_CHANNELS,
    rolling_windows: tuple[int, ...] = DEFAULT_WINDOWS,
    baseline_window: int = 10,
) -> dict[str, float]:
    cycles_full = history[CYCLE_COLUMN].to_numpy(dtype=int)
    position = int(np.searchsorted(cycles_full, cutoff, side="right"))
    if position == 0 or cycles_full[position - 1] != cutoff:
        raise ValueError(
            f"{history[ID_COLUMN].iloc[0]} has no observation at cutoff {cutoff}"
        )
    prefix = history.iloc[:position]
    cycles = prefix[CYCLE_COLUMN].to_numpy(dtype=float)
    features: dict[str, float] = {
        f"{FEATURE_PREFIX}flight_cycle": float(cutoff),
        f"{FEATURE_PREFIX}log1p_flight_cycle": float(np.log1p(cutoff)),
    }

    for channel in channels:
        values = prefix[channel].to_numpy(dtype=float)
        differences = np.diff(values)
        baseline = values[: min(baseline_window, len(values))]
        baseline_mean = float(np.mean(baseline))
        global_values = {
            "last": float(values[-1]),
            "first": float(values[0]),
            "baseline_mean": baseline_mean,
            "baseline_delta": float(values[-1] - baseline_mean),
            "history_mean": float(np.mean(values)),
            "history_sd": sample_sd(values),
            "history_min": float(np.min(values)),
            "history_max": float(np.max(values)),
            "history_slope": linear_slope(cycles, values),
            "last_delta": float(differences[-1]) if len(differences) else 0.0,
            "mean_abs_delta": (
                float(np.mean(np.abs(differences))) if len(differences) else 0.0
            ),
            "max_abs_delta": (
                float(np.max(np.abs(differences))) if len(differences) else 0.0
            ),
        }
        for statistic, value in global_values.items():
            features[f"{FEATURE_PREFIX}{channel}__{statistic}"] = value

        for window in rolling_windows:
            recent_values = values[-window:]
            recent_cycles = cycles[-window:]
            recent_mean = float(np.mean(recent_values))
            window_values = {
                "mean": recent_mean,
                "sd": sample_sd(recent_values),
                "slope": linear_slope(recent_cycles, recent_values),
                "delta": float(recent_values[-1] - recent_values[0]),
                "last_minus_mean": float(recent_values[-1] - recent_mean),
            }
            for statistic, value in window_values.items():
                features[
                    f"{FEATURE_PREFIX}{channel}__w{window}_{statistic}"
                ] = value

        if channel in STATE_CHANNELS:
            transition_count = int(np.count_nonzero(differences != 0))
            run_length = current_run_length(values)
            state_values = {
                "unique_values": float(np.unique(values).size),
                "transition_count": float(transition_count),
                "transition_rate": float(transition_count / max(len(values) - 1, 1)),
                "current_run_length": float(run_length),
                "time_since_last_change": float(max(run_length - 1, 0)),
            }
            for statistic, value in state_values.items():
                features[f"{FEATURE_PREFIX}{channel}__state_{statistic}"] = value
    return features


def build_feature_table(
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    channels: list[str] = MODEL_CHANNELS,
) -> pd.DataFrame:
    histories = {
        str(uav_id): history.reset_index(drop=True)
        for uav_id, history in data.groupby(ID_COLUMN, sort=True)
    }
    cache: dict[tuple[str, int], dict[str, float]] = {}
    feature_records: list[dict[str, float]] = []
    for row in manifest.itertuples(index=False):
        uav_id = str(getattr(row, ID_COLUMN))
        cutoff = int(row.cutoff)
        key = (uav_id, cutoff)
        if key not in cache:
            cache[key] = extract_prefix_features(
                histories[uav_id], cutoff, channels=channels
            )
        feature_records.append(cache[key])
    feature_frame = pd.DataFrame.from_records(feature_records, index=manifest.index)
    result = pd.concat([manifest.reset_index(drop=True), feature_frame], axis=1)
    feature_values = result.filter(like=FEATURE_PREFIX).to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError("Generated feature table contains missing or non-finite values")
    return result


def assert_feature_causality(train: pd.DataFrame, manifest: pd.DataFrame) -> None:
    histories = {
        str(uav_id): history.reset_index(drop=True)
        for uav_id, history in train.groupby(ID_COLUMN, sort=True)
    }
    checked = 0
    for row in manifest.sort_values([ID_COLUMN, "cutoff"]).itertuples(index=False):
        uav_id = str(getattr(row, ID_COLUMN))
        cutoff = int(row.cutoff)
        history = histories[uav_id]
        if cutoff >= int(history[CYCLE_COLUMN].max()):
            continue
        original = extract_prefix_features(history, cutoff)
        altered = history.copy()
        future = altered[CYCLE_COLUMN] > cutoff
        altered.loc[future, MODEL_CHANNELS] = (
            altered.loc[future, MODEL_CHANNELS] * -17.0 + 1_000_000.0
        )
        modified = extract_prefix_features(altered, cutoff)
        if original.keys() != modified.keys():
            raise AssertionError("Feature keys changed after future rows were modified")
        if not np.array_equal(
            np.fromiter(original.values(), dtype=float),
            np.fromiter(modified.values(), dtype=float),
        ):
            raise AssertionError(
                f"Future rows changed features for {uav_id} at cutoff {cutoff}"
            )
        checked += 1
        if checked >= 10:
            break
    if checked < 10:
        raise AssertionError("Too few samples were available for the causality check")


def save_feature_table(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, compression="gzip")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--scenario-dir", type=Path, default=STEP_3_ARTIFACT_DIR)
    parser.add_argument("--prefix-dir", type=Path, default=STEP_4_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=STEP_5_ARTIFACT_DIR)
    args = parser.parse_args()

    train = load_dataset(args.train_csv, require_target=True)
    test = load_dataset(args.test_csv, require_target=False)
    manifests = {
        "training_features.csv.gz": (
            train,
            pd.read_csv(args.prefix_dir / "training_prefixes.csv"),
        ),
        "locked_validation_features.csv.gz": (
            train,
            pd.read_csv(args.scenario_dir / "locked_validation_scenarios.csv"),
        ),
        "development_validation_features.csv.gz": (
            train,
            pd.read_csv(args.scenario_dir / "development_validation_scenarios.csv"),
        ),
        "test_features.csv.gz": (
            test,
            pd.read_csv(args.scenario_dir / "test_endpoints.csv"),
        ),
    }
    assert_feature_causality(train, manifests["training_features.csv.gz"][1])

    feature_dir = args.output_dir
    generated_paths: list[Path] = []
    for filename, (dataset, manifest) in manifests.items():
        table = build_feature_table(dataset, manifest)
        generated_paths.append(save_feature_table(table, feature_dir / filename))
    print("\n".join(f"Saved {path}" for path in generated_paths))


if __name__ == "__main__":
    main()
