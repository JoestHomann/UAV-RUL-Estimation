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
    save_json,
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
FEATURE_PROFILES = ("legacy", "extended")
DEGRADATION_DIRECTIONS = {
    "telemetry_07": 1.0,
    "telemetry_13": 1.0,
    "telemetry_15": 1.0,
    "telemetry_16": -1.0,
    "telemetry_19": 1.0,
    "telemetry_21": 1.0,
    "telemetry_22": 1.0,
    "telemetry_23": -1.0,
    "telemetry_25": -1.0,
    "telemetry_28": -1.0,
}
DEGRADATION_THRESHOLD = 2.0
DEGRADATION_PERSISTENCE = 3
CHANGE_POINT_MIN_SEGMENT = 5
CHANGE_POINT_MIN_MAGNITUDE = 1.0


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


def robust_location_scale(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return median, IQR, MAD, and a safe SD-like robust scale."""

    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = float(q75 - q25)
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = iqr / 1.349
    if scale <= 1e-12:
        scale = sample_sd(values)
    if scale <= 1e-12:
        scale = 1.0
    return median, iqr, mad, float(scale)


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


def first_sustained_threshold(
    scores: np.ndarray,
    *,
    start: int,
    threshold: float = DEGRADATION_THRESHOLD,
    persistence: int = DEGRADATION_PERSISTENCE,
) -> int | None:
    """Return the first prefix-local sustained degradation threshold index."""

    final_start = len(scores) - persistence
    for index in range(start, final_start + 1):
        if np.all(scores[index : index + persistence] >= threshold):
            return index
    return None


def degradation_change_point(
    scores: np.ndarray,
    *,
    baseline_length: int,
) -> tuple[int | None, float]:
    """Find the strongest positive mean shift using only the observed prefix."""

    first_split = max(baseline_length, CHANGE_POINT_MIN_SEGMENT)
    final_split = len(scores) - CHANGE_POINT_MIN_SEGMENT
    if first_split > final_split:
        return None, 0.0

    cumulative = np.cumsum(scores, dtype=float)
    total = float(cumulative[-1])
    best_split: int | None = None
    best_magnitude = float("-inf")
    for split in range(first_split, final_split + 1):
        before_mean = float(cumulative[split - 1] / split)
        after_mean = float((total - cumulative[split - 1]) / (len(scores) - split))
        magnitude = after_mean - before_mean
        if magnitude > best_magnitude:
            best_split = split
            best_magnitude = magnitude

    if best_magnitude < CHANGE_POINT_MIN_MAGNITUDE:
        return None, max(best_magnitude, 0.0)
    return best_split, best_magnitude


def extract_prefix_features(
    history: pd.DataFrame,
    cutoff: int,
    *,
    channels: list[str] = MODEL_CHANNELS,
    rolling_windows: tuple[int, ...] = DEFAULT_WINDOWS,
    baseline_window: int = 10,
    feature_profile: str = "legacy",
) -> dict[str, float]:
    if feature_profile not in FEATURE_PROFILES:
        raise ValueError(
            f"Unknown feature profile {feature_profile!r}; choose from "
            f"{FEATURE_PROFILES}"
        )
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
        window_summaries: dict[int, dict[str, float]] = {}
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
            window_summaries[int(window)] = window_values
            for statistic, value in window_values.items():
                features[
                    f"{FEATURE_PREFIX}{channel}__w{window}_{statistic}"
                ] = value

        if feature_profile == "extended":
            (
                baseline_median,
                baseline_iqr,
                baseline_mad,
                baseline_scale,
            ) = robust_location_scale(baseline)
            history_median, history_iqr, _, history_scale = robust_location_scale(
                values
            )
            q10, q90 = np.quantile(values, [0.10, 0.90])
            robust_global_values = {
                "baseline_median": baseline_median,
                "baseline_iqr": baseline_iqr,
                "baseline_mad": baseline_mad,
                "history_median": history_median,
                "history_iqr": history_iqr,
                "history_q10": float(q10),
                "history_q90": float(q90),
                "median_abs_delta": (
                    float(np.median(np.abs(differences)))
                    if len(differences)
                    else 0.0
                ),
                "baseline_robust_z": float(
                    (values[-1] - baseline_median) / baseline_scale
                ),
                "history_robust_z": float(
                    (values[-1] - history_median) / history_scale
                ),
                "history_slope_robust_scaled": float(
                    global_values["history_slope"] / baseline_scale
                ),
            }
            for statistic, value in robust_global_values.items():
                features[f"{FEATURE_PREFIX}{channel}__{statistic}"] = value

            for window in rolling_windows:
                recent_values = values[-window:]
                recent_differences = np.diff(recent_values)
                recent_median, recent_iqr, _, _ = robust_location_scale(
                    recent_values
                )
                robust_window_values = {
                    "median": recent_median,
                    "iqr": recent_iqr,
                    "median_abs_delta": (
                        float(np.median(np.abs(recent_differences)))
                        if len(recent_differences)
                        else 0.0
                    ),
                    "baseline_robust_z": float(
                        (recent_median - baseline_median) / baseline_scale
                    ),
                    "slope_robust_scaled": float(
                        window_summaries[int(window)]["slope"] / baseline_scale
                    ),
                    "delta_robust_scaled": float(
                        window_summaries[int(window)]["delta"] / baseline_scale
                    ),
                    "last_minus_mean_robust_scaled": float(
                        window_summaries[int(window)]["last_minus_mean"]
                        / baseline_scale
                    ),
                }
                for statistic, value in robust_window_values.items():
                    features[
                        f"{FEATURE_PREFIX}{channel}__w{window}_{statistic}"
                    ] = value

            ordered_windows = sorted(window_summaries)
            for shorter, longer in zip(
                ordered_windows,
                ordered_windows[1:],
                strict=False,
            ):
                for statistic in ("mean", "sd", "slope"):
                    value = (
                        window_summaries[shorter][statistic]
                        - window_summaries[longer][statistic]
                    )
                    features[
                        f"{FEATURE_PREFIX}{channel}__w{shorter}_minus_"
                        f"w{longer}_{statistic}"
                    ] = float(value)
            for window in ordered_windows:
                features[
                    f"{FEATURE_PREFIX}{channel}__w{window}_minus_history_mean"
                ] = float(
                    window_summaries[window]["mean"]
                    - global_values["history_mean"]
                )

            direction = DEGRADATION_DIRECTIONS.get(channel)
            if direction is not None:
                degradation_scores = (
                    direction * (values - baseline_median) / baseline_scale
                )
                baseline_length = len(baseline)
                onset_index = first_sustained_threshold(
                    degradation_scores,
                    start=baseline_length,
                )
                change_index, change_magnitude = degradation_change_point(
                    degradation_scores,
                    baseline_length=baseline_length,
                )
                degradation_values = {
                    "degradation_score": float(degradation_scores[-1]),
                    "degradation_peak_score": float(
                        np.max(degradation_scores[baseline_length:])
                        if len(degradation_scores) > baseline_length
                        else np.max(degradation_scores)
                    ),
                    "degradation_onset_detected": float(onset_index is not None),
                    "degradation_onset_cycle": (
                        float(cycles[onset_index]) if onset_index is not None else 0.0
                    ),
                    "degradation_cycles_since_onset": (
                        float(cycles[-1] - cycles[onset_index])
                        if onset_index is not None
                        else 0.0
                    ),
                    "degradation_change_detected": float(change_index is not None),
                    "degradation_change_point_cycle": (
                        float(cycles[change_index]) if change_index is not None else 0.0
                    ),
                    "degradation_cycles_since_change_point": (
                        float(cycles[-1] - cycles[change_index])
                        if change_index is not None
                        else 0.0
                    ),
                    "degradation_change_magnitude": float(change_magnitude),
                }
                for statistic, value in degradation_values.items():
                    features[f"{FEATURE_PREFIX}{channel}__{statistic}"] = value

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
    feature_profile: str = "legacy",
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
                histories[uav_id],
                cutoff,
                channels=channels,
                feature_profile=feature_profile,
            )
        feature_records.append(cache[key])
    feature_frame = pd.DataFrame.from_records(feature_records, index=manifest.index)
    result = pd.concat([manifest.reset_index(drop=True), feature_frame], axis=1)
    feature_values = result.filter(like=FEATURE_PREFIX).to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError("Generated feature table contains missing or non-finite values")
    return result


def assert_feature_causality(
    train: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    feature_profile: str = "legacy",
) -> None:
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
        original = extract_prefix_features(
            history,
            cutoff,
            feature_profile=feature_profile,
        )
        altered = history.copy()
        future = altered[CYCLE_COLUMN] > cutoff
        altered.loc[future, MODEL_CHANNELS] = (
            altered.loc[future, MODEL_CHANNELS] * -17.0 + 1_000_000.0
        )
        modified = extract_prefix_features(
            altered,
            cutoff,
            feature_profile=feature_profile,
        )
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
    parser.add_argument(
        "--feature-profile",
        choices=FEATURE_PROFILES,
        default="legacy",
    )
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
    assert_feature_causality(
        train,
        manifests["training_features.csv.gz"][1],
        feature_profile=args.feature_profile,
    )

    feature_dir = args.output_dir
    generated_paths: list[Path] = []
    for filename, (dataset, manifest) in manifests.items():
        table = build_feature_table(
            dataset,
            manifest,
            feature_profile=args.feature_profile,
        )
        generated_paths.append(save_feature_table(table, feature_dir / filename))
    feature_count = len(
        [
            column
            for column in table.columns
            if column.startswith(FEATURE_PREFIX)
        ]
    )
    generated_paths.append(
        save_json(
            {
                "feature_profile": args.feature_profile,
                "feature_count": feature_count,
                "baseline_window": 10,
                "rolling_windows": list(DEFAULT_WINDOWS),
                "causal_prefix_only": True,
            },
            feature_dir / "feature_generation_config.json",
        )
    )
    print("\n".join(f"Saved {path}" for path in generated_paths))


if __name__ == "__main__":
    main()
