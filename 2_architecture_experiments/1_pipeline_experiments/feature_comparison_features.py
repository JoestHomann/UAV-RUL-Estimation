"""Frozen causal feature representations for PE_28; no feature selection on labels."""

from dataclasses import replace
import numpy as np
import pandas as pd

SENSORS_22 = tuple(f"telemetry_{i:02d}" for i in
    (1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 18, 19, 21, 22, 23, 24, 25, 26, 28))
SENSORS_14 = tuple(f"telemetry_{i:02d}" for i in (1, 6, 7, 13, 15, 16, 18, 19, 21, 22, 23, 25, 26, 28))
SENSORS_4 = tuple(f"telemetry_{i:02d}" for i in (7, 13, 19, 21))
FEATURE_SETS = {
    "A_current": {"count": 298},
    "B_other": {"count": 266, "sensors": SENSORS_22},
    "C_other_14": {"count": 170, "sensors": SENSORS_14},
    "D_long_windows": {"count": 266, "sensors": SENSORS_22, "windows": (5, 20, 50)},
    "E_compact_22": {"count": 134, "sensors": SENSORS_22, "compact": True},
    "F_baseline_10": {"count": 266, "sensors": SENSORS_22, "baseline": 10},
    "G_all_windows": {"count": 310, "sensors": SENSORS_22, "windows": (5, 10, 20, 50)},
    "H_compact_14": {"count": 86, "sensors": SENSORS_14, "compact": True},
    "I_four_sensors": {"count": 50, "sensors": SENSORS_4},
    "J_recent_slopes": {"count": 310, "sensors": SENSORS_22, "slopes": (5, 20)},
}


def slope(values):
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    x -= x.mean()
    return float(x @ (values - values.mean()) / (x @ x))


def sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def prefix_features(history: pd.DataFrame, cutoff: int, variant: str) -> dict:
    """Use only the requested prefix, even when a complete trajectory is supplied."""
    spec = FEATURE_SETS[variant]
    cycles = history.flight_cycle.to_numpy()
    end = int(np.searchsorted(cycles, cutoff, side="right"))
    if not end or cycles[end - 1] != cutoff:
        raise ValueError("Requested cutoff does not exist in the raw trajectory")
    result = {"feature__flight_cycle": float(cutoff), "feature__log1p_flight_cycle": float(np.log1p(cutoff))}
    for sensor in spec["sensors"]:
        y = history[sensor].to_numpy(dtype=float)[:end]
        base = float(y[:spec.get("baseline", 1)].mean())
        values = {"last": float(y[-1]), "baseline_delta": float(y[-1] - base), "history_slope": slope(y)}
        if spec.get("compact"):
            values.update({"w5_mean": float(y[-5:].mean()), "w20_mean": float(y[-20:].mean()), "w20_sd": sd(y[-20:])})
        else:
            values.update({"history_mean": float(y.mean()), "history_sd": sd(y),
                           "last_minus_history_mean": float(y[-1] - y.mean())})
            for window in spec.get("windows", (5, 10, 20)):
                values[f"w{window}_mean"] = float(y[-window:].mean())
                values[f"w{window}_sd"] = sd(y[-window:])
            for window in spec.get("slopes", ()):
                values[f"w{window}_slope"] = slope(y[-window:])
        result.update({f"feature__{sensor}__{name}": value for name, value in values.items()})
    if len(result) != spec["count"] or not np.isfinite(list(result.values())).all():
        raise ValueError("Feature count or finiteness check failed")
    return result


def validate_raw(raw: pd.DataFrame) -> dict:
    required = {"uav_id", "flight_cycle", "RUL", *SENSORS_22}
    if not required <= set(raw):
        raise ValueError("Raw training data is missing required columns")
    if raw.duplicated(["uav_id", "flight_cycle"]).any():
        raise ValueError("Raw training data contains duplicate cycles")
    histories = {}
    for uav, group in raw.groupby("uav_id", sort=True):
        group = group.sort_values("flight_cycle").reset_index(drop=True)
        if not np.array_equal(group.flight_cycle, np.arange(1, len(group) + 1)):
            raise ValueError("PE_28 requires contiguous observed cycles starting at one")
        if not np.isfinite(group[[*SENSORS_22, "RUL"]].to_numpy(float)).all():
            raise ValueError("Non-finite raw data")
        histories[str(uav)] = group
    return histories


def build_views(training, development, raw: pd.DataFrame):
    """Rebuild features on exactly the existing endpoints; retain labels/weights."""
    histories = validate_raw(raw)
    for data in (training, development):
        if data.target is None or len(data.metadata) != len(data):
            raise ValueError("Missing labels or metadata")
        for row, target in zip(data.metadata.itertuples(), data.target):
            history = histories.get(str(row.uav_id))
            if history is None or not float(row.cutoff).is_integer() or not 1 <= row.cutoff <= len(history):
                raise ValueError("Endpoint is missing from raw training trajectories")
            if not np.isclose(history.RUL.iloc[int(row.cutoff) - 1], target, atol=1e-10, rtol=0):
                raise ValueError("Raw endpoint label differs from the existing evaluation/training label")
    if training.features.shape[1] != FEATURE_SETS["A_current"]["count"]:
        raise ValueError("Current feature set is not the frozen 298-column representation")
    if list(training.features) != list(development.features):
        raise ValueError("Training/development current features differ")
    views = {"A_current": (training, development)}
    # Cache repeated UAV/cutoff pairs, but never infer a feature from a later row.
    endpoints = pd.concat([training.metadata[["uav_id", "cutoff"]], development.metadata[["uav_id", "cutoff"]]]).drop_duplicates()
    for variant in list(FEATURE_SETS)[1:]:
        cache = {(str(row.uav_id), int(row.cutoff)): prefix_features(histories[str(row.uav_id)], int(row.cutoff), variant)
                 for row in endpoints.itertuples()}
        pair = []
        for data in (training, development):
            features = pd.DataFrame([cache[(str(row.uav_id), int(row.cutoff))] for row in data.metadata.itertuples()])
            pair.append(replace(data, features=features))
        views[variant] = tuple(pair)
    return views
