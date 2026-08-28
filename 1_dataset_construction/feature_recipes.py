"""Declare legacy and Run 5 feature-set membership without selecting a winner."""

from __future__ import annotations

from typing import Any

import pandas as pd

from common import CONTEXT_CHANNELS, DEGRADATION_CHANNELS, channel_role


FEATURE_PREFIX = "feature__"
LEGACY_FEATURE_SETS = (
    "age_only",
    "last_values",
    "screened",
    "all_nonconstant",
)
RUN5_FEATURE_SETS = (
    *LEGACY_FEATURE_SETS,
    "screened_v1",
    "screened_robust",
    "screened_acceleration",
    "screened_compact",
    "all_generated_v2",
)
CATALOG_METADATA_COLUMNS = (
    "feature_name",
    "channel",
    "channel_role",
    "statistic",
    "window",
)

LEGACY_GLOBAL_STATISTICS = {
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
}
LEGACY_WINDOW_STATISTICS = {"mean", "sd", "slope", "delta", "last_minus_mean"}
LEGACY_STATE_STATISTICS = {
    "state_unique_values",
    "state_transition_count",
    "state_transition_rate",
    "state_current_run_length",
    "state_time_since_last_change",
}
CONTEXT_STATISTICS = {
    "last",
    "baseline_mean",
    "baseline_delta",
    "history_mean",
    "history_sd",
    "history_min",
    "history_max",
}
ROBUST_GLOBAL_STATISTICS = {
    "baseline_median",
    "baseline_iqr",
    "baseline_mad",
    "history_median",
    "history_iqr",
    "history_q10",
    "history_q90",
    "median_abs_delta",
    "baseline_robust_z",
    "history_robust_z",
}
ROBUST_WINDOW_STATISTICS = {
    "median",
    "iqr",
    "median_abs_delta",
    "baseline_robust_z",
}
CONTEXT_ROBUST_STATISTICS = {
    "baseline_median",
    "baseline_iqr",
    "baseline_mad",
    "history_median",
    "history_iqr",
    "baseline_robust_z",
    "history_robust_z",
}
COMPACT_DEGRADATION_CHANNELS = {
    "telemetry_07",
    "telemetry_13",
    "telemetry_15",
    "telemetry_16",
    "telemetry_19",
    "telemetry_22",
    "telemetry_23",
    "telemetry_25",
}


def _parse_feature(feature_name: str) -> tuple[str, str, str]:
    body = feature_name.removeprefix(FEATURE_PREFIX)
    if body in {"flight_cycle", "log1p_flight_cycle"}:
        return "", body, ""
    channel, statistic = body.split("__", maxsplit=1)
    if statistic.startswith("w") and (
        "_minus_w" in statistic or "_minus_history_" in statistic
    ):
        return channel, statistic, "contrast"
    if statistic.startswith("w") and "_" in statistic:
        window, base_statistic = statistic.split("_", maxsplit=1)
        return channel, base_statistic, window
    return channel, statistic, ""


def _is_legacy_feature(channel: str, statistic: str, window: str) -> bool:
    if not channel:
        return True
    if window:
        return statistic in LEGACY_WINDOW_STATISTICS
    return statistic in LEGACY_GLOBAL_STATISTICS | LEGACY_STATE_STATISTICS


def _is_screened(channel: str, statistic: str, window: str) -> bool:
    if not channel:
        return True
    if channel in DEGRADATION_CHANNELS:
        return _is_legacy_feature(channel, statistic, window)
    return channel in CONTEXT_CHANNELS and not window and statistic in CONTEXT_STATISTICS


def _is_robust(statistic: str, window: str) -> bool:
    if window and window != "contrast":
        return statistic in ROBUST_WINDOW_STATISTICS
    return not window and statistic in ROBUST_GLOBAL_STATISTICS


def feature_catalog(
    feature_names: list[str],
    *,
    feature_profile: str,
    feature_sets: tuple[str, ...],
) -> pd.DataFrame:
    """Build a catalog whose Boolean set columns match the selected profile."""

    supported = set(LEGACY_FEATURE_SETS if feature_profile == "legacy" else RUN5_FEATURE_SETS)
    unknown = sorted(set(feature_sets) - supported)
    if unknown:
        raise ValueError(
            f"Feature profile {feature_profile!r} does not support sets {unknown}"
        )

    records: list[dict[str, Any]] = []
    for feature_name in feature_names:
        if not feature_name.startswith(FEATURE_PREFIX):
            raise ValueError(f"Invalid generated feature name {feature_name!r}")
        channel, statistic, window = _parse_feature(feature_name)
        age = not channel
        legacy = _is_legacy_feature(channel, statistic, window)
        screened = _is_screened(channel, statistic, window)
        acceleration = window == "contrast"
        robust = _is_robust(statistic, window)
        compact = age or (
            channel in COMPACT_DEGRADATION_CHANNELS and legacy
        ) or (
            channel in CONTEXT_CHANNELS
            and not window
            and statistic in CONTEXT_STATISTICS
        )
        memberships = {
            "age_only": age,
            "last_values": age or statistic == "last",
            "screened": screened,
            "all_nonconstant": legacy,
            "screened_v1": screened,
            "screened_robust": screened
            or (
                channel in DEGRADATION_CHANNELS and robust
            )
            or (
                channel in CONTEXT_CHANNELS
                and not window
                and statistic in CONTEXT_ROBUST_STATISTICS
            ),
            "screened_acceleration": screened
            or (channel in DEGRADATION_CHANNELS and acceleration),
            "screened_compact": compact,
            "all_generated_v2": True,
        }
        record: dict[str, Any] = {
            "feature_name": feature_name,
            "channel": channel,
            "channel_role": "age" if age else channel_role(channel),
            "statistic": statistic,
            "window": window,
        }
        record.update({name: bool(memberships[name]) for name in feature_sets})
        records.append(record)
    return pd.DataFrame.from_records(
        records,
        columns=[*CATALOG_METADATA_COLUMNS, *feature_sets],
    )


def catalog_feature_sets(catalog: pd.DataFrame) -> tuple[str, ...]:
    """Return Boolean membership columns in their declared catalog order."""

    names = tuple(
        column for column in catalog.columns if column not in CATALOG_METADATA_COLUMNS
    )
    if not names:
        raise ValueError("Feature catalog contains no feature-set memberships")
    for name in names:
        values = catalog[name]
        if not values.isin([True, False]).all():
            raise ValueError(f"Feature-set column {name!r} is not Boolean")
    return names
