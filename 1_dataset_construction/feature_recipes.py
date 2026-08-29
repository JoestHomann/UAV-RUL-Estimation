"""Declare legacy and extended feature-set membership without choosing a winner."""

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
EXTENDED_FEATURE_SETS = (
    *LEGACY_FEATURE_SETS,
    "screened_v1",
    "screened_robust",
    "screened_acceleration",
    "screened_compact",
    "screened_drift_pruned",
    "screened_drift_replaced",
    "signal_control",
    "signal_family_13_16_22_25_28",
    "signal_family_19_21",
    "signal_family_15_23",
    "signal_family_07",
    "signal_all_families",
    "normalization_raw",
    "normalization_robust",
    "normalization_combined",
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
DRIFT_ABLATION_CHANNELS = {"telemetry_15", "telemetry_16"}
DRIFT_PRUNED_GLOBAL_STATISTICS = {
    "history_sd",
    "history_min",
    "history_max",
    "mean_abs_delta",
    "max_abs_delta",
}
SIGNAL_FAMILIES = {
    "signal_family_13_16_22_25_28": {
        "telemetry_13",
        "telemetry_16",
        "telemetry_22",
        "telemetry_25",
        "telemetry_28",
    },
    "signal_family_19_21": {"telemetry_19", "telemetry_21"},
    "signal_family_15_23": {"telemetry_15", "telemetry_23"},
    "signal_family_07": {"telemetry_07"},
}
SIGNAL_GLOBAL_STATISTICS = {
    "baseline_delta",
    "history_slope",
    "baseline_robust_z",
    "degradation_score",
    "degradation_peak_score",
    "degradation_onset_detected",
    "degradation_onset_cycle",
    "degradation_cycles_since_onset",
    "degradation_change_detected",
    "degradation_change_point_cycle",
    "degradation_cycles_since_change_point",
    "degradation_change_magnitude",
}
SIGNAL_WINDOW_STATISTICS = {
    "slope",
    "delta",
    "last_minus_mean",
    "baseline_robust_z",
}
NORMALIZATION_RAW_GLOBAL_STATISTICS = {"last", "history_slope"}
NORMALIZATION_RAW_WINDOW_STATISTICS = {
    "mean",
    "slope",
    "delta",
    "last_minus_mean",
}
NORMALIZATION_ROBUST_GLOBAL_STATISTICS = {
    "baseline_robust_z",
    "history_slope_robust_scaled",
}
NORMALIZATION_ROBUST_WINDOW_STATISTICS = {
    "baseline_robust_z",
    "slope_robust_scaled",
    "delta_robust_scaled",
    "last_minus_mean_robust_scaled",
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


def _is_drift_heavy(channel: str, statistic: str, window: str) -> bool:
    if channel not in DRIFT_ABLATION_CHANNELS:
        return False
    if not window:
        return statistic in DRIFT_PRUNED_GLOBAL_STATISTICS
    return window == "w50" and statistic == "sd"


def _is_signal_temporal_feature(
    channel: str,
    statistic: str,
    window: str,
) -> bool:
    if not channel:
        return False
    if statistic.startswith("state_"):
        return channel in {"telemetry_07", "telemetry_16"}
    if window == "contrast":
        return statistic.endswith("_slope")
    if window:
        return statistic in SIGNAL_WINDOW_STATISTICS
    return statistic in SIGNAL_GLOBAL_STATISTICS


def _is_normalization_feature(
    channel: str,
    statistic: str,
    window: str,
    *,
    robust: bool,
) -> bool:
    if not channel or window == "contrast":
        return False
    if robust:
        return (
            statistic in NORMALIZATION_ROBUST_WINDOW_STATISTICS
            if window
            else statistic in NORMALIZATION_ROBUST_GLOBAL_STATISTICS
        )
    return (
        statistic in NORMALIZATION_RAW_WINDOW_STATISTICS
        if window
        else statistic in NORMALIZATION_RAW_GLOBAL_STATISTICS
    )


def feature_catalog(
    feature_names: list[str],
    *,
    feature_profile: str,
    feature_sets: tuple[str, ...],
) -> pd.DataFrame:
    """Build a catalog whose Boolean set columns match the selected profile."""

    supported = set(
        LEGACY_FEATURE_SETS
        if feature_profile == "legacy"
        else EXTENDED_FEATURE_SETS
    )
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
        drift_pruned = screened and not _is_drift_heavy(
            channel,
            statistic,
            window,
        )
        drift_replacement = (
            channel in DRIFT_ABLATION_CHANNELS and robust
        )
        compact = age or (
            channel in COMPACT_DEGRADATION_CHANNELS and legacy
        ) or (
            channel in CONTEXT_CHANNELS
            and not window
            and statistic in CONTEXT_STATISTICS
        )
        signal_control = age or statistic == "last"
        signal_memberships = {
            name: signal_control
            or (
                channel in family_channels
                and _is_signal_temporal_feature(channel, statistic, window)
            )
            for name, family_channels in SIGNAL_FAMILIES.items()
        }
        signal_all = signal_control or (
            channel in DEGRADATION_CHANNELS
            and _is_signal_temporal_feature(channel, statistic, window)
        )
        normalization_raw = age or _is_normalization_feature(
            channel,
            statistic,
            window,
            robust=False,
        )
        normalization_robust = age or _is_normalization_feature(
            channel,
            statistic,
            window,
            robust=True,
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
            "screened_drift_pruned": drift_pruned,
            "screened_drift_replaced": drift_pruned or drift_replacement,
            "signal_control": signal_control,
            **signal_memberships,
            "signal_all_families": signal_all,
            "normalization_raw": normalization_raw,
            "normalization_robust": normalization_robust,
            "normalization_combined": normalization_raw or normalization_robust,
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
