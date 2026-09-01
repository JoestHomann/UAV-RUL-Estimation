"""Fold-fitted personalized degradation-onset target builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


DetectorName = Literal["temporal_correlation", "monotonic_health_index"]


class OnsetDetectorError(ValueError):
    """Explain invalid training-only histories or detector settings."""


@dataclass(frozen=True)
class OnsetResult:
    fitting_target: pd.Series
    uav_onsets: pd.DataFrame
    selected_features: tuple[str, ...]
    threshold: float


def _numeric_features(features: pd.DataFrame) -> list[str]:
    names = [name for name in features if pd.api.types.is_numeric_dtype(features[name])]
    if not names:
        raise OnsetDetectorError("Onset detection requires numeric features")
    return names


def build_personalized_targets(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    raw_target: pd.Series,
    *,
    detector: DetectorName,
    minimum_cap: float = 40.0,
    maximum_cap: float = 200.0,
    persistence: int = 3,
) -> OnsetResult:
    """Fit one detector using only the supplied fold-training UAVs."""

    required = {"uav_id", "cutoff"}
    if not required.issubset(metadata):
        raise OnsetDetectorError(f"Metadata is missing {sorted(required - set(metadata))}")
    if not (len(features) == len(metadata) == len(raw_target)):
        raise OnsetDetectorError("Onset detector inputs are not row aligned")
    numeric = _numeric_features(features)
    values = features[numeric].replace([np.inf, -np.inf], np.nan)
    if values.isna().any().any():
        raise OnsetDetectorError("Onset features must be finite")
    cutoff = metadata["cutoff"].to_numpy(dtype=np.float64)
    correlations = {}
    for name in numeric:
        column = values[name].to_numpy(dtype=np.float64)
        correlations[name] = 0.0 if np.std(column) == 0 else float(np.corrcoef(column, cutoff)[0, 1])
    selected = tuple(
        sorted(numeric, key=lambda name: abs(correlations[name]), reverse=True)[: min(12, len(numeric))]
    )
    centers = values[list(selected)].median(axis=0)
    scales = (values[list(selected)].quantile(0.75) - values[list(selected)].quantile(0.25)).replace(0.0, 1.0)
    standardized = (values[list(selected)] - centers) / scales
    signs = pd.Series({name: np.sign(correlations[name]) or 1.0 for name in selected})
    oriented = standardized.mul(signs, axis=1)
    if detector == "temporal_correlation":
        health = oriented.mean(axis=1)
    elif detector == "monotonic_health_index":
        health = oriented.median(axis=1)
    else:
        raise OnsetDetectorError(f"Unknown onset detector {detector!r}")
    work = metadata[["uav_id", "cutoff"]].copy()
    work["raw_target"] = np.asarray(raw_target, dtype=np.float64)
    work["health"] = health.to_numpy(dtype=np.float64)
    early_mask = work.groupby("uav_id")["cutoff"].transform(
        lambda series: series <= series.quantile(0.35)
    )
    threshold = float(work.loc[early_mask, "health"].quantile(0.90))
    onset_records = []
    fitting = np.empty(len(work), dtype=np.float64)
    for uav_id, rows in work.groupby("uav_id", sort=True):
        rows = rows.sort_values("cutoff")
        smoothed = rows["health"].rolling(persistence, min_periods=persistence).mean()
        crossings = rows.loc[smoothed.gt(threshold).to_numpy(), "cutoff"]
        terminal = float((rows["cutoff"] + rows["raw_target"]).median())
        onset = float(crossings.iloc[0]) if not crossings.empty else terminal - 125.0
        cap = float(np.clip(terminal - onset, minimum_cap, maximum_cap))
        fitting[rows.index] = np.minimum(rows["raw_target"].to_numpy(float), cap)
        onset_records.append(
            {"uav_id": str(uav_id), "onset_cycle": onset, "terminal_cycle": terminal, "personalized_cap": cap}
        )
    return OnsetResult(
        fitting_target=pd.Series(fitting, index=raw_target.index, name="fitting_target"),
        uav_onsets=pd.DataFrame.from_records(onset_records),
        selected_features=selected,
        threshold=threshold,
    )
