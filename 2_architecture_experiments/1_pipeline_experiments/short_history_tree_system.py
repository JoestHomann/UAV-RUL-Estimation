"""Run 7's recipe fitted only to early prefixes and early calibration endpoints."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from advanced_r2_utils import subset_dataset
from models.tabular.residual_corrected_tree_ensemble import ResidualCorrectedTreeEnsembleAdapter


def early_prefixes(data: Any, maximum_cutoff: float, *, require_all_uavs: bool = True) -> Any:
    if not np.isfinite(maximum_cutoff) or maximum_cutoff <= 0:
        raise ValueError("Specialist maximum cutoff must be finite and positive")
    cutoff = data.metadata.cutoff.to_numpy(float)
    if not np.isfinite(cutoff).all() or np.any(cutoff <= 0):
        raise ValueError("Observed history lengths must be finite and positive")
    selected = subset_dataset(data, cutoff <= maximum_cutoff)
    if not len(selected):
        raise ValueError("No eligible early-history endpoints")
    if require_all_uavs and set(selected.metadata.uav_id) != set(data.metadata.uav_id):
        raise ValueError("Early prefixes do not cover every training UAV")
    counts = selected.metadata.uav_id.astype(str).value_counts()
    # Keep the original contract's total weight of one per training UAV.
    weights = selected.metadata.uav_id.astype(str).map(lambda uav: 1.0 / counts[uav])
    return replace(selected, sample_weights=pd.Series(weights.to_numpy(float)))


class ShortHistoryTreeEnsemble(ResidualCorrectedTreeEnsembleAdapter):
    """Reuse the exact tree/residual recipe; specialize its training support only."""

    def __init__(self, *, maximum_cutoff: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.maximum_cutoff = float(maximum_cutoff)

    def _calibration_data(self, training_data: Any) -> Any:
        calibration = early_prefixes(
            super()._calibration_data(training_data), self.maximum_cutoff, require_all_uavs=False
        )
        if calibration.metadata.uav_id.nunique() < self.internal_folds:
            raise ValueError("Too few early-calibration UAVs for the unchanged internal OOF recipe")
        return calibration

    def fit(self, training_data: Any, validation_data: Any | None = None) -> Any:
        return super().fit(early_prefixes(training_data, self.maximum_cutoff), validation_data)
