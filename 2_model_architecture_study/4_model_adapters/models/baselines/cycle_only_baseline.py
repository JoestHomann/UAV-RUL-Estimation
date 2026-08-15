"""Implement the weighted linear flight-cycle baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from base import (
    ModelAdapter,
    ModelAdapterError,
    TrainingSummary,
    root_mean_squared_error,
    sample_weight_values,
    target_values,
)


class CycleOnlyBaselineAdapter(ModelAdapter):
    """Reproduce the Phase 1 weighted linear cycle-only baseline."""

    family = "cycle_only_baseline"
    representation = "tabular"
    cycle_feature = "feature__flight_cycle"

    @staticmethod
    def _cycle_values(data: Any) -> NDArray[np.float64]:
        """Extract and validate the baseline's single age feature."""

        features = getattr(data, "features", None)
        if features is None or CycleOnlyBaselineAdapter.cycle_feature not in features:
            required_feature = CycleOnlyBaselineAdapter.cycle_feature
            raise ModelAdapterError(
                f"Cycle-only data must contain {required_feature!r}"
            )
        cycles = features[CycleOnlyBaselineAdapter.cycle_feature].to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(cycles).all():
            raise ModelAdapterError("Flight-cycle feature contains non-finite values")
        return cycles

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        """Fit weighted least squares to RUL as a function of flight cycle."""

        started_at = self.start_timer()
        cycles = self._cycle_values(training_data)
        targets = target_values(training_data)
        weights = sample_weight_values(training_data)

        # The explicit weighted least-squares calculation matches the Phase 1
        # reference exactly instead of depending on estimator defaults.
        design = np.column_stack([np.ones(len(cycles)), cycles])
        square_root_weights = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(
            design * square_root_weights[:, None],
            targets * square_root_weights,
            rcond=None,
        )
        self.intercept = float(coefficients[0])
        self.slope = float(coefficients[1])
        self._is_fitted = True

        validation_rmse = None
        validation_rows = 0
        if validation_data is not None:
            validation_rows = len(validation_data)
            validation_rmse = root_mean_squared_error(
                target_values(validation_data),
                self.predict(validation_data),
            )
        self.training_summary = TrainingSummary(
            model_family=self.family,
            seed=self.seed,
            training_rows=len(training_data),
            validation_rows=validation_rows,
            training_seconds=self.elapsed_seconds(started_at),
            epochs_or_iterations=None,
            best_epoch_or_iteration=None,
            best_validation_rmse=validation_rmse,
            trainable_parameters=2,
        )
        return self.training_summary

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        """Apply the fitted intercept and cycle slope."""

        return self.intercept + self.slope * self._cycle_values(data)
