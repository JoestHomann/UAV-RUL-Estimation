"""Implement the weighted training-fold mean RUL baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from base import (
    ModelAdapter,
    TrainingSummary,
    root_mean_squared_error,
    sample_weight_values,
    target_values,
    weighted_mean,
)


class MeanBaselineAdapter(ModelAdapter):
    """Predict the sample-weighted mean RUL observed in the training fold."""

    family = "mean_baseline"
    representation = "none"

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        """Learn one weighted mean without consuming telemetry features."""

        started_at = self.start_timer()
        targets = target_values(training_data)
        weights = sample_weight_values(training_data)
        self.mean_rul = weighted_mean(targets, weights)
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
            trainable_parameters=1,
        )
        return self.training_summary

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        """Repeat the fitted training-fold mean for every requested row."""

        return np.full(len(data), self.mean_rul, dtype=np.float64)
