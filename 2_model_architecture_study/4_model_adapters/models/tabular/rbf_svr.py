"""Implement the optional radial-basis support-vector architecture."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

from base import (
    ModelAdapter,
    TrainingSummary,
    root_mean_squared_error,
    sample_weight_values,
    tabular_values,
    target_values,
)


class RBFSVRAdapter(ModelAdapter):
    """Fit the optional robust-scaled radial-basis SVR."""

    family = "rbf_svr"
    representation = "tabular"

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        """Fit robust scaling and the kernel regressor on training rows only."""

        started_at = self.start_timer()
        training_values, self.feature_names = tabular_values(training_data)
        targets = target_values(training_data)
        weights = sample_weight_values(training_data)
        self.scaler = RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=(25.0, 75.0),
            unit_variance=True,
        )
        scaled_training = self.scaler.fit_transform(training_values)
        self.estimator = SVR(
            kernel="rbf",
            C=float(self.hyperparameters["c"]),
            gamma=float(self.hyperparameters["gamma"]),
            epsilon=float(self.hyperparameters["epsilon"]),
        )
        self.estimator.fit(scaled_training, targets, sample_weight=weights)
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
            trainable_parameters=None,
        )
        return self.training_summary

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        """Apply stored scaling and the fitted kernel estimator."""

        values, _ = tabular_values(data, self.feature_names)
        return np.asarray(
            self.estimator.predict(self.scaler.transform(values)),
            dtype=np.float64,
        )
