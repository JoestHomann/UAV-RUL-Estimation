"""Implement the combined Ridge and Elastic Net architecture family."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import RobustScaler

from base import (
    ModelAdapter,
    ModelAdapterError,
    TrainingSummary,
    root_mean_squared_error,
    sample_weight_values,
    tabular_values,
    target_values,
)
from models.tabular.elastic_net import create_elastic_net
from models.tabular.ridge import create_ridge


class RegularizedLinearAdapter(ModelAdapter):
    """Fit either Ridge or Elastic Net after training-fold robust scaling."""

    family = "regularized_linear"
    representation = "tabular"

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        """Fit the configured linear variant using only training-fold scaling."""

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
        variant = self.hyperparameters["variant"]
        if variant == "ridge":
            self.estimator = create_ridge(self.hyperparameters)
        elif variant == "elastic_net":
            self.estimator = create_elastic_net(
                self.hyperparameters,
                seed=self.seed,
            )
        else:
            raise ModelAdapterError(f"Unsupported regularized variant {variant!r}")
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
        parameter_count = int(np.size(self.estimator.coef_) + 1)
        self.training_summary = TrainingSummary(
            model_family=self.family,
            seed=self.seed,
            training_rows=len(training_data),
            validation_rows=validation_rows,
            training_seconds=self.elapsed_seconds(started_at),
            epochs_or_iterations=None,
            best_epoch_or_iteration=None,
            best_validation_rmse=validation_rmse,
            trainable_parameters=parameter_count,
        )
        return self.training_summary

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        """Apply stored feature order, robust scaling, and linear estimator."""

        values, _ = tabular_values(data, self.feature_names)
        return np.asarray(
            self.estimator.predict(self.scaler.transform(values)),
            dtype=np.float64,
        )
