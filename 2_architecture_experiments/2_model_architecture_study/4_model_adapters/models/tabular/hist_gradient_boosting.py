"""Implement scikit-learn histogram gradient boosting for tabular RUL data."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from base import (
    ModelAdapter,
    TrainingSummary,
    root_mean_squared_error,
    sample_weight_values,
    tabular_values,
    target_values,
)


class HistGradientBoostingAdapter(ModelAdapter):
    """Fit deterministic histogram-based boosted trees without feature scaling."""

    family = "hist_gradient_boosting"
    representation = "tabular"
    stochastic = False

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        """Fit a fixed-iteration estimator using fold-local rows and weights."""

        started_at = self.start_timer()
        training_values, self.feature_names = tabular_values(training_data)
        configured_depth = self.hyperparameters["max_depth"]
        maximum_depth = None if configured_depth == "none" else int(configured_depth)
        self.estimator = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(self.hyperparameters["learning_rate"]),
            max_iter=int(self.hyperparameters["max_iter"]),
            max_leaf_nodes=int(self.hyperparameters["max_leaf_nodes"]),
            max_depth=maximum_depth,
            min_samples_leaf=int(self.hyperparameters["min_samples_leaf"]),
            l2_regularization=float(self.hyperparameters["l2_regularization"]),
            early_stopping=False,
            random_state=self.seed,
        )
        # The experiment runner parallelizes independent studies. Limiting the
        # estimator avoids nested OpenMP thread pools inside each worker.
        with threadpool_limits(limits=1, user_api="openmp"):
            self.estimator.fit(
                training_values,
                self.fitting_target_values(training_data),
                sample_weight=sample_weight_values(training_data),
            )
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
            epochs_or_iterations=int(self.estimator.n_iter_),
            best_epoch_or_iteration=None,
            best_validation_rmse=validation_rmse,
            trainable_parameters=None,
        )
        return self.training_summary

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        """Predict from engineered features in their fitted column order."""

        values, _ = tabular_values(data, self.feature_names)
        with threadpool_limits(limits=1, user_api="openmp"):
            predictions = self.estimator.predict(values)
        return np.asarray(predictions, dtype=np.float64)
