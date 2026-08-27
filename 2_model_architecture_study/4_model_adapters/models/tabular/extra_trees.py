"""Implement the sample-weighted Extra Trees architecture."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesRegressor

from base import (
    ModelAdapter,
    TrainingSummary,
    root_mean_squared_error,
    sample_weight_values,
    tabular_values,
    target_values,
)


class ExtraTreesAdapter(ModelAdapter):
    """Fit unscaled extremely randomized regression trees."""

    family = "extra_trees"
    representation = "tabular"
    stochastic = True

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        """Fit randomized splits to unchanged engineered features."""

        started_at = self.start_timer()
        training_values, self.feature_names = tabular_values(training_data)
        targets = target_values(training_data)
        weights = sample_weight_values(training_data)
        configured_depth = self.hyperparameters["max_depth"]
        maximum_depth = None if configured_depth == "none" else int(configured_depth)

        self.estimator = ExtraTreesRegressor(
            n_estimators=int(self.hyperparameters["n_estimators"]),
            max_depth=maximum_depth,
            min_samples_leaf=int(self.hyperparameters["min_samples_leaf"]),
            max_features=float(self.hyperparameters["max_features"]),
            bootstrap=False,
            random_state=self.seed,
            n_jobs=1,
        )
        self.estimator.fit(training_values, targets, sample_weight=weights)
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
            epochs_or_iterations=int(self.hyperparameters["n_estimators"]),
            best_epoch_or_iteration=None,
            best_validation_rmse=validation_rmse,
            trainable_parameters=None,
        )
        return self.training_summary

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        """Predict from engineered features in the fitted column order."""

        values, _ = tabular_values(data, self.feature_names)
        return np.asarray(self.estimator.predict(values), dtype=np.float64)
