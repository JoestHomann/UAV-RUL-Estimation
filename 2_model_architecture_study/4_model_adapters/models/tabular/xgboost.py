"""Implement the sample-weighted XGBoost architecture."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from xgboost import XGBRegressor

from base import (
    ModelAdapter,
    TrainingSummary,
    root_mean_squared_error,
    sample_weight_values,
    tabular_values,
    target_values,
)


class XGBoostAdapter(ModelAdapter):
    """Fit an unscaled XGBoost regressor with optional validation stopping."""

    family = "xgboost"
    representation = "tabular"
    stochastic = True

    def __init__(
        self,
        *,
        hyperparameters: dict[str, Any],
        seed: int,
        prediction_minimum: float = 0.0,
        early_stopping_patience: int | None = None,
        training_iterations: int | None = None,
    ) -> None:
        """Store inner-stopping or fixed outer-retraining settings."""

        super().__init__(
            hyperparameters=hyperparameters,
            seed=seed,
            prediction_minimum=prediction_minimum,
        )
        self.early_stopping_patience = early_stopping_patience
        self.training_iterations = training_iterations

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        """Fit boosted trees with validation used only during inner selection."""

        started_at = self.start_timer()
        training_values, self.feature_names = tabular_values(training_data)
        targets = target_values(training_data)
        weights = sample_weight_values(training_data)

        maximum_trees = int(self.hyperparameters["maximum_trees"])
        tree_count = self.training_iterations or maximum_trees
        use_early_stopping = (
            validation_data is not None
            and self.training_iterations is None
            and self.early_stopping_patience is not None
        )
        self.estimator = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=tree_count,
            learning_rate=float(self.hyperparameters["learning_rate"]),
            max_depth=int(self.hyperparameters["max_depth"]),
            min_child_weight=float(self.hyperparameters["min_child_weight"]),
            subsample=float(self.hyperparameters["subsample"]),
            colsample_bytree=float(self.hyperparameters["colsample_bytree"]),
            reg_alpha=float(self.hyperparameters["reg_alpha"]),
            reg_lambda=float(self.hyperparameters["reg_lambda"]),
            random_state=self.seed,
            # Candidate-level work is controlled outside the estimator. One
            # worker prevents nested CPU oversubscription.
            n_jobs=1,
            tree_method="hist",
            early_stopping_rounds=(
                int(self.early_stopping_patience) if use_early_stopping else None
            ),
        )

        fit_arguments: dict[str, Any] = {
            "sample_weight": weights,
            "verbose": False,
        }
        if validation_data is not None:
            validation_values, _ = tabular_values(
                validation_data,
                self.feature_names,
            )
            fit_arguments["eval_set"] = [
                (validation_values, target_values(validation_data))
            ]
            fit_arguments["sample_weight_eval_set"] = [
                sample_weight_values(validation_data)
            ]
        self.estimator.fit(training_values, targets, **fit_arguments)
        self._is_fitted = True

        validation_rmse = None
        validation_rows = 0
        if validation_data is not None:
            validation_rows = len(validation_data)
            validation_rmse = root_mean_squared_error(
                target_values(validation_data),
                self.predict(validation_data),
            )
        best_iteration = getattr(self.estimator, "best_iteration", None)
        completed_iterations = (
            int(best_iteration) + 1 if best_iteration is not None else tree_count
        )
        self.training_summary = TrainingSummary(
            model_family=self.family,
            seed=self.seed,
            training_rows=len(training_data),
            validation_rows=validation_rows,
            training_seconds=self.elapsed_seconds(started_at),
            epochs_or_iterations=completed_iterations,
            best_epoch_or_iteration=(
                int(best_iteration) + 1 if best_iteration is not None else None
            ),
            best_validation_rmse=validation_rmse,
            trainable_parameters=None,
        )
        return self.training_summary

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        """Predict from unscaled features in their fitted order."""

        values, _ = tabular_values(data, self.feature_names)
        return np.asarray(self.estimator.predict(values), dtype=np.float64)
