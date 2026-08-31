"""Implement the sample-weighted Extra Trees architecture."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesRegressor

from base import (
    ModelAdapter,
    ModelAdapterError,
    TrainingSummary,
    root_mean_squared_error,
    sample_weight_values,
    tabular_values,
    target_values,
)
from models.tabular.fold_fitted_transforms import (
    FaultModeTransformer,
    SignalCompressionTransformer,
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
        self.signal_transformer = SignalCompressionTransformer(
            str(self.hyperparameters["signal_compression_strategy"])
        )
        training_values, self.transformed_feature_names = (
            self.signal_transformer.fit_transform(training_values, self.feature_names)
        )
        self.fault_mode_strategy = str(
            self.hyperparameters["fault_mode_strategy"]
        )
        self.fault_mode_transformer = FaultModeTransformer(
            self.fault_mode_strategy,
            seed=self.seed,
        )
        training_assignments = self.fault_mode_transformer.fit(
            training_values,
            self.transformed_feature_names,
            training_data.metadata,
        )
        if self.fault_mode_strategy == "indicator":
            training_values, self.transformed_feature_names = (
                self.fault_mode_transformer.append_indicator(
                    training_values,
                    self.transformed_feature_names,
                    training_assignments,
                )
            )
        targets = self.fitting_target_values(training_data)
        weights = sample_weight_values(training_data)

        self.estimator = self._new_estimator(seed=self.seed)
        self.estimator.fit(training_values, targets, sample_weight=weights)
        self.expert_estimators: dict[int, ExtraTreesRegressor] = {}
        if self.fault_mode_strategy == "experts":
            for mode in (0, 1):
                mask = (
                    (training_assignments.modes == mode)
                    & training_assignments.trusted
                )
                if np.count_nonzero(mask) < 10:
                    continue
                estimator = self._new_estimator(seed=self.seed + mode + 1)
                estimator.fit(
                    training_values[mask],
                    targets[mask],
                    sample_weight=weights[mask],
                )
                self.expert_estimators[mode] = estimator
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

    def _new_estimator(self, *, seed: int) -> ExtraTreesRegressor:
        configured_depth = self.hyperparameters["max_depth"]
        maximum_depth = None if configured_depth == "none" else int(configured_depth)
        return ExtraTreesRegressor(
            n_estimators=int(self.hyperparameters["n_estimators"]),
            max_depth=maximum_depth,
            min_samples_leaf=int(self.hyperparameters["min_samples_leaf"]),
            max_features=float(self.hyperparameters["max_features"]),
            bootstrap=False,
            random_state=seed,
            n_jobs=1,
        )

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        """Predict from engineered features in the fitted column order."""

        values, names = tabular_values(data, self.feature_names)
        values, transformed_names = self.signal_transformer.transform(values, names)
        assignments = self.fault_mode_transformer.assign(
            values,
            transformed_names,
        )
        if self.fault_mode_strategy == "indicator":
            values, transformed_names = self.fault_mode_transformer.append_indicator(
                values,
                transformed_names,
                assignments,
            )
        if transformed_names != self.transformed_feature_names:
            raise ModelAdapterError("Transformed ExtraTrees feature columns changed")
        predictions = np.asarray(self.estimator.predict(values), dtype=np.float64)
        if self.fault_mode_strategy == "experts":
            for mode, estimator in self.expert_estimators.items():
                mask = (assignments.modes == mode) & assignments.trusted
                if np.any(mask):
                    predictions[mask] = estimator.predict(values[mask])
        return predictions
