"""Implement baseline and classical tabular model adapters.

Every adapter consumes the same TabularDataset interface from Step 2. Models
that require comparable feature scales fit their scaler inside the adapter from
the supplied training rows only. Tree models receive unchanged feature values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from base import (
    ModelAdapter,
    ModelAdapterError,
    TrainingSummary,
    root_mean_squared_error,
    sample_weight_values,
    tabular_values,
    target_values,
    weighted_mean,
)


class MeanBaselineAdapter(ModelAdapter):
    """Predict the weighted mean RUL observed in the training fold."""

    family = "mean_baseline"
    representation = "none"

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
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
        return np.full(len(data), self.mean_rul, dtype=np.float64)


class CycleOnlyBaselineAdapter(ModelAdapter):
    """Reproduce the Phase 1 weighted linear cycle-only baseline."""

    family = "cycle_only_baseline"
    representation = "tabular"
    cycle_feature = "feature__flight_cycle"

    @staticmethod
    def _cycle_values(data: Any) -> NDArray[np.float64]:
        """Extract the single age feature used by the baseline."""

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
        started_at = self.start_timer()
        cycles = self._cycle_values(training_data)
        targets = target_values(training_data)
        weights = sample_weight_values(training_data)

        # Weighted least squares is written explicitly to match the established
        # Phase 1 baseline rather than depending on estimator defaults.
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
        return self.intercept + self.slope * self._cycle_values(data)


class RegularizedLinearAdapter(ModelAdapter):
    """Fit either Ridge or Elastic Net after training-fold robust scaling."""

    family = "regularized_linear"
    representation = "tabular"

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
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
            self.estimator = Ridge(
                alpha=float(self.hyperparameters["ridge_alpha"]),
            )
        elif variant == "elastic_net":
            self.estimator = ElasticNet(
                alpha=float(self.hyperparameters["elastic_net_alpha"]),
                l1_ratio=float(self.hyperparameters["elastic_net_l1_ratio"]),
                max_iter=20_000,
                random_state=self.seed,
                selection="cyclic",
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
        values, _ = tabular_values(data, self.feature_names)
        return np.asarray(
            self.estimator.predict(self.scaler.transform(values)),
            dtype=np.float64,
        )


class RandomForestAdapter(ModelAdapter):
    """Fit an unscaled, sample-weighted random forest regressor."""

    family = "random_forest"
    representation = "tabular"

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        started_at = self.start_timer()
        training_values, self.feature_names = tabular_values(training_data)
        targets = target_values(training_data)
        weights = sample_weight_values(training_data)
        configured_depth = self.hyperparameters["max_depth"]
        maximum_depth = None if configured_depth == "none" else int(configured_depth)

        self.estimator = RandomForestRegressor(
            n_estimators=int(self.hyperparameters["n_estimators"]),
            max_depth=maximum_depth,
            min_samples_leaf=int(self.hyperparameters["min_samples_leaf"]),
            max_features=float(self.hyperparameters["max_features"]),
            random_state=self.seed,
            # Candidate-level parallelism belongs to the later experiment
            # runner. One worker here prevents nested parallelism and keeps
            # resource use comparable across candidates.
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
        values, _ = tabular_values(data, self.feature_names)
        return np.asarray(self.estimator.predict(values), dtype=np.float64)


class XGBoostAdapter(ModelAdapter):
    """Fit an unscaled XGBoost regressor with optional validation stopping."""

    family = "xgboost"
    representation = "tabular"

    def __init__(
        self,
        *,
        hyperparameters: dict[str, Any],
        seed: int,
        prediction_minimum: float = 0.0,
        early_stopping_patience: int | None = None,
        training_iterations: int | None = None,
    ) -> None:
        super().__init__(
            hyperparameters=hyperparameters,
            seed=seed,
            prediction_minimum=prediction_minimum,
        )
        self.early_stopping_patience = early_stopping_patience
        self.training_iterations = training_iterations

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
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
            # As with Random Forest, the runner may parallelize candidates.
            # Keeping one worker inside the estimator avoids oversubscription.
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
        values, _ = tabular_values(data, self.feature_names)
        return np.asarray(self.estimator.predict(values), dtype=np.float64)


class RBFSVRAdapter(ModelAdapter):
    """Fit the optional robust-scaled radial-basis support-vector regressor."""

    family = "rbf_svr"
    representation = "tabular"

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
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
        values, _ = tabular_values(data, self.feature_names)
        return np.asarray(
            self.estimator.predict(self.scaler.transform(values)),
            dtype=np.float64,
        )
