"""Implement the sample-weighted XGBoost architecture."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from xgboost import XGBRegressor, build_info

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


class AsymmetricSquaredObjective:
    """Provide a picklable XGBoost objective with heavier late-RUL errors."""

    def __init__(self, overprediction_weight: float) -> None:
        self.overprediction_weight = float(overprediction_weight)

    def __call__(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        sample_weight: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        residual = np.asarray(y_pred) - np.asarray(y_true)
        weights = np.where(residual > 0.0, self.overprediction_weight, 1.0)
        if sample_weight is not None:
            weights = weights * np.asarray(sample_weight)
        return 2.0 * weights * residual, 2.0 * weights


class SeverityAsymmetricObjective:
    """Increase the marginal penalty as positive residual severity grows."""

    def __init__(self, overprediction_weight: float, severity_scale: float) -> None:
        self.severity = (float(overprediction_weight) - 1.0) / float(
            severity_scale
        )

    def __call__(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        sample_weight: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        residual = np.asarray(y_pred) - np.asarray(y_true)
        positive = np.maximum(residual, 0.0)
        gradient = 2.0 * residual + 3.0 * self.severity * np.square(positive)
        hessian = 2.0 + 6.0 * self.severity * positive
        if sample_weight is not None:
            weights = np.asarray(sample_weight)
            gradient = gradient * weights
            hessian = hessian * weights
        return gradient, hessian


def _xgboost_cuda_available() -> bool:
    """Return whether both XGBoost and the active machine can use CUDA."""

    if not bool(build_info().get("USE_CUDA", False)):
        return False
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_xgboost_device(requested_device: str = "auto") -> str:
    """Resolve an explicit or portable automatic XGBoost device choice."""

    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ModelAdapterError(
            "XGBoost device must be one of 'auto', 'cpu', or 'cuda'"
        )
    cuda_available = _xgboost_cuda_available()
    if requested_device == "cuda" and not cuda_available:
        raise ModelAdapterError(
            "XGBoost CUDA was requested, but the installed build or active "
            "machine does not provide CUDA"
        )
    if requested_device == "auto":
        return "cuda" if cuda_available else "cpu"
    return requested_device


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
        training_monitor: Any | None = None,
        device: str = "auto",
    ) -> None:
        """Store inner-stopping or fixed outer-retraining settings."""

        super().__init__(
            hyperparameters=hyperparameters,
            seed=seed,
            prediction_minimum=prediction_minimum,
            training_monitor=training_monitor,
        )
        self.early_stopping_patience = early_stopping_patience
        self.training_iterations = training_iterations
        self.requested_device = device
        self.device = resolve_xgboost_device(device)

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        """Fit boosted trees with validation used only during inner selection."""

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

        maximum_trees = int(self.hyperparameters["maximum_trees"])
        tree_count = self.training_iterations or maximum_trees
        use_early_stopping = (
            validation_data is not None
            and self.training_iterations is None
            and self.early_stopping_patience is not None
        )
        progress_callback = self.require_training_monitor().create_xgboost_callback(
            self.log_training_step
        )
        objective: str | AsymmetricSquaredObjective | SeverityAsymmetricObjective = (
            "reg:squarederror"
        )
        objective_arguments: dict[str, Any] = {}
        if self.prediction_policy.loss == "asymmetric_mse":
            objective = AsymmetricSquaredObjective(
                self.prediction_policy.overprediction_weight
            )
        elif self.prediction_policy.loss == "severity_asymmetric_mse":
            objective = SeverityAsymmetricObjective(
                self.prediction_policy.overprediction_weight,
                self.prediction_policy.severity_scale,
            )
        elif self.prediction_policy.loss == "quantile":
            objective = "reg:quantileerror"
            objective_arguments["quantile_alpha"] = self.prediction_policy.quantile
        self.estimator = self._new_estimator(
            tree_count=tree_count,
            objective=objective,
            objective_arguments=objective_arguments,
            use_early_stopping=use_early_stopping,
            callbacks=[progress_callback],
            seed=self.seed,
        )

        fit_arguments: dict[str, Any] = {
            "sample_weight": weights,
            "verbose": False,
            # Training data is evaluated only to expose a monitoring curve. It
            # remains the first set, so the last set is still the validation
            # fold used by XGBoost's established early-stopping behavior.
            "eval_set": [(training_values, targets)],
            "sample_weight_eval_set": [weights],
        }
        if validation_data is not None:
            validation_values, validation_names = tabular_values(
                validation_data,
                self.feature_names,
            )
            validation_values, transformed_names = self.signal_transformer.transform(
                validation_values,
                validation_names,
            )
            validation_assignments = self.fault_mode_transformer.assign(
                validation_values,
                transformed_names,
            )
            if self.fault_mode_strategy == "indicator":
                validation_values, transformed_names = (
                    self.fault_mode_transformer.append_indicator(
                        validation_values,
                        transformed_names,
                        validation_assignments,
                    )
                )
            if transformed_names != self.transformed_feature_names:
                raise ModelAdapterError("Transformed XGBoost feature columns changed")
            fit_arguments["eval_set"].append(
                (validation_values, self.fitting_target_values(validation_data))
            )
            fit_arguments["sample_weight_eval_set"].append(
                sample_weight_values(validation_data)
            )
        self.estimator.fit(training_values, targets, **fit_arguments)
        progress_callback.log_final_round(self.estimator.evals_result())
        # The callback owns a temporary reference to this adapter. Removing it
        # after training keeps the persisted XGBoost estimator free of live
        # monitoring objects and unnecessary circular references.
        self.estimator.set_params(callbacks=None)
        best_iteration = getattr(self.estimator, "best_iteration", None)
        completed_iterations = (
            int(best_iteration) + 1 if best_iteration is not None else tree_count
        )
        self.expert_estimators: dict[int, XGBRegressor] = {}
        if self.fault_mode_strategy == "experts":
            for mode in (0, 1):
                mask = (
                    (training_assignments.modes == mode)
                    & training_assignments.trusted
                )
                if np.count_nonzero(mask) < 10:
                    continue
                estimator = self._new_estimator(
                    tree_count=completed_iterations,
                    objective=objective,
                    objective_arguments=objective_arguments,
                    use_early_stopping=False,
                    callbacks=None,
                    seed=self.seed + mode + 1,
                )
                estimator.fit(
                    training_values[mask],
                    targets[mask],
                    sample_weight=weights[mask],
                    verbose=False,
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
            epochs_or_iterations=completed_iterations,
            best_epoch_or_iteration=(
                int(best_iteration) + 1 if best_iteration is not None else None
            ),
            best_validation_rmse=validation_rmse,
            trainable_parameters=None,
        )
        return self.training_summary

    def _new_estimator(
        self,
        *,
        tree_count: int,
        objective: str | AsymmetricSquaredObjective,
        objective_arguments: dict[str, Any],
        use_early_stopping: bool,
        callbacks: list[Any] | None,
        seed: int,
    ) -> XGBRegressor:
        return XGBRegressor(
            objective=objective,
            eval_metric="rmse",
            n_estimators=tree_count,
            learning_rate=float(self.hyperparameters["learning_rate"]),
            max_depth=int(self.hyperparameters["max_depth"]),
            min_child_weight=float(self.hyperparameters["min_child_weight"]),
            subsample=float(self.hyperparameters["subsample"]),
            colsample_bytree=float(self.hyperparameters["colsample_bytree"]),
            reg_alpha=float(self.hyperparameters["reg_alpha"]),
            reg_lambda=float(self.hyperparameters["reg_lambda"]),
            random_state=seed,
            # Candidate-level work is controlled outside the estimator. One
            # worker prevents nested CPU oversubscription.
            n_jobs=1,
            tree_method="hist",
            device=self.device,
            early_stopping_rounds=(
                int(self.early_stopping_patience) if use_early_stopping else None
            ),
            callbacks=callbacks,
            **objective_arguments,
        )

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        """Predict from unscaled features in their fitted order."""

        values, names = tabular_values(data, self.feature_names)
        values, transformed_names = self.signal_transformer.transform(values, names)
        assignments = self.fault_mode_transformer.assign(values, transformed_names)
        if self.fault_mode_strategy == "indicator":
            values, transformed_names = self.fault_mode_transformer.append_indicator(
                values,
                transformed_names,
                assignments,
            )
        if transformed_names != self.transformed_feature_names:
            raise ModelAdapterError("Transformed XGBoost feature columns changed")
        predictions = np.asarray(self.estimator.predict(values), dtype=np.float64)
        if self.fault_mode_strategy == "experts":
            for mode, estimator in self.expert_estimators.items():
                mask = (assignments.modes == mode) & assignments.trusted
                if np.any(mask):
                    predictions[mask] = estimator.predict(values[mask])
        return predictions
