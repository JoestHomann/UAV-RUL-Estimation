"""Censored-survival and discrete-horizon tabular RUL architectures."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
import xgboost as xgb
from xgboost import XGBClassifier

from base import (
    ModelAdapter,
    ModelAdapterError,
    TrainingSummary,
    root_mean_squared_error,
    sample_weight_values,
    tabular_values,
    target_values,
)
from models.tabular.xgboost import resolve_xgboost_device


def _positive_targets(data: Any) -> NDArray[np.float64]:
    values = target_values(data)
    if np.any(values < 0.0):
        raise ModelAdapterError("RUL targets must be nonnegative")
    return np.maximum(values, 1.0e-3)


class XGBoostAFTAdapter(ModelAdapter):
    """Fit accelerated failure time trees with a right-censored RUL tail."""

    family = "xgboost_aft"
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
        super().__init__(
            hyperparameters=hyperparameters,
            seed=seed,
            prediction_minimum=prediction_minimum,
            training_monitor=training_monitor,
        )
        self.early_stopping_patience = early_stopping_patience
        self.training_iterations = training_iterations
        self.device = resolve_xgboost_device(device)

    def _matrix(self, data: Any, *, training: bool) -> xgb.DMatrix:
        values, names = tabular_values(
            data, None if training else self.feature_names
        )
        if training:
            self.feature_names = names
        matrix = xgb.DMatrix(values, feature_names=names)
        labels = _positive_targets(data)
        threshold = float(self.hyperparameters["censoring_threshold"])
        censored = labels > threshold
        lower = np.where(censored, threshold, labels)
        upper = np.where(censored, np.inf, labels)
        matrix.set_float_info("label_lower_bound", lower)
        matrix.set_float_info("label_upper_bound", upper)
        matrix.set_weight(sample_weight_values(data))
        return matrix

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        started_at = self.start_timer()
        training = self._matrix(training_data, training=True)
        evaluations = [(training, "train")]
        if validation_data is not None:
            evaluations.append((self._matrix(validation_data, training=False), "validation"))
        rounds = self.training_iterations or int(self.hyperparameters["maximum_trees"])
        callback = self.require_training_monitor().create_xgboost_callback(
            self.log_training_step
        )
        evaluation_results: dict[str, dict] = {}
        self.estimator = xgb.train(
            {
                "objective": "survival:aft",
                "eval_metric": "aft-nloglik",
                "aft_loss_distribution": str(
                    self.hyperparameters["aft_loss_distribution"]
                ),
                "aft_loss_distribution_scale": float(
                    self.hyperparameters["aft_loss_distribution_scale"]
                ),
                "eta": float(self.hyperparameters["learning_rate"]),
                "max_depth": int(self.hyperparameters["max_depth"]),
                "min_child_weight": float(self.hyperparameters["min_child_weight"]),
                "subsample": float(self.hyperparameters["subsample"]),
                "colsample_bytree": float(self.hyperparameters["colsample_bytree"]),
                "alpha": float(self.hyperparameters["reg_alpha"]),
                "lambda": float(self.hyperparameters["reg_lambda"]),
                "tree_method": "hist",
                "device": self.device,
                "seed": self.seed,
                "nthread": 1,
            },
            training,
            num_boost_round=rounds,
            evals=evaluations,
            early_stopping_rounds=(
                int(self.early_stopping_patience)
                if validation_data is not None and self.training_iterations is None
                else None
            ),
            evals_result=evaluation_results,
            verbose_eval=False,
            callbacks=[callback],
        )
        callback.log_final_round(evaluation_results)
        best_iteration = getattr(self.estimator, "best_iteration", None)
        completed = int(best_iteration) + 1 if best_iteration is not None else rounds
        self._is_fitted = True
        validation_rmse = None
        if validation_data is not None:
            validation_rmse = root_mean_squared_error(
                target_values(validation_data), self.predict(validation_data)
            )
        self.training_summary = TrainingSummary(
            model_family=self.family,
            seed=self.seed,
            training_rows=len(training_data),
            validation_rows=0 if validation_data is None else len(validation_data),
            training_seconds=self.elapsed_seconds(started_at),
            epochs_or_iterations=completed,
            best_epoch_or_iteration=completed if best_iteration is not None else None,
            best_validation_rmse=validation_rmse,
            trainable_parameters=None,
        )
        return self.training_summary

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        values, names = tabular_values(data, self.feature_names)
        matrix = xgb.DMatrix(values, feature_names=names)
        iteration_range = (0, int(self.estimator.best_iteration) + 1) if hasattr(self.estimator, "best_iteration") else (0, 0)
        return np.asarray(
            self.estimator.predict(matrix, iteration_range=iteration_range),
            dtype=np.float64,
        )


class HorizonXGBoostAdapter(ModelAdapter):
    """Estimate capped RUL by integrating ordered failure-horizon probabilities."""

    family = "horizon_xgboost"
    representation = "tabular"
    stochastic = True

    @property
    def horizons(self) -> NDArray[np.float64]:
        try:
            values = np.asarray(
                [float(value) for value in str(self.hyperparameters["horizons"]).split(",")],
                dtype=np.float64,
            )
        except ValueError as error:
            raise ModelAdapterError("horizons must be a comma-separated numeric list") from error
        if len(values) < 2 or np.any(values <= 0.0) or np.any(np.diff(values) <= 0.0):
            raise ModelAdapterError("horizons must be positive and strictly increasing")
        return values

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        started_at = self.start_timer()
        values, self.feature_names = tabular_values(training_data)
        targets = target_values(training_data)
        weights = sample_weight_values(training_data)
        self.estimators: list[XGBClassifier] = []
        tree_count = int(self.hyperparameters["maximum_trees"])
        for index, horizon in enumerate(self.horizons):
            labels = (targets <= horizon).astype(np.int8)
            if np.unique(labels).size != 2:
                raise ModelAdapterError(
                    f"Horizon {horizon:g} does not contain both outcome classes"
                )
            estimator = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=tree_count,
                learning_rate=float(self.hyperparameters["learning_rate"]),
                max_depth=int(self.hyperparameters["max_depth"]),
                min_child_weight=float(self.hyperparameters["min_child_weight"]),
                subsample=float(self.hyperparameters["subsample"]),
                colsample_bytree=float(self.hyperparameters["colsample_bytree"]),
                reg_alpha=float(self.hyperparameters["reg_alpha"]),
                reg_lambda=float(self.hyperparameters["reg_lambda"]),
                random_state=self.seed + index,
                n_jobs=1,
                tree_method="hist",
                device=resolve_xgboost_device("auto"),
            )
            estimator.fit(values, labels, sample_weight=weights, verbose=False)
            self.estimators.append(estimator)
        self._is_fitted = True
        validation_rmse = None
        if validation_data is not None:
            validation_rmse = root_mean_squared_error(
                target_values(validation_data), self.predict(validation_data)
            )
        self.training_summary = TrainingSummary(
            model_family=self.family,
            seed=self.seed,
            training_rows=len(training_data),
            validation_rows=0 if validation_data is None else len(validation_data),
            training_seconds=self.elapsed_seconds(started_at),
            epochs_or_iterations=len(self.estimators) * tree_count,
            best_epoch_or_iteration=None,
            best_validation_rmse=validation_rmse,
            trainable_parameters=None,
        )
        return self.training_summary

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        values, _ = tabular_values(data, self.feature_names)
        failure_cdf = np.column_stack(
            [estimator.predict_proba(values)[:, 1] for estimator in self.estimators]
        )
        failure_cdf = np.maximum.accumulate(failure_cdf, axis=1)
        survival = 1.0 - failure_cdf
        grid = np.r_[0.0, self.horizons]
        survival_grid = np.column_stack([np.ones(len(values)), survival])
        return np.trapezoid(survival_grid, grid, axis=1).astype(np.float64)
