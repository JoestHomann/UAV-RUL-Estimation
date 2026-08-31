"""Implement the frozen calibrated XGBoost/ExtraTrees blend policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from base import (
    ModelAdapter,
    ModelAdapterError,
    TrainingSummary,
    cutoff_values,
    root_mean_squared_error,
    target_values,
)
from models.tabular.extra_trees import ExtraTreesAdapter
from models.tabular.xgboost import XGBoostAdapter


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]


def _repository_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ModelAdapterError(f"{name} must be a repository-relative path")
    supplied = Path(value)
    path = (REPOSITORY_ROOT / supplied).resolve()
    if not path.exists() and supplied.parts:
        moved_prefixes = {
            "pipeline_experiments": Path(
                "2_architecture_experiments/1_pipeline_experiments"
            ),
            "2_model_architecture_study": Path(
                "2_architecture_experiments/2_model_architecture_study"
            ),
        }
        replacement = moved_prefixes.get(supplied.parts[0])
        if replacement is not None:
            path = (
                REPOSITORY_ROOT / replacement.joinpath(*supplied.parts[1:])
            ).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ModelAdapterError(f"{name} escapes the repository") from error
    return path


class CalibratedTreeBlendAdapter(ModelAdapter):
    """Fit two frozen-policy tree components and calibrate their fixed blend."""

    family = "calibrated_tree_blend"
    representation = "tabular"
    stochastic = True

    def __init__(
        self,
        *,
        hyperparameters: dict[str, Any],
        seed: int,
        prediction_minimum: float = 0.0,
        training_monitor: Any | None = None,
    ) -> None:
        super().__init__(
            hyperparameters=hyperparameters,
            seed=seed,
            prediction_minimum=prediction_minimum,
            training_monitor=training_monitor,
        )
        configuration_path = _repository_path(
            hyperparameters["component_configurations_path"],
            "component_configurations_path",
        )
        try:
            payload = json.loads(configuration_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelAdapterError(
                f"Cannot read component configurations {configuration_path}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise ModelAdapterError("Component configurations must contain an object")
        self.component_configurations = payload
        self.calibrator_path = _repository_path(
            hyperparameters["residual_calibrator_path"],
            "residual_calibrator_path",
        )
        self.xgboost_weight = float(hyperparameters["xgboost_weight"])
        if not 0.0 < self.xgboost_weight < 1.0:
            raise ModelAdapterError("xgboost_weight must be in (0, 1)")

    def _component_configuration(self, family: str, index_name: str) -> dict[str, Any]:
        values = self.component_configurations.get(family)
        index = int(self.hyperparameters[index_name])
        if not isinstance(values, list) or not 0 <= index < len(values):
            raise ModelAdapterError(f"Invalid {family} component configuration index")
        value = values[index]
        if not isinstance(value, dict) or not isinstance(value.get("hyperparameters"), dict):
            raise ModelAdapterError(f"Malformed {family} component configuration")
        return value

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        started_at = self.start_timer()
        extra_configuration = self._component_configuration(
            "extra_trees",
            "extra_trees_configuration_index",
        )
        xgboost_configuration = self._component_configuration(
            "xgboost",
            "xgboost_configuration_index",
        )
        self.extra_trees = ExtraTreesAdapter(
            hyperparameters=extra_configuration["hyperparameters"],
            seed=self.seed,
            prediction_minimum=self.prediction_minimum,
        )
        self.xgboost = XGBoostAdapter(
            hyperparameters=xgboost_configuration["hyperparameters"],
            seed=self.seed,
            prediction_minimum=self.prediction_minimum,
            early_stopping_patience=None,
            training_iterations=int(xgboost_configuration["training_iterations"]),
            training_monitor=self._training_monitor,
        )
        for component in (self.extra_trees, self.xgboost):
            component.configure_policies(self.target_policy, self.prediction_policy)
        extra_summary = self.extra_trees.fit(training_data, None)
        xgboost_summary = self.xgboost.fit(training_data, None)
        try:
            self.calibrator = joblib.load(self.calibrator_path)
        except Exception as error:
            raise ModelAdapterError(
                f"Cannot load frozen residual calibrator {self.calibrator_path}: {error}"
            ) from error
        self._is_fitted = True
        validation_rmse = None
        if validation_data is not None:
            validation_rmse = root_mean_squared_error(
                target_values(validation_data),
                self.predict(validation_data),
            )
        summary = TrainingSummary(
            model_family=self.family,
            seed=self.seed,
            training_rows=len(training_data),
            validation_rows=0 if validation_data is None else len(validation_data),
            training_seconds=self.elapsed_seconds(started_at),
            epochs_or_iterations=xgboost_summary.epochs_or_iterations,
            best_epoch_or_iteration=xgboost_summary.best_epoch_or_iteration,
            best_validation_rmse=validation_rmse,
            trainable_parameters=None,
        )
        self.training_summary = summary
        return summary

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        extra_prediction = self.extra_trees.predict(data)
        xgboost_prediction = self.xgboost.predict(data)
        raw_blend = (
            self.xgboost_weight * xgboost_prediction
            + (1.0 - self.xgboost_weight) * extra_prediction
        )
        cutoffs = cutoff_values(data)
        if cutoffs is None:
            raise ModelAdapterError("Calibrated blend requires endpoint cutoffs")
        features = pd.DataFrame(
            {"raw_blend": raw_blend, "cutoff": cutoffs}
        )
        correction = np.asarray(self.calibrator.predict(features), dtype=np.float64)
        return raw_blend - correction

    def predict(self, data: Any) -> NDArray[np.float64]:
        """Return calibrated raw-RUL predictions without a second target inverse."""

        if not self._is_fitted:
            raise ModelAdapterError("Calibrated tree blend is not fitted")
        predictions = np.asarray(self._predict_raw(data), dtype=np.float64).reshape(-1)
        if len(predictions) != len(data) or not np.isfinite(predictions).all():
            raise ModelAdapterError("Calibrated tree blend produced invalid predictions")
        return np.maximum(predictions, self.prediction_minimum)

    def detach_training_monitor(self) -> None:
        if hasattr(self, "xgboost"):
            self.xgboost.detach_training_monitor()
        super().detach_training_monitor()
