"""Implement the frozen calibrated XGBoost/ExtraTrees blend policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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
        local_calibration = hyperparameters.get("calibration_features_path")
        self.calibration_features_path = (
            None
            if local_calibration is None
            else _repository_path(local_calibration, "calibration_features_path")
        )
        self.calibration_internal_folds = int(
            hyperparameters.get("calibration_internal_folds", 4)
        )
        self.calibration_degree = int(hyperparameters.get("calibration_degree", 2))
        self.calibration_ridge_alpha = float(
            hyperparameters.get("calibration_ridge_alpha", 10.0)
        )
        if self.calibration_features_path is not None:
            if self.calibration_internal_folds < 2:
                raise ModelAdapterError("Local calibration requires at least two folds")
            if self.calibration_degree < 1:
                raise ModelAdapterError("Local calibration degree must be positive")
            if self.calibration_ridge_alpha <= 0.0:
                raise ModelAdapterError("Local calibration ridge alpha must be positive")
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

    def _new_components(self) -> tuple[ExtraTreesAdapter, XGBoostAdapter]:
        extra_configuration = self._component_configuration(
            "extra_trees",
            "extra_trees_configuration_index",
        )
        xgboost_configuration = self._component_configuration(
            "xgboost",
            "xgboost_configuration_index",
        )
        extra_trees = ExtraTreesAdapter(
            hyperparameters=extra_configuration["hyperparameters"],
            seed=self.seed,
            prediction_minimum=self.prediction_minimum,
        )
        xgboost = XGBoostAdapter(
            hyperparameters=xgboost_configuration["hyperparameters"],
            seed=self.seed,
            prediction_minimum=self.prediction_minimum,
            early_stopping_patience=None,
            training_iterations=int(xgboost_configuration["training_iterations"]),
            training_monitor=self._training_monitor,
        )
        for component in (extra_trees, xgboost):
            component.configure_policies(self.target_policy, self.prediction_policy)
        return extra_trees, xgboost

    @staticmethod
    def _subset(data: Any, mask: NDArray[np.bool_]) -> Any:
        selected = pd.Series(mask, index=data.features.index)

        def take(value: pd.Series | None) -> pd.Series | None:
            return None if value is None else value.loc[selected].reset_index(drop=True)

        return type(data)(
            features=data.features.loc[selected].reset_index(drop=True),
            metadata=data.metadata.loc[selected].reset_index(drop=True),
            target=take(data.target),
            sample_weights=take(data.sample_weights),
            fitting_target=take(data.fitting_target),
        )

    def _calibration_data(self, training_data: Any) -> Any:
        if self.calibration_features_path is None:
            raise ModelAdapterError("No fold-local calibration source is configured")
        feature_names = [str(column) for column in training_data.features.columns]
        columns = ["sample_id", "scenario", "uav_id", "cutoff", "RUL", *feature_names]
        try:
            table = pd.read_csv(self.calibration_features_path, usecols=columns)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise ModelAdapterError(f"Cannot load local calibration endpoints: {error}") from error
        training_uavs = set(training_data.metadata["uav_id"].astype(str))
        table = table.loc[table["uav_id"].astype(str).isin(training_uavs)].copy()
        if table.empty or table["uav_id"].astype(str).nunique() != len(training_uavs):
            raise ModelAdapterError(
                "Local calibration endpoints do not cover every training UAV"
            )
        if table["sample_id"].astype(str).duplicated().any():
            raise ModelAdapterError("Local calibration sample IDs are duplicated")
        metadata = table[["sample_id", "scenario", "uav_id", "cutoff"]].copy()
        metadata["uav_id"] = metadata["uav_id"].astype(str)
        return type(training_data)(
            features=table[feature_names].reset_index(drop=True),
            metadata=metadata.reset_index(drop=True),
            target=table["RUL"].astype(float).reset_index(drop=True),
            sample_weights=None,
            fitting_target=None,
        )

    def _raw_component_blend(
        self,
        extra_trees: ExtraTreesAdapter,
        xgboost: XGBoostAdapter,
        data: Any,
    ) -> NDArray[np.float64]:
        return np.asarray(
            self.xgboost_weight * xgboost.predict(data)
            + (1.0 - self.xgboost_weight) * extra_trees.predict(data),
            dtype=np.float64,
        )

    def _fit_local_calibrator(self, training_data: Any) -> Any:
        calibration = self._calibration_data(training_data)
        groups = calibration.metadata["uav_id"].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        if len(unique_groups) < self.calibration_internal_folds:
            raise ModelAdapterError("Too few UAVs for fold-local calibration")
        oof_prediction = np.full(len(calibration), np.nan, dtype=np.float64)
        splitter = GroupKFold(n_splits=self.calibration_internal_folds)
        for _, held_index in splitter.split(calibration.features, groups=groups):
            held_uavs = set(groups[held_index])
            training_mask = ~training_data.metadata["uav_id"].astype(str).isin(
                held_uavs
            ).to_numpy()
            held_mask = np.zeros(len(calibration), dtype=bool)
            held_mask[held_index] = True
            fold_training = self._subset(training_data, training_mask)
            fold_calibration = self._subset(calibration, held_mask)
            extra_trees, xgboost = self._new_components()
            extra_trees.fit(fold_training, None)
            xgboost.fit(fold_training, None)
            oof_prediction[held_index] = self._raw_component_blend(
                extra_trees,
                xgboost,
                fold_calibration,
            )
        if not np.isfinite(oof_prediction).all():
            raise ModelAdapterError("Local calibration OOF predictions are incomplete")
        calibration_features = pd.DataFrame(
            {
                "raw_blend": oof_prediction,
                "cutoff": calibration.metadata["cutoff"].to_numpy(float),
            }
        )
        observed = target_values(calibration)
        residual = oof_prediction - observed
        counts = pd.Series(groups).value_counts()
        weights = np.asarray([1.0 / float(counts[value]) for value in groups])
        weights *= len(weights) / weights.sum()
        calibrator = make_pipeline(
            PolynomialFeatures(degree=self.calibration_degree, include_bias=False),
            StandardScaler(),
            Ridge(alpha=self.calibration_ridge_alpha),
        )
        calibrator.fit(calibration_features, residual, ridge__sample_weight=weights)
        return calibrator

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        started_at = self.start_timer()
        if self.calibration_features_path is None:
            try:
                self.calibrator = joblib.load(self.calibrator_path)
            except Exception as error:
                raise ModelAdapterError(
                    f"Cannot load frozen residual calibrator {self.calibrator_path}: {error}"
                ) from error
        else:
            self.calibrator = self._fit_local_calibrator(training_data)
        self.extra_trees, self.xgboost = self._new_components()
        extra_summary = self.extra_trees.fit(training_data, None)
        xgboost_summary = self.xgboost.fit(training_data, None)
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
        raw_blend = self._raw_component_blend(self.extra_trees, self.xgboost, data)
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
