"""Fit the PE_11 seeded tree ensemble with leakage-safe residual correction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from base import (
    ModelAdapter,
    ModelAdapterError,
    TrainingSummary,
    root_mean_squared_error,
    target_values,
)
from models.tabular.extra_trees import ExtraTreesAdapter
from models.tabular.xgboost import XGBoostAdapter
from no_op_training_monitor import NoOpTrainingMonitor
from policies import PredictionPolicy


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]


def _repository_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ModelAdapterError(f"{name} must be a repository-relative path")
    supplied = Path(value)
    path = (REPOSITORY_ROOT / supplied).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ModelAdapterError(f"{name} escapes the repository") from error
    if not path.is_file():
        raise ModelAdapterError(f"{name} does not exist: {path}")
    return path


def _subset(dataset: Any, mask: NDArray[np.bool_]) -> Any:
    selected = pd.Series(mask, index=dataset.features.index)

    def take(value: pd.Series | None) -> pd.Series | None:
        return None if value is None else value.loc[selected].reset_index(drop=True)

    return type(dataset)(
        features=dataset.features.loc[selected].reset_index(drop=True),
        metadata=dataset.metadata.loc[selected].reset_index(drop=True),
        target=take(dataset.target),
        sample_weights=take(dataset.sample_weights),
        fitting_target=take(dataset.fitting_target),
    )


class ResidualCorrectedTreeEnsembleAdapter(ModelAdapter):
    """Combine six seeded tree members and an internally cross-fitted error model."""

    family = "residual_corrected_tree_ensemble"
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
        contract_path = _repository_path(
            hyperparameters["ensemble_contract_path"],
            "ensemble_contract_path",
        )
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelAdapterError(
                f"Cannot read residual ensemble contract {contract_path}: {error}"
            ) from error
        if not isinstance(contract, dict) or contract.get("contract_version") != 1:
            raise ModelAdapterError("Residual ensemble contract has an unknown format")
        self.contract = contract
        self.calibration_path = _repository_path(
            contract.get("calibration_features_path"),
            "calibration_features_path",
        )
        self.member_seeds = tuple(int(value) for value in contract["member_seeds"])
        if len(self.member_seeds) < 2 or len(set(self.member_seeds)) != len(
            self.member_seeds
        ):
            raise ModelAdapterError("Residual ensemble requires distinct member seeds")
        self.internal_folds = int(contract["internal_folds"])
        if self.internal_folds < 2:
            raise ModelAdapterError("Residual ensemble requires at least two OOF folds")
        self.residual_features = tuple(str(value) for value in contract["residual_features"])
        self.weight_grid = tuple(float(value) for value in contract["xgboost_weight_grid"])
        if not self.weight_grid or any(not 0.0 <= value <= 1.0 for value in self.weight_grid):
            raise ModelAdapterError("Residual ensemble blend weights must be in [0, 1]")

    def _calibration_data(self, training_data: Any) -> Any:
        feature_names = [str(column) for column in training_data.features.columns]
        missing = sorted(set(self.residual_features) - set(feature_names))
        if missing:
            raise ModelAdapterError(
                f"Residual correction features are missing from the model input: {missing}"
            )
        columns = [
            "sample_id",
            "scenario",
            "uav_id",
            "cutoff",
            "RUL",
            *feature_names,
        ]
        try:
            table = pd.read_csv(self.calibration_path, usecols=columns)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise ModelAdapterError(
                f"Cannot load residual calibration endpoints: {error}"
            ) from error
        training_uavs = set(training_data.metadata["uav_id"].astype(str))
        table = table.loc[table["uav_id"].astype(str).isin(training_uavs)].copy()
        if table.empty or table["uav_id"].astype(str).nunique() != len(training_uavs):
            raise ModelAdapterError(
                "Residual calibration endpoints do not cover every training UAV"
            )
        if table["sample_id"].astype(str).duplicated().any():
            raise ModelAdapterError("Residual calibration sample IDs are duplicated")
        metadata = table[["sample_id", "scenario", "uav_id", "cutoff"]].copy()
        metadata["uav_id"] = metadata["uav_id"].astype(str)
        return type(training_data)(
            features=table[feature_names].reset_index(drop=True),
            metadata=metadata.reset_index(drop=True),
            target=table["RUL"].astype(float).reset_index(drop=True),
            sample_weights=None,
            fitting_target=None,
        )

    def _component(self, family: str, seed: int) -> ModelAdapter:
        component = self.contract["components"][family]
        hyperparameters = component["hyperparameters"]
        if family == "extra_trees":
            model: ModelAdapter = ExtraTreesAdapter(
                hyperparameters=hyperparameters,
                seed=seed,
                prediction_minimum=self.prediction_minimum,
            )
        else:
            model = XGBoostAdapter(
                hyperparameters=hyperparameters,
                seed=seed,
                prediction_minimum=self.prediction_minimum,
                early_stopping_patience=None,
                training_iterations=int(component["training_iterations"]),
                training_monitor=NoOpTrainingMonitor(),
            )
        model.configure_policies(self.target_policy, PredictionPolicy())
        return model

    def _fit_members(
        self,
        training_data: Any,
        prediction_data: Any,
        *,
        retain: bool,
    ) -> tuple[NDArray[np.float64], list[ModelAdapter]]:
        predictions: list[NDArray[np.float64]] = []
        models: list[ModelAdapter] = []
        for family in ("xgboost", "extra_trees"):
            for member_seed in self.member_seeds:
                model = self._component(family, member_seed)
                model.fit(training_data, None)
                predictions.append(model.predict(prediction_data))
                model.detach_training_monitor()
                if retain:
                    models.append(model)
        return np.column_stack(predictions), models

    def _member_statistics(
        self,
        predictions: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        member_count = len(self.member_seeds)
        xgboost_mean = predictions[:, :member_count].mean(axis=1)
        extra_trees_mean = predictions[:, member_count:].mean(axis=1)
        uncertainty_std = predictions.std(axis=1, ddof=1)
        uncertainty_range = predictions.max(axis=1) - predictions.min(axis=1)
        family_disagreement = np.abs(xgboost_mean - extra_trees_mean)
        return (
            xgboost_mean,
            extra_trees_mean,
            uncertainty_std,
            uncertainty_range,
            family_disagreement,
        )

    def _residual_matrix(
        self,
        data: Any,
        base_prediction: NDArray[np.float64],
        uncertainty_std: NDArray[np.float64],
        uncertainty_range: NDArray[np.float64],
        family_disagreement: NDArray[np.float64],
    ) -> pd.DataFrame:
        matrix = pd.DataFrame(
            {
                "nonnegative_blend": base_prediction,
                "cutoff": data.metadata["cutoff"].to_numpy(dtype=float),
                "uncertainty_std": uncertainty_std,
                "uncertainty_range": uncertainty_range,
                "family_disagreement": family_disagreement,
            }
        )
        for name in self.residual_features:
            matrix[name] = data.features[name].to_numpy(dtype=float)
        return matrix

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        if not isinstance(getattr(training_data, "features", None), pd.DataFrame):
            raise ModelAdapterError("Residual ensemble requires tabular training data")
        started_at = self.start_timer()
        calibration = self._calibration_data(training_data)
        groups = calibration.metadata["uav_id"].astype(str).to_numpy()
        if len(np.unique(groups)) < self.internal_folds:
            raise ModelAdapterError("Too few training UAVs for residual OOF fitting")

        oof_members = np.empty(
            (len(calibration), 2 * len(self.member_seeds)),
            dtype=np.float64,
        )
        splitter = GroupKFold(n_splits=self.internal_folds)
        internal_splits = list(splitter.split(calibration.features, groups=groups))
        for fold_number, (_, held_index) in enumerate(internal_splits, start=1):
            print(
                "Residual ensemble OOF fold "
                f"{fold_number}/{len(internal_splits)}: fitting "
                f"{2 * len(self.member_seeds)} members",
                flush=True,
            )
            held_uavs = set(groups[held_index])
            training_mask = ~training_data.metadata["uav_id"].astype(str).isin(
                held_uavs
            ).to_numpy()
            fold_training = _subset(training_data, training_mask)
            held_mask = np.zeros(len(calibration), dtype=bool)
            held_mask[held_index] = True
            fold_calibration = _subset(calibration, held_mask)
            fold_predictions, _ = self._fit_members(
                fold_training,
                fold_calibration,
                retain=False,
            )
            oof_members[held_index] = fold_predictions

        xgb, extra, uncertainty_std, uncertainty_range, disagreement = (
            self._member_statistics(oof_members)
        )
        observed = target_values(calibration)
        scored_weights = []
        for weight in self.weight_grid:
            estimate = weight * xgb + (1.0 - weight) * extra
            scored_weights.append((root_mean_squared_error(observed, estimate), weight))
        _, self.xgboost_weight = min(scored_weights, key=lambda item: (item[0], item[1]))
        base = np.maximum(
            self.xgboost_weight * xgb + (1.0 - self.xgboost_weight) * extra,
            self.prediction_minimum,
        )
        residual_settings = self.contract["residual_model"]
        self.residual_model = HistGradientBoostingRegressor(
            max_iter=int(residual_settings["maximum_iterations"]),
            max_leaf_nodes=int(residual_settings["maximum_leaf_nodes"]),
            min_samples_leaf=int(residual_settings["minimum_samples_leaf"]),
            l2_regularization=float(residual_settings["l2_regularization"]),
            learning_rate=float(residual_settings["learning_rate"]),
            random_state=int(residual_settings["seed"]),
        )
        residual_matrix = self._residual_matrix(
            calibration,
            base,
            uncertainty_std,
            uncertainty_range,
            disagreement,
        )
        self.residual_model.fit(residual_matrix, base - observed)

        print(
            "Residual ensemble final refit: fitting "
            f"{2 * len(self.member_seeds)} members",
            flush=True,
        )
        _, self.members = self._fit_members(
            training_data,
            calibration,
            retain=True,
        )
        self._is_fitted = True
        validation_rmse = None
        if validation_data is not None:
            validation_rmse = root_mean_squared_error(
                target_values(validation_data),
                self.predict(validation_data),
            )
        self.training_summary = TrainingSummary(
            model_family=self.family,
            seed=self.seed,
            training_rows=len(training_data),
            validation_rows=0 if validation_data is None else len(validation_data),
            training_seconds=self.elapsed_seconds(started_at),
            epochs_or_iterations=None,
            best_epoch_or_iteration=None,
            best_validation_rmse=validation_rmse,
            trainable_parameters=None,
        )
        return self.training_summary

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        member_predictions = np.column_stack(
            [model.predict(data) for model in self.members]
        )
        xgb, extra, uncertainty_std, uncertainty_range, disagreement = (
            self._member_statistics(member_predictions)
        )
        base = np.maximum(
            self.xgboost_weight * xgb + (1.0 - self.xgboost_weight) * extra,
            self.prediction_minimum,
        )
        matrix = self._residual_matrix(
            data,
            base,
            uncertainty_std,
            uncertainty_range,
            disagreement,
        )
        correction = np.asarray(self.residual_model.predict(matrix), dtype=np.float64)
        return base - correction

    def predict(self, data: Any) -> NDArray[np.float64]:
        """Return residual-corrected RUL without applying target inversion twice."""

        if not self._is_fitted:
            raise ModelAdapterError("Residual-corrected tree ensemble is not fitted")
        predictions = np.asarray(self._predict_raw(data), dtype=np.float64).reshape(-1)
        if len(predictions) != len(data) or not np.isfinite(predictions).all():
            raise ModelAdapterError("Residual-corrected ensemble produced invalid predictions")
        predictions = self.prediction_policy.adjust_predictions(predictions)
        return np.maximum(predictions, self.prediction_minimum)

    def detach_training_monitor(self) -> None:
        for model in getattr(self, "members", []):
            model.detach_training_monitor()
        super().detach_training_monitor()
