"""Fit the PE_11 seeded tree ensemble with leakage-safe residual correction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

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


class ScaledRidgeResidualRegressor:
    """Small serializable ridge head for residual-correction ablations."""

    def __init__(self, *, alpha: float) -> None:
        if alpha <= 0.0:
            raise ModelAdapterError("Residual ridge alpha must be positive")
        self.alpha = float(alpha)

    def fit(
        self,
        features: pd.DataFrame,
        target: NDArray[np.float64],
        *,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> ScaledRidgeResidualRegressor:
        self.feature_names = tuple(str(value) for value in features.columns)
        values = features.to_numpy(dtype=np.float64)
        self.scaler = StandardScaler()
        transformed = self.scaler.fit_transform(values)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(transformed, target, sample_weight=sample_weight)
        return self

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        if tuple(str(value) for value in features.columns) != self.feature_names:
            raise ModelAdapterError("Residual ridge feature order changed")
        values = self.scaler.transform(features.to_numpy(dtype=np.float64))
        return np.asarray(self.model.predict(values), dtype=np.float64)


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


def calibration_sample_weights(
    metadata: pd.DataFrame,
    strategy: str,
) -> NDArray[np.float64]:
    """Give every UAV, and optionally every distinct endpoint, equal total weight."""

    uavs = metadata["uav_id"].astype(str)
    if strategy == "equal_uav_rows":
        uav_counts = uavs.value_counts()
        weights = np.asarray([1.0 / float(uav_counts[uav]) for uav in uavs])
    elif strategy == "equal_uav_unique_endpoints":
        cutoffs = metadata["cutoff"].astype(float)
        endpoints = pd.DataFrame(
            {"uav_id": uavs.to_numpy(), "cutoff": cutoffs.to_numpy()}
        )
        endpoint_counts = endpoints.groupby(["uav_id", "cutoff"]).size()
        unique_per_uav = endpoints.drop_duplicates().groupby("uav_id").size()
        weights = np.asarray(
            [
                1.0 / (float(unique_per_uav[uav]) * float(endpoint_counts[(uav, cutoff)]))
                for uav, cutoff in endpoints.itertuples(index=False, name=None)
            ],
            dtype=np.float64,
        )
    else:
        raise ModelAdapterError(f"Unknown calibration weighting {strategy!r}")
    return weights * len(weights) / weights.sum()


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
        self.correction_strength = float(contract.get("correction_strength", 1.0))
        if not 0.0 <= self.correction_strength <= 1.5:
            raise ModelAdapterError("Residual correction strength must be in [0, 1.5]")
        self.calibration_weighting = str(
            contract.get("calibration_weighting", "equal_uav_rows")
        )
        if self.calibration_weighting not in {
            "equal_uav_rows",
            "equal_uav_unique_endpoints",
        }:
            raise ModelAdapterError(
                f"Unknown calibration weighting {self.calibration_weighting!r}"
            )
        self.weight_grid = tuple(float(value) for value in contract["xgboost_weight_grid"])
        if not self.weight_grid or any(not 0.0 <= value <= 1.0 for value in self.weight_grid):
            raise ModelAdapterError("Residual ensemble blend weights must be in [0, 1]")
        self.adaptive_uav_weighting = self._validate_adaptive_weighting(
            contract.get("adaptive_uav_weighting")
        )

    def _validate_adaptive_weighting(
        self,
        value: Any,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ModelAdapterError("Adaptive UAV weighting contract must be an object")
        required = {
            "method",
            "multiplier",
            "hard_fraction",
            "difficulty_folds",
            "difficulty_seed",
            "base_ensemble_contract_path",
            "pilot_promoted",
            "pilot_eligible",
            "deployment_purpose",
        }
        if set(value) != required:
            raise ModelAdapterError(
                "Adaptive UAV weighting contract fields differ from the adapter contract"
            )
        method = str(value["method"])
        multiplier = float(value["multiplier"])
        expected_multiplier = {
            "adaptive_uav_1_5": 1.5,
            "adaptive_uav_2_0": 2.0,
        }.get(method)
        if expected_multiplier is None or multiplier != expected_multiplier:
            raise ModelAdapterError("Adaptive UAV method and multiplier disagree")
        fraction = float(value["hard_fraction"])
        folds = int(value["difficulty_folds"])
        if not 0.0 < fraction < 1.0 or folds < 2:
            raise ModelAdapterError("Adaptive UAV fraction or fold count is invalid")
        if value["pilot_promoted"] is not False or value["pilot_eligible"] is not False:
            raise ModelAdapterError("PE_34 deployment must retain its unpromoted status")
        if value["deployment_purpose"] != "leaderboard_probe":
            raise ModelAdapterError("PE_34 deployment purpose is not explicit")
        base_path = _repository_path(
            value["base_ensemble_contract_path"],
            "base_ensemble_contract_path",
        )
        try:
            base_contract = json.loads(base_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelAdapterError(
                f"Cannot read base residual ensemble contract {base_path}: {error}"
            ) from error
        if base_contract.get("adaptive_uav_weighting") is not None:
            raise ModelAdapterError("Adaptive difficulty model cannot itself be adaptive")
        return {
            **value,
            "multiplier": multiplier,
            "hard_fraction": fraction,
            "difficulty_folds": folds,
            "difficulty_seed": int(value["difficulty_seed"]),
            "base_ensemble_contract_path": str(value["base_ensemble_contract_path"]),
        }

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

    def _difficulty_model(self) -> ResidualCorrectedTreeEnsembleAdapter:
        assert self.adaptive_uav_weighting is not None
        model = ResidualCorrectedTreeEnsembleAdapter(
            hyperparameters={
                "ensemble_contract_path": self.adaptive_uav_weighting[
                    "base_ensemble_contract_path"
                ]
            },
            seed=self.seed,
            prediction_minimum=self.prediction_minimum,
            training_monitor=NoOpTrainingMonitor(),
        )
        model.configure_policies(self.target_policy, PredictionPolicy())
        return model

    def _adaptive_training_data(self, training_data: Any) -> Any:
        settings = self.adaptive_uav_weighting
        if settings is None:
            return training_data
        if training_data.sample_weights is None:
            raise ModelAdapterError("Adaptive UAV weighting requires sample weights")
        calibration = self._calibration_data(training_data)
        uavs = np.asarray(
            sorted(training_data.metadata["uav_id"].astype(str).unique()),
            dtype=object,
        )
        folds = int(settings["difficulty_folds"])
        if len(uavs) < folds:
            raise ModelAdapterError("Too few UAVs for adaptive difficulty fitting")
        oof_prediction = np.empty(len(calibration), dtype=np.float64)
        seen = np.zeros(len(calibration), dtype=bool)
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=int(settings["difficulty_seed"]),
        )
        for fold_number, (fit_index, held_index) in enumerate(
            splitter.split(uavs),
            start=1,
        ):
            fitting_uavs = set(uavs[fit_index])
            held_uavs = set(uavs[held_index])
            training_mask = training_data.metadata["uav_id"].astype(str).isin(
                fitting_uavs
            ).to_numpy()
            calibration_mask = calibration.metadata["uav_id"].astype(str).isin(
                held_uavs
            ).to_numpy()
            print(
                "Adaptive UAV difficulty fold "
                f"{fold_number}/{folds}: fitting unchanged Run 7",
                flush=True,
            )
            model = self._difficulty_model()
            model.fit(_subset(training_data, training_mask), None)
            oof_prediction[calibration_mask] = model.predict(
                _subset(calibration, calibration_mask)
            )
            seen[calibration_mask] = True
        if not seen.all() or not np.isfinite(oof_prediction).all():
            raise ModelAdapterError("Adaptive difficulty predictions are incomplete")

        scored = calibration.metadata[["uav_id"]].copy()
        scored["squared_error"] = np.square(
            oof_prediction - target_values(calibration)
        )
        ranking = (
            scored.groupby("uav_id", as_index=False)["squared_error"]
            .mean()
            .assign(rmse=lambda table: np.sqrt(table["squared_error"]))
            .sort_values(["rmse", "uav_id"], ascending=[False, True])
        )
        hard_count = max(
            1,
            int(np.ceil(len(ranking) * float(settings["hard_fraction"]))),
        )
        hard_uavs = set(ranking.iloc[:hard_count]["uav_id"].astype(str))
        base_weights = training_data.sample_weights.to_numpy(dtype=np.float64)
        if not np.isfinite(base_weights).all() or (base_weights <= 0.0).any():
            raise ModelAdapterError("Adaptive base weights must be finite and positive")
        relative = np.where(
            training_data.metadata["uav_id"].astype(str).isin(hard_uavs),
            float(settings["multiplier"]),
            1.0,
        )
        weights = base_weights * relative
        weights *= base_weights.sum() / weights.sum()
        if not np.isclose(weights.sum(), base_weights.sum(), rtol=1e-12, atol=1e-12):
            raise ModelAdapterError("Adaptive weighting changed total weight mass")
        self.last_adaptive_weighting = {
            "training_uavs": len(uavs),
            "hard_uavs": sorted(hard_uavs),
            "multiplier": float(settings["multiplier"]),
            "original_weight_sum": float(base_weights.sum()),
            "adaptive_weight_sum": float(weights.sum()),
        }
        return type(training_data)(
            features=training_data.features,
            metadata=training_data.metadata,
            target=training_data.target,
            sample_weights=pd.Series(weights, index=training_data.sample_weights.index),
            fitting_target=training_data.fitting_target,
        )

    def _fit_members(
        self,
        training_data: Any,
        prediction_data: Any,
        *,
        retain: bool,
    ) -> tuple[NDArray[np.float64], list[ModelAdapter]]:
        training_data = self._adaptive_training_data(training_data)
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
        calibration_weights = calibration_sample_weights(
            calibration.metadata,
            self.calibration_weighting,
        )
        scored_weights = []
        for weight in self.weight_grid:
            estimate = weight * xgb + (1.0 - weight) * extra
            weighted_rmse = float(
                np.sqrt(np.average(np.square(estimate - observed), weights=calibration_weights))
            )
            scored_weights.append((weighted_rmse, weight))
        _, self.xgboost_weight = min(scored_weights, key=lambda item: (item[0], item[1]))
        base = np.maximum(
            self.xgboost_weight * xgb + (1.0 - self.xgboost_weight) * extra,
            self.prediction_minimum,
        )
        residual_settings = self.contract["residual_model"]
        residual_family = str(
            residual_settings.get("family", "hist_gradient_boosting")
        )
        if residual_family == "hist_gradient_boosting":
            self.residual_model: Any = HistGradientBoostingRegressor(
                max_iter=int(residual_settings["maximum_iterations"]),
                max_leaf_nodes=int(residual_settings["maximum_leaf_nodes"]),
                min_samples_leaf=int(residual_settings["minimum_samples_leaf"]),
                l2_regularization=float(residual_settings["l2_regularization"]),
                learning_rate=float(residual_settings["learning_rate"]),
                random_state=int(residual_settings["seed"]),
            )
        elif residual_family == "ridge":
            self.residual_model = ScaledRidgeResidualRegressor(
                alpha=float(residual_settings["alpha"])
            )
        else:
            raise ModelAdapterError(
                f"Unknown residual model family {residual_family!r}"
            )
        residual_matrix = self._residual_matrix(
            calibration,
            base,
            uncertainty_std,
            uncertainty_range,
            disagreement,
        )
        self.residual_model.fit(
            residual_matrix,
            base - observed,
            sample_weight=calibration_weights,
        )

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

    def _predict_raw_with_diagnostics(
        self,
        data: Any,
    ) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
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
        raw_prediction = base - self.correction_strength * correction
        return raw_prediction, {
            "base_prediction": np.asarray(base, dtype=np.float64),
            "uncertainty_std": np.asarray(uncertainty_std, dtype=np.float64),
            "uncertainty_range": np.asarray(uncertainty_range, dtype=np.float64),
            "family_disagreement": np.asarray(disagreement, dtype=np.float64),
        }

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        prediction, _ = self._predict_raw_with_diagnostics(data)
        return prediction

    def predict_with_diagnostics(self, data: Any) -> pd.DataFrame:
        """Return the final prediction and inference-time ensemble diagnostics."""

        if not self._is_fitted:
            raise ModelAdapterError("Residual-corrected tree ensemble is not fitted")
        raw_prediction, diagnostics = self._predict_raw_with_diagnostics(data)
        prediction = np.asarray(
            self.prediction_policy.adjust_predictions(raw_prediction),
            dtype=np.float64,
        ).reshape(-1)
        prediction = np.maximum(prediction, self.prediction_minimum)
        result = pd.DataFrame(
            {
                "predicted_rul": prediction,
                **diagnostics,
            }
        )
        if len(result) != len(data) or not np.isfinite(result.to_numpy(float)).all():
            raise ModelAdapterError(
                "Residual-corrected ensemble produced invalid diagnostics"
            )
        return result

    def predict(self, data: Any) -> NDArray[np.float64]:
        """Return residual-corrected RUL without applying target inversion twice."""

        return self.predict_with_diagnostics(data)["predicted_rul"].to_numpy(float)

    def detach_training_monitor(self) -> None:
        for model in getattr(self, "members", []):
            model.detach_training_monitor()
        super().detach_training_monitor()
