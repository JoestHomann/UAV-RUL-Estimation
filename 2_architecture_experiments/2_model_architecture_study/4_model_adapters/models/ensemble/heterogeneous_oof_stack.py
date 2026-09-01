"""Deploy a meta-model frozen from OOF tabular and temporal predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from base import ModelAdapter, ModelAdapterError, TrainingSummary
from models.neural.gru import GRUAdapter
from models.neural.lstm import LSTMAdapter
from models.neural.multiscale_cnn import MultiScaleCNNAdapter
from models.neural.neural_base import NeuralTrainingConfig
from models.neural.tcn import TCNAdapter
from models.tabular.calibrated_tree_blend import CalibratedTreeBlendAdapter
from models.tabular.extra_trees import ExtraTreesAdapter
from models.tabular.xgboost import XGBoostAdapter


TEMPORAL_CLASSES = {
    "tcn": TCNAdapter,
    "multiscale_cnn": MultiScaleCNNAdapter,
    "gru": GRUAdapter,
    "lstm": LSTMAdapter,
}
TREE_CLASSES = {
    "extra_trees": ExtraTreesAdapter,
    "xgboost": XGBoostAdapter,
    "calibrated_tree_blend": CalibratedTreeBlendAdapter,
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]


class HeterogeneousOOFStackAdapter(ModelAdapter):
    """Train frozen base configurations and apply an OOF-only meta-model."""

    family = "heterogeneous_oof_stack"
    representation = "heterogeneous"
    stochastic = True

    def _component(self, specification: dict[str, Any], *, temporal: bool) -> ModelAdapter:
        family = str(specification.get("family"))
        hyperparameters = specification.get("hyperparameters")
        if not isinstance(hyperparameters, dict):
            raise ModelAdapterError(f"{family} component has no hyperparameters")
        common = {
            "hyperparameters": hyperparameters,
            "seed": int(specification.get("seed", self.seed)),
            "prediction_minimum": self.prediction_minimum,
            "training_monitor": self._training_monitor,
        }
        if temporal:
            adapter_class = TEMPORAL_CLASSES.get(family)
            training = specification.get("neural_training")
            if adapter_class is None or not isinstance(training, dict):
                raise ModelAdapterError(f"Unsupported temporal component {family!r}")
            adapter = adapter_class(
                **common,
                training_config=NeuralTrainingConfig(**training),
                training_epochs=specification.get("training_iterations"),
            )
        else:
            adapter_class = TREE_CLASSES.get(family)
            if adapter_class is None:
                raise ModelAdapterError(f"Unsupported tree component {family!r}")
            if family == "xgboost":
                adapter = adapter_class(
                    **common,
                    early_stopping_patience=int(specification.get("early_stopping_patience", 25)),
                    training_iterations=specification.get("training_iterations"),
                )
            else:
                adapter = adapter_class(**common)
        adapter.configure_policies(self.target_policy, self.prediction_policy)
        return adapter

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        if not hasattr(training_data, "tabular") or not hasattr(training_data, "sequence"):
            raise ModelAdapterError("Heterogeneous stack requires aligned tabular and sequence data")
        tree_spec = self.hyperparameters.get("tree_component")
        temporal_spec = self.hyperparameters.get("temporal_component")
        if not isinstance(tree_spec, dict) or not isinstance(temporal_spec, dict):
            raise ModelAdapterError("Heterogeneous stack component contracts are missing")
        started = self.start_timer()
        self.tree_adapter = self._component(tree_spec, temporal=False)
        self.temporal_adapter = self._component(temporal_spec, temporal=True)
        validation_tabular = None if validation_data is None else validation_data.tabular
        validation_sequence = None if validation_data is None else validation_data.sequence
        tree_summary = self.tree_adapter.fit(training_data.tabular, validation_tabular)
        temporal_summary = self.temporal_adapter.fit(training_data.sequence, validation_sequence)
        supplied_meta_path = Path(
            str(self.hyperparameters.get("meta_model_path"))
        )
        meta_path = (
            supplied_meta_path.resolve()
            if supplied_meta_path.is_absolute()
            else (REPOSITORY_ROOT / supplied_meta_path).resolve()
        )
        try:
            meta_path.relative_to(REPOSITORY_ROOT)
        except ValueError as error:
            raise ModelAdapterError("Frozen OOF meta-model escapes the repository") from error
        try:
            self.meta_model = joblib.load(meta_path)
        except Exception as error:
            raise ModelAdapterError(f"Cannot load frozen OOF meta-model {meta_path}: {error}") from error
        self._is_fitted = True
        self.training_summary = TrainingSummary(
            model_family=self.family,
            seed=self.seed,
            training_rows=len(training_data),
            validation_rows=0 if validation_data is None else len(validation_data),
            training_seconds=self.elapsed_seconds(started),
            epochs_or_iterations=temporal_summary.epochs_or_iterations,
            best_epoch_or_iteration=temporal_summary.best_epoch_or_iteration,
            best_validation_rmse=temporal_summary.best_validation_rmse,
            trainable_parameters=(tree_summary.trainable_parameters or 0) + (temporal_summary.trainable_parameters or 0),
        )
        return self.training_summary

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        component_predictions = np.column_stack(
            [self.tree_adapter.predict(data.tabular), self.temporal_adapter.predict(data.sequence)]
        )
        if isinstance(self.meta_model, dict) and self.meta_model.get("method") == "convex_blend":
            prediction = self.meta_model["tree_weight"] * component_predictions[:, 0] + self.meta_model["temporal_weight"] * component_predictions[:, 1]
        elif hasattr(self.meta_model, "predict"):
            prediction = self.meta_model.predict(
                pd.DataFrame(
                    component_predictions,
                    columns=["prediction__tree", "prediction__temporal"],
                )
            )
        else:
            raise ModelAdapterError("Frozen OOF meta-model has an unknown format")
        return np.asarray(prediction, dtype=np.float64)
