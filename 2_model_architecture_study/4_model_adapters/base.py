"""Define the common behavior shared by every Phase 2 model adapter.

The architecture study exchanges representation and estimator modules while
keeping fitting, prediction, clipping, and persistence calls uniform. This file
contains that stable boundary and data-independent validation helpers. Concrete
model logic belongs in one categorized module per architecture below "models".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import joblib
import numpy as np
from numpy.typing import NDArray
import pandas as pd


Representation = Literal["none", "tabular", "sequence"]


class ModelAdapterError(ValueError):
    """Represent a readable adapter configuration or data failure."""


@dataclass(frozen=True)
class TrainingSummary:
    """Record comparable facts produced by one adapter fitting call."""

    model_family: str
    seed: int
    training_rows: int
    validation_rows: int
    training_seconds: float
    epochs_or_iterations: int | None
    best_epoch_or_iteration: int | None
    best_validation_rmse: float | None
    trainable_parameters: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return ordinary JSON-compatible values for later result tables."""

        return asdict(self)


class ModelAdapter(ABC):
    """Expose one logical interface across every model implementation.

    Subclasses implement fitting and raw prediction. This base class applies the
    common nonnegative RUL boundary and provides one trusted-local persistence
    format through joblib. Model artifacts must never be loaded from untrusted
    locations because joblib uses Python object serialization.
    """

    family: str
    representation: Representation
    stochastic = False

    def __init__(
        self,
        *,
        hyperparameters: dict[str, Any],
        seed: int,
        prediction_minimum: float = 0.0,
        training_monitor: Any | None = None,
    ) -> None:
        if not np.isfinite(prediction_minimum):
            raise ModelAdapterError("Prediction minimum must be finite")
        self.hyperparameters = dict(hyperparameters)
        self.seed = int(seed)
        self.prediction_minimum = float(prediction_minimum)
        # Model adapters know only the small neutral monitoring interface. The
        # TensorBoard writer and all filesystem decisions remain isolated in
        # the dedicated "tensorboard_monitoring" folder.
        self._training_monitor = training_monitor
        self.training_summary: TrainingSummary | None = None
        self._is_fitted = False

    @abstractmethod
    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        """Fit preprocessing and the estimator using training data only."""

    @abstractmethod
    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        """Return one unclipped floating-point prediction per input row."""

    def predict(self, data: Any) -> NDArray[np.float64]:
        """Predict RUL and enforce the common nonnegative output boundary."""

        if not self._is_fitted:
            raise ModelAdapterError(f"Model family {self.family!r} is not fitted")
        predictions = np.asarray(self._predict_raw(data), dtype=np.float64).reshape(-1)
        if len(predictions) != len(data):
            raise ModelAdapterError(
                f"Model returned {len(predictions)} predictions for {len(data)} rows"
            )
        if not np.isfinite(predictions).all():
            raise ModelAdapterError("Model produced missing or non-finite predictions")
        return np.maximum(predictions, self.prediction_minimum)

    def log_training_step(
        self,
        *,
        step: int,
        scalars: dict[str, Any],
        force: bool = False,
    ) -> bool:
        """Forward iterative progress without importing TensorBoard here.

        Only neural models and XGBoost call this method because the remaining
        libraries expose one atomic fitting operation rather than meaningful
        optimization iterations. Their start, finish, timing, and performance
        facts are recorded by the shared Step 5 and Step 6 runners.
        """

        if self._training_monitor is None:
            raise ModelAdapterError(
                f"Mandatory training monitor is missing for {self.family!r}"
            )
        return bool(
            self._training_monitor.log_training_step(
                step=step,
                scalars=scalars,
                force=force,
            )
        )

    def detach_training_monitor(self) -> None:
        """Remove the live writer before a fitted adapter is serialized."""

        self._training_monitor = None

    def require_training_monitor(self) -> Any:
        """Return the mandatory neutral monitor for a framework-specific hook."""

        if self._training_monitor is None:
            raise ModelAdapterError(
                f"Mandatory training monitor is missing for {self.family!r}"
            )
        return self._training_monitor

    def save(self, path: Path) -> Path:
        """Persist a fitted adapter, including preprocessing and model state."""

        if not self._is_fitted:
            raise ModelAdapterError("Cannot save an unfitted model adapter")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "artifact_format_version": 1,
                "model_family": self.family,
                "adapter": self,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> ModelAdapter:
        """Load a trusted local artifact and confirm its adapter type."""

        try:
            payload = joblib.load(path)
        except Exception as error:
            message = f"Cannot load model artifact {path}: {error}"
            raise ModelAdapterError(message) from error
        if not isinstance(payload, dict) or payload.get("artifact_format_version") != 1:
            raise ModelAdapterError(f"Model artifact {path} has an unknown format")
        adapter = payload.get("adapter")
        if not isinstance(adapter, cls):
            raise ModelAdapterError(
                f"Artifact contains {type(adapter).__name__}, expected {cls.__name__}"
            )
        return adapter

    @staticmethod
    def start_timer() -> float:
        """Start the shared wall-clock training timer."""

        return perf_counter()

    @staticmethod
    def elapsed_seconds(started_at: float) -> float:
        """Finish the shared wall-clock training timer."""

        return float(perf_counter() - started_at)


def load_model_adapter(path: Path) -> ModelAdapter:
    """Load any trusted Step 4 model artifact through the common base type."""

    return ModelAdapter.load(path)


def target_values(dataset: Any) -> NDArray[np.float64]:
    """Extract and validate the required one-dimensional RUL target."""

    target = getattr(dataset, "target", None)
    if target is None:
        raise ModelAdapterError("Training or validation data has no RUL target")
    values = np.asarray(target, dtype=np.float64).reshape(-1)
    if len(values) != len(dataset):
        raise ModelAdapterError("Target length does not match dataset length")
    if not np.isfinite(values).all():
        raise ModelAdapterError("RUL target contains missing or non-finite values")
    return values


def sample_weight_values(dataset: Any) -> NDArray[np.float64]:
    """Return validated weights or uniform weights when none are supplied."""

    sample_weights = getattr(dataset, "sample_weights", None)
    if sample_weights is None:
        return np.ones(len(dataset), dtype=np.float64)
    values = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if len(values) != len(dataset):
        raise ModelAdapterError("Sample-weight length does not match dataset length")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ModelAdapterError("Sample weights must be finite and positive")
    return values


def tabular_values(
    dataset: Any,
    expected_features: tuple[str, ...] | None = None,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Extract a finite tabular matrix and preserve its feature-name order."""

    features = getattr(dataset, "features", None)
    if not isinstance(features, pd.DataFrame):
        raise ModelAdapterError("Tabular adapter requires a pandas feature DataFrame")
    names = tuple(str(column) for column in features.columns)
    if not names:
        raise ModelAdapterError("Tabular feature matrix has no columns")
    if expected_features is not None and names != expected_features:
        raise ModelAdapterError("Prediction feature order differs from fitted order")
    values = features.to_numpy(dtype=np.float64)
    if values.shape != (len(dataset), len(names)):
        raise ModelAdapterError("Tabular feature matrix has an invalid shape")
    if not np.isfinite(values).all():
        raise ModelAdapterError("Tabular features contain missing or non-finite values")
    return values, names


def weighted_mean(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    """Calculate a numerically explicit weighted arithmetic mean."""

    return float(np.sum(values * weights) / np.sum(weights))


def root_mean_squared_error(
    targets: NDArray[np.float64],
    predictions: NDArray[np.float64],
) -> float:
    """Calculate the unweighted validation RMSE used for early stopping."""

    return float(np.sqrt(np.mean(np.square(predictions - targets))))
