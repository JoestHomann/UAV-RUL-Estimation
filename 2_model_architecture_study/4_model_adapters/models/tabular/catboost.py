"""Implement the sample-weighted CatBoost architecture."""

from __future__ import annotations

from typing import Any

from catboost import CatBoostRegressor, Pool
import numpy as np
from numpy.typing import NDArray

from base import (
    ModelAdapter,
    ModelAdapterError,
    TrainingSummary,
    root_mean_squared_error,
    sample_weight_values,
    tabular_values,
    target_values,
)


class CatBoostAdapter(ModelAdapter):
    """Fit deterministic CPU CatBoost with optional validation stopping."""

    family = "catboost"
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
    ) -> None:
        """Store inner-stopping or fixed outer-retraining settings."""

        super().__init__(
            hyperparameters=hyperparameters,
            seed=seed,
            prediction_minimum=prediction_minimum,
            training_monitor=training_monitor,
        )
        if training_iterations is not None and training_iterations <= 0:
            raise ModelAdapterError("Fixed CatBoost iterations must be positive")
        self.early_stopping_patience = early_stopping_patience
        self.training_iterations = training_iterations
        # CatBoost documents GPU training as nondeterministic. The architecture
        # study therefore fixes this family to one CPU worker.
        self.device = "cpu"

    def _log_training_curves(self) -> None:
        """Publish CatBoost's structured metric history to the shared monitor."""

        evaluations = self.estimator.get_evals_result()
        training = evaluations.get("learn", {}).get("RMSE", [])
        validation = evaluations.get("validation", {}).get("RMSE", [])
        for index, training_rmse in enumerate(training, start=1):
            scalars: dict[str, float] = {"train/loss": float(training_rmse)}
            if index <= len(validation):
                scalars["val/rmse"] = float(validation[index - 1])
            self.log_training_step(
                step=index,
                scalars=scalars,
                force=index == len(training),
            )

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        """Fit boosted symmetric trees with optional inner-fold stopping."""

        started_at = self.start_timer()
        training_values, self.feature_names = tabular_values(training_data)
        targets = target_values(training_data)
        weights = sample_weight_values(training_data)
        training_pool = Pool(
            training_values,
            label=targets,
            weight=weights,
            feature_names=list(self.feature_names),
        )

        maximum_trees = int(self.hyperparameters["maximum_trees"])
        tree_count = self.training_iterations or maximum_trees
        use_early_stopping = (
            validation_data is not None
            and self.training_iterations is None
            and self.early_stopping_patience is not None
        )
        self.require_training_monitor()
        self.estimator = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=tree_count,
            learning_rate=float(self.hyperparameters["learning_rate"]),
            depth=int(self.hyperparameters["depth"]),
            l2_leaf_reg=float(self.hyperparameters["l2_leaf_reg"]),
            random_strength=float(self.hyperparameters["random_strength"]),
            bagging_temperature=float(
                self.hyperparameters["bagging_temperature"]
            ),
            rsm=float(self.hyperparameters["rsm"]),
            boosting_type=str(self.hyperparameters["boosting_type"]),
            random_seed=self.seed,
            task_type="CPU",
            thread_count=1,
            allow_writing_files=False,
            verbose=False,
        )

        validation_pool = None
        if validation_data is not None:
            validation_values, _ = tabular_values(
                validation_data,
                self.feature_names,
            )
            validation_pool = Pool(
                validation_values,
                label=target_values(validation_data),
                weight=sample_weight_values(validation_data),
                feature_names=list(self.feature_names),
            )
        self.estimator.fit(
            training_pool,
            eval_set=validation_pool,
            use_best_model=use_early_stopping,
            early_stopping_rounds=(
                int(self.early_stopping_patience) if use_early_stopping else None
            ),
            verbose=False,
        )
        self._log_training_curves()
        self._is_fitted = True

        validation_rmse = None
        validation_rows = 0
        if validation_data is not None:
            validation_rows = len(validation_data)
            validation_rmse = root_mean_squared_error(
                target_values(validation_data),
                self.predict(validation_data),
            )
        best_iteration = self.estimator.get_best_iteration()
        completed_iterations = int(self.estimator.tree_count_)
        self.training_summary = TrainingSummary(
            model_family=self.family,
            seed=self.seed,
            training_rows=len(training_data),
            validation_rows=validation_rows,
            training_seconds=self.elapsed_seconds(started_at),
            epochs_or_iterations=completed_iterations,
            best_epoch_or_iteration=(
                int(best_iteration) + 1
                if (
                    use_early_stopping
                    and best_iteration is not None
                    and best_iteration >= 0
                )
                else None
            ),
            best_validation_rmse=validation_rmse,
            trainable_parameters=None,
        )
        return self.training_summary

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        """Predict from unscaled features in their fitted order."""

        values, _ = tabular_values(data, self.feature_names)
        return np.asarray(self.estimator.predict(values), dtype=np.float64)
