"""Verify event writing and readability without running model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PHASE_DIR = Path(__file__).resolve().parent.parent
DEPENDENCY_DIRS = (
    PHASE_DIR,
    PHASE_DIR / "2_tabular_data_adapter",
    PHASE_DIR / "4_model_adapters",
)
for dependency_dir in DEPENDENCY_DIRS:
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from tensorboard_monitoring import (  # noqa: E402
    DEFAULT_LOG_ROOT,
    TrainingRunContext,
    calculate_age_band_regression_metrics,
    calculate_regression_metrics,
    create_training_monitor,
    ensure_tensorboard_available,
    publish_step_7_comparison,
)
from models.neural.mlp import MLPAdapter  # noqa: E402
from models.neural.neural_base import NeuralTrainingConfig  # noqa: E402
from models.tabular.xgboost import XGBoostAdapter  # noqa: E402
from tabular_data_adapter import TabularDataset  # noqa: E402


@dataclass(frozen=True)
class _SmallDataset:
    """Provide only the dimensions inspected by the monitoring layer."""

    rows: int
    columns: int

    @property
    def features(self) -> np.ndarray:
        """Return a small matrix without loading any project dataset."""

        return np.zeros((self.rows, self.columns), dtype=np.float64)

    def __len__(self) -> int:
        """Return the synthetic row count."""

        return self.rows


def _model_dataset(rows: int, *, seed: int) -> TabularDataset:
    """Create a deterministic nonlinear regression sample for callback checks."""

    generator = np.random.default_rng(seed)
    features = generator.normal(size=(rows, 3))
    target = (
        3.0 * features[:, 0]
        - 1.5 * features[:, 1]
        + np.square(features[:, 2])
        + 10.0
    )
    return TabularDataset(
        features=pd.DataFrame(
            features,
            columns=["feature_a", "feature_b", "feature_c"],
        ),
        metadata=pd.DataFrame(
            {"uav_id": [f"verification_{index}" for index in range(rows)]}
        ),
        target=pd.Series(target, name="RUL"),
        sample_weights=pd.Series(np.ones(rows), name="sample_weight"),
    )


def _verify_iterative_model_callbacks(log_root: Path) -> None:
    """Fit tiny XGBoost and MLP models and inspect their real callback tags."""

    training_data = _model_dataset(24, seed=11)
    validation_data = _model_dataset(8, seed=12)

    xgboost_context = TrainingRunContext(
        stage="step_5",
        model_family="xgboost",
        representation="tabular",
        outer_fold=0,
        seed=13,
        configuration_id="verification_xgboost",
        candidate_number=2,
        inner_fold=0,
        feature_set="verification_features",
    )
    xgboost_hyperparameters = {
        "maximum_trees": 12,
        "learning_rate": 0.1,
        "max_depth": 2,
        "min_child_weight": 1.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    with create_training_monitor(xgboost_context, log_root=log_root) as monitor:
        monitor.start_fit(
            hyperparameters=xgboost_hyperparameters,
            training_data=training_data,
            validation_data=validation_data,
        )
        model = XGBoostAdapter(
            hyperparameters=xgboost_hyperparameters,
            seed=13,
            early_stopping_patience=3,
            training_monitor=monitor,
        )
        summary = model.fit(training_data, validation_data)
        predictions = model.predict(validation_data)
        monitor.complete_fit(
            training_summary=summary.to_dict(),
            inference_seconds=0.0,
            evaluation_metrics=calculate_regression_metrics(
                validation_data.target,
                predictions,
            ),
            prediction_rows=len(predictions),
        )
        model.detach_training_monitor()
        xgboost_model_path = model.save(log_root / "verification_xgboost.joblib")
        restored_xgboost = XGBoostAdapter.load(xgboost_model_path)
        restored_xgboost.predict(validation_data)
        xgboost_log_directory = monitor.log_directory

    xgboost_tags = set(
        EventAccumulator(str(xgboost_log_directory)).Reload().Tags()["scalars"]
    )
    required_xgboost_tags = {
        "optimization/training_rmse",
        "optimization/validation_rmse",
        "timing/seconds_per_iteration",
    }
    if missing := sorted(required_xgboost_tags - xgboost_tags):
        raise RuntimeError(f"XGBoost callback tags are missing: {missing}")

    mlp_context = TrainingRunContext(
        stage="step_5",
        model_family="mlp",
        representation="tabular",
        outer_fold=0,
        seed=13,
        configuration_id="verification_mlp",
        candidate_number=2,
        inner_fold=1,
        feature_set="verification_features",
    )
    mlp_hyperparameters = {
        "hidden_layers": [8],
        "dropout": 0.0,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
    }
    with create_training_monitor(mlp_context, log_root=log_root) as monitor:
        monitor.start_fit(
            hyperparameters=mlp_hyperparameters,
            training_data=training_data,
            validation_data=validation_data,
        )
        model = MLPAdapter(
            hyperparameters=mlp_hyperparameters,
            seed=13,
            training_config=NeuralTrainingConfig(
                batch_size=8,
                maximum_epochs=3,
                early_stopping_patience=2,
                gradient_clip_global_norm=1.0,
            ),
            training_monitor=monitor,
        )
        summary = model.fit(training_data, validation_data)
        predictions = model.predict(validation_data)
        monitor.complete_fit(
            training_summary=summary.to_dict(),
            inference_seconds=0.0,
            evaluation_metrics=calculate_regression_metrics(
                validation_data.target,
                predictions,
            ),
            prediction_rows=len(predictions),
        )
        model.detach_training_monitor()
        mlp_model_path = model.save(log_root / "verification_mlp.joblib")
        restored_mlp = MLPAdapter.load(mlp_model_path)
        restored_mlp.predict(validation_data)
        mlp_log_directory = monitor.log_directory

    mlp_tags = set(
        EventAccumulator(str(mlp_log_directory)).Reload().Tags()["scalars"]
    )
    required_mlp_tags = {
        "optimization/training_weighted_mse",
        "optimization/validation_rmse",
        "optimization/learning_rate",
        "timing/seconds_per_epoch",
        "early_stopping/patience_used",
    }
    if missing := sorted(required_mlp_tags - mlp_tags):
        raise RuntimeError(f"Neural callback tags are missing: {missing}")


def _verify_locked_metric_boundary(log_root: Path) -> None:
    """Confirm Step 6 hides scores and Step 7 publishes them after completion."""

    context = TrainingRunContext(
        stage="step_6",
        model_family="random_forest",
        representation="tabular",
        outer_fold=0,
        seed=13,
        configuration_id="verification_locked_run",
        feature_set="verification_features",
    )
    training_data = _SmallDataset(rows=8, columns=3)
    with create_training_monitor(context, log_root=log_root) as monitor:
        monitor.start_fit(
            hyperparameters={"n_estimators": 10},
            training_data=training_data,
            validation_data=None,
        )
        monitor.complete_fit(
            training_summary={
                "training_seconds": 1.0,
                "epochs_or_iterations": 10,
                "best_epoch_or_iteration": None,
                "trainable_parameters": None,
            },
            inference_seconds=0.1,
            evaluation_metrics=None,
            prediction_rows=4,
        )
        step_6_log_directory = monitor.log_directory

    step_6_tags = set(
        EventAccumulator(str(step_6_log_directory)).Reload().Tags()["scalars"]
    )
    leaked = sorted(tag for tag in step_6_tags if tag.startswith("performance/"))
    if leaked:
        raise RuntimeError(f"Step 6 exposed locked performance tags: {leaked}")

    comparison = pd.DataFrame(
        [
            {
                "model_family": "random_forest",
                **{
                    f"{metric}_{suffix}": value
                    for metric, value in {
                        "rmse": 10.0,
                        "mae": 8.0,
                        "r2": 0.8,
                        "bias": 1.0,
                    }.items()
                    for suffix in (
                        "mean",
                        "seed_sd",
                        "ci_lower_95",
                        "ci_upper_95",
                    )
                },
            }
        ]
    )
    efficiency = pd.DataFrame(
        [
            {
                "model_family": "random_forest",
                "training_seconds_mean_per_run": 1.0,
                "training_seconds_total": 5.0,
                "inference_seconds_mean_per_run": 0.1,
                "inference_milliseconds_per_endpoint": 0.25,
                "trainable_parameters_mean": float("nan"),
                "serialized_model_bytes_mean": 1024.0,
            }
        ]
    )
    grouped = pd.DataFrame(
        [
            {
                "model_family": "random_forest",
                "group_type": "age_band",
                "group_value": "1-50",
                **{
                    f"{metric}_{suffix}": value
                    for metric, value in {
                        "rmse": 11.0,
                        "mae": 9.0,
                        "r2": 0.7,
                        "bias": 1.5,
                    }.items()
                    for suffix in ("mean", "seed_sd")
                },
            }
        ]
    )
    publish_step_7_comparison(
        comparison,
        efficiency,
        grouped,
        log_root=log_root,
    )
    step_7_directory = (
        log_root / "step_7" / "final_comparison" / "random_forest"
    )
    step_7_tags = set(
        EventAccumulator(str(step_7_directory)).Reload().Tags()["scalars"]
    )
    if "locked_performance/rmse_mean" not in step_7_tags:
        raise RuntimeError("Step 7 did not publish final locked performance")
    if "locked_age_band/1_50/rmse_mean" not in step_7_tags:
        raise RuntimeError("Step 7 did not publish locked age-band performance")


def main() -> None:
    """Write one isolated run and confirm TensorBoard can read every key tag."""

    installed_version = ensure_tensorboard_available()
    context = TrainingRunContext(
        stage="step_5",
        model_family="xgboost",
        representation="tabular",
        outer_fold=0,
        seed=13,
        configuration_id="verification_candidate",
        candidate_number=1,
        inner_fold=0,
        feature_set="verification_features",
    )
    training_data = _SmallDataset(rows=8, columns=3)
    validation_data = _SmallDataset(rows=4, columns=3)
    metrics = calculate_regression_metrics(
        targets=np.array([4.0, 3.0, 2.0, 1.0]),
        predictions=np.array([3.5, 3.0, 2.5, 1.0]),
    )
    age_band_metrics = calculate_age_band_regression_metrics(
        targets=np.array([4.0, 3.0, 2.0, 1.0]),
        predictions=np.array([3.5, 3.0, 2.5, 1.0]),
        cutoffs=np.array([25, 50, 75, 100]),
    )

    DEFAULT_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_root = DEFAULT_LOG_ROOT / "verification_scratch"
    try:
        with create_training_monitor(context, log_root=log_root) as monitor:
            monitor.start_fit(
                hyperparameters={"learning_rate": 0.05},
                training_data=training_data,
                validation_data=validation_data,
            )
            monitor.log_training_step(
                step=10,
                scalars={
                    "optimization/training_rmse": 2.0,
                    "optimization/validation_rmse": 2.5,
                },
            )
            monitor.complete_fit(
                training_summary={
                    "training_seconds": 1.25,
                    "epochs_or_iterations": 10,
                    "best_epoch_or_iteration": 8,
                    "trainable_parameters": None,
                },
                inference_seconds=0.05,
                evaluation_metrics={**metrics, **age_band_metrics},
                prediction_rows=4,
            )
            run_directory = monitor.log_directory

        events = EventAccumulator(str(run_directory)).Reload()
        scalar_tags = set(events.Tags()["scalars"])
        required_tags = {
            "data/training_rows",
            "data/validation_rows",
            "data/input_features",
            "optimization/training_rmse",
            "optimization/validation_rmse",
            "performance/rmse",
            "performance/mae",
            "performance/r2",
            "performance/bias",
            "performance/age_band/1_50/rmse",
            "performance/age_band/51_100/rmse",
            "timing/training_seconds",
            "timing/inference_seconds",
            "progress/status",
        }
        missing = sorted(required_tags - scalar_tags)
        if missing:
            raise RuntimeError(
                f"TensorBoard verification is missing scalar tags: {missing}"
            )
        status_values = events.Scalars("progress/status")
        if not status_values or float(status_values[-1].value) != 1.0:
            raise RuntimeError(
                "TensorBoard verification did not record a completed fit"
            )
        _verify_iterative_model_callbacks(log_root)
        _verify_locked_metric_boundary(log_root)
    finally:
        # This exact generated directory is fixed below the ignored log root.
        # The parent check prevents a future edit from broadening the cleanup.
        if log_root.parent.resolve() != DEFAULT_LOG_ROOT.resolve():
            raise RuntimeError("Unsafe TensorBoard verification cleanup path")
        if log_root.exists():
            shutil.rmtree(log_root)

    print(f"TensorBoard {installed_version} monitoring verification passed")


if __name__ == "__main__":
    main()
