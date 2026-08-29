"""Verify event writing and readability without running model training.

The monitoring layer publishes exactly three things, so this check confirms
exactly three things:

- a fit writes "train/loss" and "val/rmse" and nothing else,
- a Step 5 study writes "search/candidate_rmse" plus one hyperparameter text,
- Step 5 per-fit curves stay silent until the debug variable is set, and no
  locked predictive metric ever reaches a Step 6 run.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile

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

from run_layout import tensorboard_log_root  # noqa: E402
from tensorboard_monitoring import (  # noqa: E402
    FIT_CURVE_ENVIRONMENT_VARIABLE,
    TrainingRunContext,
    calculate_regression_metrics,
    create_study_monitor,
    ensure_tensorboard_available,
    log_step_5_candidate,
)
from models.neural.mlp import MLPAdapter  # noqa: E402
from models.neural.neural_base import NeuralTrainingConfig  # noqa: E402
from models.tabular.xgboost import XGBoostAdapter  # noqa: E402
from tabular_data_adapter import TabularDataset  # noqa: E402


XGBOOST_HYPERPARAMETERS = {
    "maximum_trees": 12,
    "learning_rate": 0.1,
    "max_depth": 2,
    "min_child_weight": 1.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "fault_mode_strategy": "none",
    "signal_compression_strategy": "none",
}
MLP_HYPERPARAMETERS = {
    "hidden_layers": [8],
    "dropout": 0.0,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
}


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


def _scalar_tags(run_directory: Path) -> set[str]:
    """Read back every scalar tag TensorBoard can actually see in one run."""

    if not run_directory.is_dir():
        return set()
    return set(EventAccumulator(str(run_directory)).Reload().Tags()["scalars"])


def _text_tags(run_directory: Path) -> set[str]:
    """Read back every text tag TensorBoard can actually see in one run."""

    if not run_directory.is_dir():
        return set()
    return set(EventAccumulator(str(run_directory)).Reload().Tags()["tensors"])


def _require_exact_tags(
    observed: set[str],
    expected: set[str],
    *,
    description: str,
) -> None:
    """Fail on a missing tag and equally on a tag that was quietly reintroduced."""

    if missing := sorted(expected - observed):
        raise RuntimeError(f"{description} is missing tags: {missing}")
    if unexpected := sorted(observed - expected):
        raise RuntimeError(f"{description} wrote unexpected tags: {unexpected}")


def _require_no_per_fit_subdirectories(study_directory: Path) -> None:
    """Confirm a study reused one shared directory instead of one per fit.

    Before every fit shared one writer, each candidate and inner fold created
    its own generated subdirectory below the study. Any such subdirectory
    appearing now would mean the consolidation regressed back to per-fit
    directories.
    """

    parent = study_directory.parent
    if not parent.is_dir():
        return
    unexpected = sorted(
        entry.name
        for entry in parent.iterdir()
        if entry.is_dir() and entry.name not in {"fit_progress", "study_progress"}
    )
    if unexpected:
        raise RuntimeError(
            f"Unexpected per-fit directories were created under {parent}: "
            f"{unexpected}"
        )


def _verify_iterative_model_curves(log_root: Path) -> None:
    """Fit tiny XGBoost and MLP models and inspect their real callback tags."""

    training_data = _model_dataset(24, seed=11)
    validation_data = _model_dataset(8, seed=12)

    with create_study_monitor(
        stage="step_5",
        model_family="xgboost",
        outer_fold=0,
        log_root=log_root,
    ) as xgboost_study:
        if not xgboost_study.fit_curves_enabled:
            raise RuntimeError(
                "Step 5 fit curves must be enabled for this verification run"
            )
        first_context = TrainingRunContext(
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
        with xgboost_study.fit(first_context) as monitor:
            model = XGBoostAdapter(
                hyperparameters=XGBOOST_HYPERPARAMETERS,
                seed=13,
                early_stopping_patience=3,
                training_monitor=monitor,
            )
            model.fit(training_data, validation_data)
            predictions = model.predict(validation_data)
            # Step 5 still calculates this metric; it simply is not published
            # to TensorBoard any more.
            calculate_regression_metrics(validation_data.target, predictions)
            model.detach_training_monitor()
            model_path = model.save(log_root / "verification_xgboost.joblib")
            XGBoostAdapter.load(model_path).predict(validation_data)

        # A second candidate inside the same study must reuse the study's
        # directory instead of creating a new one. That reuse is the entire
        # point of the shared writer.
        second_context = TrainingRunContext(
            stage="step_5",
            model_family="xgboost",
            representation="tabular",
            outer_fold=0,
            seed=13,
            configuration_id="verification_xgboost_second",
            candidate_number=3,
            inner_fold=1,
            feature_set="verification_features",
        )
        with xgboost_study.fit(second_context) as second_monitor:
            second_monitor.log_training_step(
                step=10,
                scalars={"train/loss": 2.0, "val/rmse": 2.5},
            )
        xgboost_log_directory = xgboost_study.log_directory

    _require_no_per_fit_subdirectories(xgboost_log_directory)
    _require_exact_tags(
        _scalar_tags(xgboost_log_directory),
        {
            "candidate_002/inner_fold_00/train/loss",
            "candidate_002/inner_fold_00/val/rmse",
            "candidate_003/inner_fold_01/train/loss",
            "candidate_003/inner_fold_01/val/rmse",
        },
        description="The XGBoost study",
    )

    with create_study_monitor(
        stage="step_5",
        model_family="mlp",
        outer_fold=0,
        log_root=log_root,
    ) as mlp_study:
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
        with mlp_study.fit(mlp_context) as monitor:
            model = MLPAdapter(
                hyperparameters=MLP_HYPERPARAMETERS,
                seed=13,
                training_config=NeuralTrainingConfig(
                    batch_size=8,
                    maximum_epochs=3,
                    early_stopping_patience=2,
                    gradient_clip_global_norm=1.0,
                ),
                training_monitor=monitor,
            )
            model.fit(training_data, validation_data)
            model.detach_training_monitor()
            model_path = model.save(log_root / "verification_mlp.joblib")
            MLPAdapter.load(model_path).predict(validation_data)
        mlp_log_directory = mlp_study.log_directory

    _require_no_per_fit_subdirectories(mlp_log_directory)
    _require_exact_tags(
        _scalar_tags(mlp_log_directory),
        {
            "candidate_002/inner_fold_01/train/loss",
            "candidate_002/inner_fold_01/val/rmse",
        },
        description="The MLP study",
    )

    # The xgboost and mlp studies are distinct families and must never share
    # a directory, even though both used outer fold 0.
    if xgboost_log_directory == mlp_log_directory:
        raise RuntimeError(
            "Distinct model families unexpectedly shared one study directory"
        )


def _verify_search_curve(log_root: Path) -> None:
    """Confirm the Step 5 study curve carries one point per candidate."""

    for candidate_number, rmse in enumerate([31.0, 28.5, 29.75], start=1):
        log_step_5_candidate(
            model_family="xgboost",
            outer_fold=0,
            candidate_number=candidate_number,
            mean_inner_rmse=rmse,
            hyperparameters={**XGBOOST_HYPERPARAMETERS, "max_depth": candidate_number},
            log_root=log_root,
        )

    study_progress = (
        log_root / "step_5" / "xgboost" / "outer_fold_00" / "study_progress"
    )
    _require_exact_tags(
        _scalar_tags(study_progress),
        {"search/candidate_rmse"},
        description="The Step 5 search curve",
    )
    events = EventAccumulator(str(study_progress)).Reload()
    steps = [event.step for event in events.Scalars("search/candidate_rmse")]
    if steps != [1, 2, 3]:
        raise RuntimeError(
            f"The search curve should hold one point per candidate, got {steps}"
        )
    # PyTorch's add_text stores each text tag with a "/text_summary" suffix.
    expected_text = {
        "search/candidate_001/text_summary",
        "search/candidate_002/text_summary",
        "search/candidate_003/text_summary",
    }
    if missing := sorted(expected_text - _text_tags(study_progress)):
        raise RuntimeError(f"Candidate hyperparameter text is missing: {missing}")


def _verify_step_6_boundary(log_root: Path) -> None:
    """Confirm Step 6 publishes a training curve and no locked score."""

    training_data = _model_dataset(24, seed=21)
    with create_study_monitor(
        stage="step_6",
        model_family="mlp",
        outer_fold=0,
        log_root=log_root,
    ) as step_6_study:
        if not step_6_study.fit_curves_enabled:
            raise RuntimeError("Step 6 must always publish its training curve")
        for seed in (13, 37):
            context = TrainingRunContext(
                stage="step_6",
                model_family="mlp",
                representation="tabular",
                outer_fold=0,
                seed=seed,
                configuration_id=f"verification_locked_seed_{seed}",
                feature_set="verification_features",
            )
            with step_6_study.fit(context) as monitor:
                model = MLPAdapter(
                    hyperparameters=MLP_HYPERPARAMETERS,
                    seed=seed,
                    training_config=NeuralTrainingConfig(
                        batch_size=8,
                        maximum_epochs=3,
                        early_stopping_patience=2,
                        gradient_clip_global_norm=1.0,
                    ),
                    training_epochs=3,
                    training_monitor=monitor,
                )
                # Step 6 never receives the locked validation dataset, so it
                # cannot publish a locked predictive metric even by accident.
                model.fit(training_data, None)
                model.detach_training_monitor()
        step_6_log_directory = step_6_study.log_directory

    _require_no_per_fit_subdirectories(step_6_log_directory)
    _require_exact_tags(
        _scalar_tags(step_6_log_directory),
        {"seed_013/train/loss", "seed_037/train/loss"},
        description="The Step 6 study",
    )


def _verify_step_5_curves_are_opt_in(log_root: Path) -> None:
    """Confirm a normal Step 5 study writes no per-fit curve at all."""

    previous = os.environ.pop(FIT_CURVE_ENVIRONMENT_VARIABLE, None)
    try:
        with create_study_monitor(
            stage="step_5",
            model_family="mlp",
            outer_fold=1,
            log_root=log_root,
        ) as study:
            if study.fit_curves_enabled:
                raise RuntimeError(
                    "Step 5 fit curves must be off unless "
                    f"{FIT_CURVE_ENVIRONMENT_VARIABLE} is set"
                )
            context = TrainingRunContext(
                stage="step_5",
                model_family="mlp",
                representation="tabular",
                outer_fold=1,
                seed=13,
                configuration_id="verification_quiet",
                candidate_number=1,
                inner_fold=0,
            )
            with study.fit(context) as monitor:
                if monitor.log_training_step(
                    step=1,
                    scalars={"train/loss": 1.0},
                ):
                    raise RuntimeError(
                        "A disabled Step 5 study reported a written curve point"
                    )
            quiet_directory = study.log_directory
    finally:
        if previous is not None:
            os.environ[FIT_CURVE_ENVIRONMENT_VARIABLE] = previous

    if _scalar_tags(quiet_directory):
        raise RuntimeError(
            f"A disabled Step 5 study wrote events to {quiet_directory}"
        )


def main() -> None:
    """Write isolated studies and confirm TensorBoard reads exactly what is logged."""

    installed_version = ensure_tensorboard_available()
    # A scratch root outside any real run folder, so verification can never
    # touch or replace a run's actual curves.
    log_root = Path(tempfile.mkdtemp(prefix="tensorboard_verification_"))
    previous = os.environ.get(FIT_CURVE_ENVIRONMENT_VARIABLE)
    # The per-fit checks need the debug curves the pipeline leaves switched
    # off; _verify_step_5_curves_are_opt_in removes the variable again.
    os.environ[FIT_CURVE_ENVIRONMENT_VARIABLE] = "1"
    try:
        _verify_iterative_model_curves(log_root)
        _verify_search_curve(log_root)
        _verify_step_6_boundary(log_root)
        _verify_step_5_curves_are_opt_in(log_root)
    finally:
        if previous is None:
            os.environ.pop(FIT_CURVE_ENVIRONMENT_VARIABLE, None)
        else:
            os.environ[FIT_CURVE_ENVIRONMENT_VARIABLE] = previous
        shutil.rmtree(log_root, ignore_errors=True)

    print(f"TensorBoard {installed_version} monitoring verified")
    print(f"Event files are written per run, e.g. {tensorboard_log_root(1)}")
    print("Fit curves: train/loss, val/rmse")
    print("Step 5 study curve: search/candidate_rmse plus candidate text")
    print("Step 6 published no locked predictive metric")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as error:
        print(f"TensorBoard monitoring verification failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
