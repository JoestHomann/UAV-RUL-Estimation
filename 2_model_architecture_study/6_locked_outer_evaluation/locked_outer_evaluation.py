"""Retrain selected configurations and predict the locked outer scenarios.

The Step 5 completion gate is evaluated before this module constructs either
data adapter. Every selected configuration is retrained on the 80 outer-training
UAVs, then predicts the 20 held-out UAVs across 20 locked scenarios. Locked
targets are never passed into model fitting or early stopping.
"""

from __future__ import annotations

from importlib.metadata import version
import json
import os
from pathlib import Path
from time import perf_counter, sleep
import sys
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
PHASE_DIR = STEP_DIR.parent
DEFAULT_OUTPUT_DIR = STEP_DIR / "artifacts"

DEPENDENCY_DIRS = [
    PHASE_DIR,
    PHASE_DIR / "2_tabular_data_adapter",
    PHASE_DIR / "3_sequence_data_adapter",
    PHASE_DIR / "4_model_adapters",
]
for dependency_dir in DEPENDENCY_DIRS:
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from base import ModelAdapter, target_values  # noqa: E402
from model_registry import ModelAdapterFactory  # noqa: E402
from sequence_data_adapter import SequenceDataAdapter  # noqa: E402
from tabular_data_adapter import TabularDataAdapter  # noqa: E402
from tensorboard_monitoring import (  # noqa: E402
    TrainingRunContext,
    create_study_monitor,
    ensure_tensorboard_available,
)

from evaluation_gate import (  # noqa: E402
    DEFAULT_SELECTED_CONFIGURATIONS_PATH,
    DEFAULT_SELECTION_MANIFEST_PATH,
    DEFAULT_SPECIFICATION_PATH,
    LockedEvaluationPlan,
    SelectedConfiguration,
    build_locked_evaluation_plan,
)


RUNNER_VERSION = 1

PREDICTION_COLUMNS = [
    "settings_version",
    "model_family",
    "configuration_id",
    "seed",
    "outer_fold",
    "scenario",
    "sample_id",
    "uav_id",
    "cutoff",
    "terminal_lifetime",
    "lifetime_quantile",
    "feature_set",
    "lookback",
    "y_true",
    "y_pred",
    "residual",
]

RUN_COLUMNS = [
    "settings_version",
    "model_family",
    "configuration_id",
    "seed",
    "outer_fold",
    "feature_set",
    "lookback",
    "selected_mean_inner_rmse",
    "fixed_retraining_iterations",
    "training_rows",
    "training_uavs",
    "validation_rows",
    "validation_uavs",
    "locked_scenarios",
    "training_seconds",
    "inference_seconds",
    "epochs_or_iterations",
    "trainable_parameters",
    "serialized_model_bytes",
    "prediction_rows",
    "model_path",
    "prediction_path",
]


class LockedOuterEvaluationError(ValueError):
    """Represent a locked split, prediction, or artifact consistency failure."""


def _replace_with_retry(temporary_path: Path, path: Path) -> None:
    """Atomically replace ``path`` with ``temporary_path``, tolerating Windows locks.

    POSIX ``rename`` succeeds even while another process has ``path`` open.
    Windows instead raises ``PermissionError`` (WinError 32) if any process --
    including a sibling Step 6 family/outer-fold subprocess consolidating this
    same shared artifact a few milliseconds apart -- currently has it open.
    That lock is always transient (the other process releases it as soon as
    its own read or replace finishes), so a short bounded retry resolves it
    without weakening the atomicity of the replace itself.
    """

    last_error: PermissionError | None = None
    for attempt in range(10):
        try:
            temporary_path.replace(path)
            return
        except PermissionError as error:
            last_error = error
            sleep(0.1 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _unique_temporary_path(path: Path) -> Path:
    """Choose a per-writer temporary path so concurrent writers never race
    on the temporary file itself, only on the final atomic replace.

    A fixed name such as ``locked_evaluation_manifest.json.tmp`` is itself
    contended when two Step 6 subprocesses consolidate the same shared
    artifact a few milliseconds apart: one process can still be writing that
    file when the other tries to open the same path, which raises
    ``PermissionError`` before either process even reaches
    ``_replace_with_retry``. Mixing in the process id and a random token
    makes every writer's temporary file unique, so the destination path is
    the only thing ever shared -- exactly what the retry-wrapped replace
    above is designed to tolerate.
    """

    return path.with_suffix(path.suffix + f".{os.getpid()}.{uuid4().hex[:8]}.tmp")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Atomically replace one readable generated JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _unique_temporary_path(path)
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _replace_with_retry(temporary_path, path)


def _write_csv(
    records: list[dict[str, Any]],
    columns: list[str],
    path: Path,
    *,
    compressed: bool = False,
) -> None:
    """Write one stable-column CSV, optionally using deterministic gzip time."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records, columns=columns)
    temporary_path = _unique_temporary_path(path)
    compression: str | dict[str, Any] | None = None
    if compressed:
        compression = {"method": "gzip", "mtime": 0}
    frame.to_csv(
        temporary_path,
        index=False,
        float_format="%.12g",
        compression=compression,
    )
    _replace_with_retry(temporary_path, path)


def _atomic_save_model(model: ModelAdapter, path: Path) -> int:
    """Save a complete trusted-local adapter before replacing an older file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _unique_temporary_path(path)
    model.save(temporary_path)
    _replace_with_retry(temporary_path, path)
    return int(path.stat().st_size)


class LockedRunArtifactWriter:
    """Manage isolated artifacts for one family, outer fold, and seed."""

    def __init__(
        self,
        output_dir: Path,
        selected: SelectedConfiguration,
        seed: int,
    ) -> None:
        self.output_dir = output_dir
        self.selected = selected
        self.seed = seed
        self.prefix = (
            f"{selected.model_family}__outer_{selected.outer_fold:02d}"
            f"__seed_{seed:03d}"
        )
        self.run_dir = output_dir / "runs"
        self.model_path = output_dir / "models" / f"{self.prefix}.joblib"
        self.prediction_path = self.run_dir / f"{self.prefix}__predictions.csv.gz"
        self.record_path = self.run_dir / f"{self.prefix}__run.json"
        self.status_path = self.run_dir / f"{self.prefix}__status.json"

    def start(self) -> None:
        """Invalidate any previous run before locked data is requested."""

        self._write_status("running")

    def finish(
        self,
        predictions: pd.DataFrame,
        run_record: dict[str, Any],
    ) -> None:
        """Write validated predictions and mark the run complete last."""

        _write_csv(
            predictions.to_dict("records"),
            PREDICTION_COLUMNS,
            self.prediction_path,
            compressed=True,
        )
        _write_json(run_record, self.record_path)
        self._write_status(
            "complete",
            prediction_rows=len(predictions),
            locked_data_loaded=True,
        )

    def fail(self, message: str, *, locked_data_loaded: bool) -> None:
        """Prevent incomplete outputs from entering consolidated results."""

        self._write_status(
            "failed",
            locked_data_loaded=locked_data_loaded,
            error=message,
        )

    def _write_status(
        self,
        state: str,
        *,
        prediction_rows: int = 0,
        locked_data_loaded: bool = False,
        error: str | None = None,
    ) -> None:
        """Record state without hashes, timestamps, or automatic rankings."""

        payload: dict[str, Any] = {
            "runner_version": RUNNER_VERSION,
            "settings_version": self.selected.settings_version,
            "model_family": self.selected.model_family,
            "configuration_id": self.selected.configuration_id,
            "outer_fold": self.selected.outer_fold,
            "seed": self.seed,
            "state": state,
            "prediction_rows": prediction_rows,
            "locked_data_loaded": locked_data_loaded,
            "test_data_loaded": False,
        }
        if error is not None:
            payload["error"] = error
        _write_json(payload, self.status_path)


class LockedOuterEvaluationRunner:
    """Execute locked evaluation only after every Step 5 study is complete."""

    def __init__(
        self,
        specification_path: Path = DEFAULT_SPECIFICATION_PATH,
        selection_manifest_path: Path = DEFAULT_SELECTION_MANIFEST_PATH,
        selected_configurations_path: Path = DEFAULT_SELECTED_CONFIGURATIONS_PATH,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        # This is the critical ordering constraint: the gate reads only Step 1,
        # Step 4, and Step 5 metadata. A failure occurs before data adapters are
        # constructed and therefore before locked tables can be loaded.
        ensure_tensorboard_available()
        self.plan = build_locked_evaluation_plan(
            specification_path,
            selection_manifest_path,
            selected_configurations_path,
        )
        self.output_dir = output_dir.resolve()
        self.factory = ModelAdapterFactory(specification_path)
        self.settings = self.plan.settings
        self.settings_version = int(self.settings["settings_version"])
        self.locked_scenarios = int(
            self.settings["phase_1"]["expected_locked_scenarios"]
        )
        self.training_uavs = int(
            self.settings["phase_1"]["expected_training_uavs"]
        )
        self.prefixes_per_uav = int(
            self.settings["phase_1"]["expected_prefixes_per_training_uav"]
        )
        self._tabular_adapter: TabularDataAdapter | None = None
        self._sequence_adapter: SequenceDataAdapter | None = None

    def validate_request(
        self,
        families: list[str] | None,
        outer_folds: list[int] | None,
    ) -> tuple[list[str], list[int]]:
        """Resolve optional filters while preserving the completed Step 5 plan."""

        selected_families = (
            list(self.plan.enabled_families) if families is None else families
        )
        if not selected_families:
            raise LockedOuterEvaluationError("At least one model family is required")
        if len(selected_families) != len(set(selected_families)):
            raise LockedOuterEvaluationError(
                "Requested model families contain duplicates"
            )
        invalid_families = [
            family
            for family in selected_families
            if family not in self.plan.enabled_families
        ]
        if invalid_families:
            raise LockedOuterEvaluationError(
                "Families are absent from the completed Step 5 plan: "
                f"{invalid_families}"
            )

        selected_folds = (
            list(self.plan.outer_fold_labels)
            if outer_folds is None
            else outer_folds
        )
        if len(selected_folds) != len(set(selected_folds)):
            raise LockedOuterEvaluationError(
                "Requested outer folds contain duplicates"
            )
        invalid_folds = [
            fold
            for fold in selected_folds
            if fold not in self.plan.outer_fold_labels
        ]
        if invalid_folds:
            raise LockedOuterEvaluationError(
                f"Unknown outer folds {invalid_folds}. "
                f"Available: {list(self.plan.outer_fold_labels)}"
            )
        return selected_families, selected_folds

    def run(
        self,
        families: list[str] | None = None,
        outer_folds: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run requested locked evaluations and consolidate complete outputs."""

        selected_families, selected_folds = self.validate_request(
            families,
            outer_folds,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for family in selected_families:
            for outer_fold in selected_folds:
                selected = self.plan.configuration(family, outer_fold)
                # One shared writer covers every retraining seed for this
                # family/outer-fold pair, so the generated directory count
                # stays fixed at one study instead of growing with the
                # retraining seed count.
                with create_study_monitor(
                    stage="step_6",
                    model_family=family,
                    outer_fold=outer_fold,
                ) as study_monitor:
                    for seed in self.plan.seeds_for(family):
                        self.run_model(selected, seed, study_monitor)
        return self.consolidate_artifacts()

    def run_model(
        self,
        selected: SelectedConfiguration,
        seed: int,
        study_monitor: Any,
    ) -> dict[str, Any]:
        """Retrain and evaluate one selected family/fold/seed combination."""

        if seed not in self.plan.seeds_for(selected.model_family):
            raise LockedOuterEvaluationError(
                f"Seed {seed} is not configured for {selected.model_family!r}"
            )
        writer = LockedRunArtifactWriter(self.output_dir, selected, seed)
        writer.start()
        self.consolidate_artifacts()

        locked_data_loaded = False
        try:
            split = self._locked_split(selected)
            locked_data_loaded = True
            split_facts = self._validate_locked_split(split, selected.outer_fold)
            architecture = self.settings["architectures"][selected.model_family]
            context = TrainingRunContext(
                stage="step_6",
                model_family=selected.model_family,
                representation=architecture["representation"],
                outer_fold=selected.outer_fold,
                seed=seed,
                configuration_id=selected.configuration_id,
                feature_set=selected.feature_set,
                lookback=selected.lookback,
            )
            model = None
            with study_monitor.fit(context) as monitor:
                # The locked validation dataset is deliberately omitted here.
                # Step 6 may display training progress and operating timings,
                # but no locked predictive metric before Step 7 completes.
                monitor.start_fit(
                    hyperparameters=selected.hyperparameters,
                    training_data=split.training,
                    validation_data=None,
                )
                model = self.factory.create(
                    selected.model_family,
                    selected.hyperparameters,
                    seed=seed,
                    training_iterations=selected.outer_retraining_iterations,
                    training_monitor=monitor,
                )
                try:
                    # Passing None is the model-facing leakage boundary. Locked
                    # labels are used only after training has finished.
                    training_summary = model.fit(split.training, None)
                    if training_summary.validation_rows != 0:
                        raise LockedOuterEvaluationError(
                            "Model training unexpectedly consumed locked "
                            "validation rows"
                        )
                    if training_summary.best_validation_rmse is not None:
                        raise LockedOuterEvaluationError(
                            "Model training unexpectedly calculated locked "
                            "validation RMSE"
                        )
                    if (
                        selected.outer_retraining_iterations is not None
                        and training_summary.epochs_or_iterations
                        != selected.outer_retraining_iterations
                    ):
                        raise LockedOuterEvaluationError(
                            "Completed training duration differs from the "
                            "Step 5 duration"
                        )

                    prediction_started = perf_counter()
                    predicted_rul = model.predict(split.validation)
                    inference_seconds = float(perf_counter() - prediction_started)
                    predictions = self._prediction_table(
                        split.validation,
                        selected,
                        seed,
                        predicted_rul,
                    )

                    # SummaryWriter cannot be serialized and is not part of a
                    # scientific model artifact. Detach it before joblib writes.
                    model.detach_training_monitor()
                    serialized_bytes = _atomic_save_model(model, writer.model_path)
                    run_record = {
                        "settings_version": self.settings_version,
                        "model_family": selected.model_family,
                        "configuration_id": selected.configuration_id,
                        "seed": seed,
                        "outer_fold": selected.outer_fold,
                        "feature_set": selected.feature_set,
                        "lookback": selected.lookback,
                        "selected_mean_inner_rmse": selected.mean_inner_rmse,
                        "fixed_retraining_iterations": (
                            selected.outer_retraining_iterations
                        ),
                        "training_rows": split_facts["training_rows"],
                        "training_uavs": split_facts["training_uavs"],
                        "validation_rows": split_facts["validation_rows"],
                        "validation_uavs": split_facts["validation_uavs"],
                        "locked_scenarios": split_facts["locked_scenarios"],
                        "training_seconds": training_summary.training_seconds,
                        "inference_seconds": inference_seconds,
                        "epochs_or_iterations": (
                            training_summary.epochs_or_iterations
                        ),
                        "trainable_parameters": (
                            training_summary.trainable_parameters
                        ),
                        "serialized_model_bytes": serialized_bytes,
                        "prediction_rows": len(predictions),
                        "model_path": writer.model_path.relative_to(
                            self.output_dir
                        ).as_posix(),
                        "prediction_path": writer.prediction_path.relative_to(
                            self.output_dir
                        ).as_posix(),
                    }
                    monitor.complete_fit(
                        training_summary=training_summary.to_dict(),
                        inference_seconds=inference_seconds,
                        evaluation_metrics=None,
                        prediction_rows=len(predictions),
                    )
                finally:
                    if model is not None:
                        model.detach_training_monitor()
            writer.finish(predictions, run_record)
        except Exception as error:
            writer.fail(
                str(error),
                locked_data_loaded=locked_data_loaded,
            )
            self.consolidate_artifacts()
            raise

        self.consolidate_artifacts()
        return run_record

    def _locked_split(self, selected: SelectedConfiguration) -> Any:
        """Load the one locked split matching the selected representation."""

        architecture = self.settings["architectures"][selected.model_family]
        representation = architecture["representation"]
        if representation in {"none", "tabular"}:
            feature_set = selected.feature_set or "age_only"
            return self._tabular().get_locked_outer_evaluation_split(
                selected.outer_fold,
                feature_set,
            )
        if representation == "sequence":
            if selected.lookback is None:
                raise LockedOuterEvaluationError(
                    f"Sequence family {selected.model_family!r} has no lookback"
                )
            return self._sequence().get_locked_outer_evaluation_split(
                selected.outer_fold,
                selected.lookback,
            )
        raise LockedOuterEvaluationError(
            f"Unsupported representation {representation!r}"
        )

    def _tabular(self) -> TabularDataAdapter:
        """Create the Step 2 adapter only after the Step 5 gate has opened."""

        if self._tabular_adapter is None:
            self._tabular_adapter = TabularDataAdapter()
        return self._tabular_adapter

    def _sequence(self) -> SequenceDataAdapter:
        """Create the Step 3 adapter only after the Step 5 gate has opened."""

        if self._sequence_adapter is None:
            self._sequence_adapter = SequenceDataAdapter()
        return self._sequence_adapter

    def _validate_locked_split(
        self,
        split: Any,
        outer_fold: int,
    ) -> dict[str, int]:
        """Verify dimensions, grouping, scenarios, labels, and training weights."""

        training = split.training
        validation = split.validation
        for name, dataset in (("training", training), ("validation", validation)):
            if len(dataset) == 0 or dataset.target is None:
                raise LockedOuterEvaluationError(
                    f"Locked {name} data is empty or lacks an RUL target"
                )
            if "uav_id" not in dataset.metadata.columns:
                raise LockedOuterEvaluationError(
                    f"Locked {name} metadata has no uav_id"
                )
        if training.sample_weights is None:
            raise LockedOuterEvaluationError(
                "Outer-training prefixes have no sample weights"
            )

        training_uav_values = training.metadata["uav_id"].astype(str)
        validation_uav_values = validation.metadata["uav_id"].astype(str)
        training_uavs = set(training_uav_values)
        validation_uavs = set(validation_uav_values)
        if training_uavs & validation_uavs:
            raise LockedOuterEvaluationError(
                "Outer-training and locked-validation UAVs overlap"
            )
        if "scenario" not in validation.metadata.columns:
            raise LockedOuterEvaluationError(
                "Locked validation metadata has no scenario column"
            )
        observed_scenarios = int(validation.metadata["scenario"].nunique())

        fold_count = len(self.plan.outer_fold_labels)
        if self.training_uavs % fold_count != 0:
            raise LockedOuterEvaluationError(
                "Training UAV count is not divisible by outer-fold count"
            )
        expected_validation_uavs = self.training_uavs // fold_count
        expected_training_uavs = self.training_uavs - expected_validation_uavs
        expected_training_rows = expected_training_uavs * self.prefixes_per_uav
        expected_validation_rows = (
            expected_validation_uavs * self.locked_scenarios
        )
        observed = {
            "training_rows": len(training),
            "training_uavs": len(training_uavs),
            "validation_rows": len(validation),
            "validation_uavs": len(validation_uavs),
            "locked_scenarios": observed_scenarios,
        }
        expected = {
            "training_rows": expected_training_rows,
            "training_uavs": expected_training_uavs,
            "validation_rows": expected_validation_rows,
            "validation_uavs": expected_validation_uavs,
            "locked_scenarios": self.locked_scenarios,
        }
        if observed != expected:
            raise LockedOuterEvaluationError(
                f"Locked split dimensions differ from the settings: {observed}"
            )

        metadata_outer_folds = set(
            validation.metadata["outer_fold"].astype(int)
        )
        if metadata_outer_folds != {outer_fold}:
            raise LockedOuterEvaluationError(
                "Locked validation rows do not match the requested outer fold"
            )
        if validation.metadata.duplicated(["scenario", "uav_id"]).any():
            raise LockedOuterEvaluationError(
                "Locked validation has duplicate scenario/UAV endpoints"
            )

        weights = np.asarray(training.sample_weights, dtype=np.float64)
        weight_table = pd.DataFrame(
            {"uav_id": training_uav_values, "weight": weights}
        )
        total_weights = weight_table.groupby("uav_id", sort=False)["weight"].sum()
        if not np.allclose(total_weights.to_numpy(), 1.0):
            raise LockedOuterEvaluationError(
                "Outer-training sample weights do not give every UAV total weight 1"
            )
        return observed

    def _prediction_table(
        self,
        validation: Any,
        selected: SelectedConfiguration,
        seed: int,
        predicted_rul: np.ndarray,
    ) -> pd.DataFrame:
        """Create and validate the common locked prediction schema."""

        required_metadata = {
            "scenario",
            "sample_id",
            "uav_id",
            "cutoff",
            "terminal_lifetime",
            "lifetime_quantile",
            "outer_fold",
        }
        missing = sorted(required_metadata - set(validation.metadata.columns))
        if missing:
            raise LockedOuterEvaluationError(
                f"Locked prediction metadata is missing columns {missing}"
            )
        observed_rul = target_values(validation)
        predicted = np.asarray(predicted_rul, dtype=np.float64).reshape(-1)
        if len(predicted) != len(validation) or not np.isfinite(predicted).all():
            raise LockedOuterEvaluationError(
                "Locked predictions have an invalid length or non-finite values"
            )

        table = validation.metadata.copy()
        table.insert(0, "seed", seed)
        table.insert(0, "configuration_id", selected.configuration_id)
        table.insert(0, "model_family", selected.model_family)
        table.insert(0, "settings_version", self.settings_version)
        table["feature_set"] = selected.feature_set
        table["lookback"] = selected.lookback
        table["y_true"] = observed_rul
        table["y_pred"] = predicted
        table["residual"] = predicted - observed_rul
        table = table.loc[:, PREDICTION_COLUMNS]
        if table.duplicated(
            ["model_family", "seed", "scenario", "uav_id"]
        ).any():
            raise LockedOuterEvaluationError(
                "Locked prediction table contains duplicate evaluation keys"
            )
        return table

    def _expected_run_keys(self) -> dict[tuple[str, int, int], str]:
        """Map every required family/fold/seed run to its configuration ID."""

        return {
            (family, outer_fold, seed): self.plan.configuration(
                family,
                outer_fold,
            ).configuration_id
            for family in self.plan.enabled_families
            for outer_fold in self.plan.outer_fold_labels
            for seed in self.plan.seeds_for(family)
        }

    def consolidate_artifacts(self) -> dict[str, Any]:
        """Combine only complete runs and expose partial status explicitly."""

        run_dir = self.output_dir / "runs"
        prediction_records: list[dict[str, Any]] = []
        run_records: list[dict[str, Any]] = []
        completed_runs: list[str] = []
        incomplete_runs: list[str] = []
        expected_keys = self._expected_run_keys()
        any_locked_data_loaded = False

        for (family, outer_fold, seed), configuration_id in expected_keys.items():
            prefix = f"{family}__outer_{outer_fold:02d}__seed_{seed:03d}"
            status_path = run_dir / f"{prefix}__status.json"
            status = self._read_json_if_present(status_path)
            if status is not None and status.get("locked_data_loaded") is True:
                any_locked_data_loaded = True
            valid = (
                status is not None
                and status.get("state") == "complete"
                and status.get("runner_version") == RUNNER_VERSION
                and status.get("settings_version") == self.settings_version
                and status.get("configuration_id") == configuration_id
                and status.get("seed") == seed
            )
            if not valid:
                incomplete_runs.append(prefix)
                continue

            prediction_path = run_dir / f"{prefix}__predictions.csv.gz"
            record_path = run_dir / f"{prefix}__run.json"
            try:
                predictions = pd.read_csv(prediction_path)
            except (OSError, pd.errors.ParserError) as error:
                raise LockedOuterEvaluationError(
                    f"Cannot read completed predictions {prediction_path}: {error}"
                ) from error
            missing_columns = sorted(
                set(PREDICTION_COLUMNS) - set(predictions.columns)
            )
            if missing_columns:
                raise LockedOuterEvaluationError(
                    f"Completed predictions are missing columns {missing_columns}"
                )
            expected_rows = int(status.get("prediction_rows", -1))
            if len(predictions) != expected_rows:
                raise LockedOuterEvaluationError(
                    f"Completed run {prefix} has inconsistent prediction rows"
                )
            record = self._read_json_if_present(record_path)
            if record is None:
                raise LockedOuterEvaluationError(
                    f"Completed run {prefix} has no run record"
                )
            prediction_records.extend(
                predictions.loc[:, PREDICTION_COLUMNS].to_dict("records")
            )
            run_records.append(record)
            completed_runs.append(prefix)

        # Phase 1 labels locked scenarios ``locked_01`` ... ``locked_NN``
        # (1_dataset_construction/3_test_like_validation_scenarios), so
        # ``scenario`` is text and must not be coerced to an integer. The
        # labels are zero padded, so sorting them as text still orders the
        # rows by scenario number.
        prediction_records.sort(
            key=lambda row: (
                row["model_family"],
                int(row["seed"]),
                int(row["outer_fold"]),
                str(row["scenario"]),
                str(row["uav_id"]),
            )
        )
        run_records.sort(
            key=lambda row: (
                row["model_family"],
                int(row["seed"]),
                int(row["outer_fold"]),
            )
        )
        _write_csv(
            prediction_records,
            PREDICTION_COLUMNS,
            self.output_dir / "locked_predictions.csv.gz",
            compressed=True,
        )
        _write_csv(
            run_records,
            RUN_COLUMNS,
            self.output_dir / "model_runs.csv",
        )

        expected_prediction_rows = sum(
            self._expected_validation_rows()
            for _ in expected_keys
        )
        manifest = {
            "runner_version": RUNNER_VERSION,
            "settings_version": self.settings_version,
            "status": (
                "complete"
                if len(completed_runs) == len(expected_keys)
                else "partial"
            ),
            "step_5_prerequisite": "complete",
            "locked_results_used_for_tuning": False,
            "fixed_training_duration_from_step_5": True,
            "enabled_families": list(self.plan.enabled_families),
            "outer_fold_labels": list(self.plan.outer_fold_labels),
            "retraining_seeds": list(self.plan.retraining_seeds),
            "completed_runs": completed_runs,
            "incomplete_runs": incomplete_runs,
            "expected_run_count": len(expected_keys),
            "completed_run_count": len(completed_runs),
            "expected_prediction_rows": expected_prediction_rows,
            "prediction_rows": len(prediction_records),
            "locked_data_loaded": any_locked_data_loaded,
            "test_data_loaded": False,
            "libraries": {
                "numpy": version("numpy"),
                "pandas": version("pandas"),
            },
            "artifacts": {
                "locked_predictions": "locked_predictions.csv.gz",
                "model_runs": "model_runs.csv",
                "models": "models/",
                "run_checkpoints": "runs/",
            },
        }
        _write_json(
            manifest,
            self.output_dir / "locked_evaluation_manifest.json",
        )
        return manifest

    def _expected_validation_rows(self) -> int:
        """Return the locked endpoints expected in each individual model run."""

        validation_uavs = self.training_uavs // len(self.plan.outer_fold_labels)
        return validation_uavs * self.locked_scenarios

    @staticmethod
    def _read_json_if_present(path: Path) -> dict[str, Any] | None:
        """Read one optional generated object and reject malformed content."""

        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise LockedOuterEvaluationError(
                f"Cannot read generated JSON {path}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise LockedOuterEvaluationError(
                f"Generated JSON {path} is not an object"
            )
        return value
