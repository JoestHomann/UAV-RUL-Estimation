"""Tune the selected Phase 2 family across all five development folds."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack
import gc
from importlib.metadata import version
import json
import math
from pathlib import Path
from time import perf_counter
import sys
from typing import Any, Callable

import numpy as np
import optuna
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial, TrialState
import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
PHASE_DIR = STEP_DIR.parent
REPOSITORY_ROOT = PHASE_DIR.parent
PHASE_2_DIR = REPOSITORY_ROOT / "2_model_architecture_study"
DEPENDENCY_DIRS = [
    PHASE_DIR,
    PHASE_2_DIR,
    PHASE_2_DIR / "2_tabular_data_adapter",
    PHASE_2_DIR / "3_sequence_data_adapter",
    PHASE_2_DIR / "3_trajectory_data_adapter",
    PHASE_2_DIR / "4_model_adapters",
    PHASE_2_DIR / "5_inner_model_selection",
]
for dependency_dir in DEPENDENCY_DIRS:
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from base import target_values  # noqa: E402
from candidate_space import CandidateSpace, ResolvedCandidate  # noqa: E402
from model_registry import ModelAdapterFactory  # noqa: E402
from sequence_data_adapter import SequenceDataAdapter  # noqa: E402
from tabular_data_adapter import TabularDataAdapter  # noqa: E402
from trajectory_data_adapter import TrajectoryDataAdapter  # noqa: E402
from tensorboard_monitoring import (  # noqa: E402
    TrainingRunContext,
    calculate_regression_metrics,
    create_study_monitor,
    log_step_5_candidate,
)
from phase_3_common import (  # noqa: E402
    PHASE_2_SPECIFICATION_PATH,
    Phase3Error,
    complete_manifest,
    configured_repository_path,
    invalidate_downstream_manifests,
    load_resolved_phase_3_settings,
    read_json,
    selected_architecture_path,
    step_directory,
    write_csv,
    write_json,
)
from phase_3_run_layout import tensorboard_log_root  # noqa: E402


SEARCH_VERSION = 2
EARLY_STOPPED_FAMILIES = {
    "xgboost",
    "catboost",
    "mlp",
    "tcn",
    "multiscale_cnn",
    "sensor_graph_tcn",
    "lstm",
    "transformer",
}

CANDIDATE_COLUMNS = [
    "settings_version",
    "phase_3_run_number",
    "model_family",
    "candidate_number",
    "configuration_id",
    "optuna_trial_number",
    "feature_set",
    "lookback",
    "model_seed",
    "mean_fold_rmse",
    "fold_rmse_standard_deviation",
    "mean_fold_r2",
    "mean_fold_mae",
    "mean_fold_bias",
    "mean_fold_overprediction_rate",
    "mean_fold_mean_overprediction",
    "mean_fold_root_mean_squared_overprediction",
    "mean_fold_underprediction_rate",
    "mean_fold_mean_underprediction",
    "mean_training_seconds",
    "total_training_seconds",
    "mean_inference_seconds",
    "final_training_iterations",
    "trainable_parameters",
    "configuration_json",
    "hyperparameters_json",
    "sampler_parameters_json",
    "selected",
]

FOLD_COLUMNS = [
    "settings_version",
    "phase_3_run_number",
    "model_family",
    "candidate_number",
    "configuration_id",
    "optuna_trial_number",
    "outer_fold",
    "feature_set",
    "lookback",
    "model_seed",
    "rmse",
    "r2",
    "mae",
    "bias",
    "overprediction_rate",
    "mean_overprediction",
    "root_mean_squared_overprediction",
    "overprediction_q90",
    "overprediction_q95",
    "maximum_overprediction",
    "underprediction_rate",
    "mean_underprediction",
    "root_mean_squared_underprediction",
    "training_rows",
    "training_uavs",
    "validation_rows",
    "validation_uavs",
    "validation_scenarios",
    "training_seconds",
    "inference_seconds",
    "epochs_or_iterations",
    "best_epoch_or_iteration",
    "trainable_parameters",
]

OOF_COLUMNS = [
    "settings_version",
    "phase_3_run_number",
    "model_family",
    "candidate_number",
    "configuration_id",
    "optuna_trial_number",
    "outer_fold",
    "validation_row",
    "uav_id",
    "scenario",
    "cutoff",
    "observed_rul",
    "predicted_rul",
    "residual",
]


class FinalConfigurationSearchError(Phase3Error):
    """Represent an invalid split, candidate, checkpoint, or search result."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _scenario_label(value: Any) -> str:
    """Preserve numeric and named scenario identifiers as non-empty text."""

    if value is None or pd.isna(value):
        raise FinalConfigurationSearchError(
            "Development validation metadata contain a missing scenario"
        )
    label = str(value).strip()
    if not label:
        raise FinalConfigurationSearchError(
            "Development validation metadata contain an empty scenario"
        )
    return label


class FinalSearchSplitRepository:
    """Cache only training-prefix and development-scenario fold views."""

    def __init__(
        self,
        expected_scenarios: int,
        *,
        tabular_manifest_path: Path | None = None,
        sequence_manifest_path: Path | None = None,
        trajectory_manifest_path: Path | None = None,
    ) -> None:
        self.expected_scenarios = expected_scenarios
        self._tabular_manifest_path = tabular_manifest_path
        self._sequence_manifest_path = sequence_manifest_path
        self._trajectory_manifest_path = trajectory_manifest_path
        self._tabular_adapter: TabularDataAdapter | None = None
        self._sequence_adapter: SequenceDataAdapter | None = None
        self._trajectory_adapter: TrajectoryDataAdapter | None = None
        self._tabular_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._sequence_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._trajectory_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

    def _tabular(self) -> TabularDataAdapter:
        if self._tabular_adapter is None:
            self._tabular_adapter = (
                TabularDataAdapter(self._tabular_manifest_path)
                if self._tabular_manifest_path is not None
                else TabularDataAdapter()
            )
        return self._tabular_adapter

    def _sequence(self) -> SequenceDataAdapter:
        if self._sequence_adapter is None:
            self._sequence_adapter = (
                SequenceDataAdapter(self._sequence_manifest_path)
                if self._sequence_manifest_path is not None
                else SequenceDataAdapter()
            )
        return self._sequence_adapter

    def _trajectory(self) -> TrajectoryDataAdapter:
        if self._trajectory_adapter is None:
            self._trajectory_adapter = (
                TrajectoryDataAdapter(self._trajectory_manifest_path)
                if self._trajectory_manifest_path is not None
                else TrajectoryDataAdapter()
            )
        return self._trajectory_adapter

    @staticmethod
    def _cached(
        cache: OrderedDict[tuple[Any, ...], Any],
        key: tuple[Any, ...],
        maximum_entries: int,
        loader: Callable[[], Any],
    ) -> Any:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        value = loader()
        cache[key] = value
        while len(cache) > maximum_entries:
            cache.popitem(last=False)
        return value

    def outer_fold_labels(self, representation: str) -> tuple[int, ...]:
        if representation == "sequence":
            return self._sequence().outer_fold_labels()
        if representation == "trajectory":
            return self._trajectory().outer_fold_labels()
        return self._tabular().outer_fold_labels()

    def get(
        self,
        *,
        family: str,
        representation: str,
        candidate: ResolvedCandidate,
        outer_fold: int,
    ) -> Any:
        """Return a Phase 3 development split with no locked/test accessor."""

        if representation in {"none", "tabular"}:
            feature_set = candidate.feature_set or "age_only"
            key = (outer_fold, feature_set)
            split = self._cached(
                self._tabular_cache,
                key,
                15,
                lambda: self._tabular().get_final_search_split(
                    outer_fold,
                    feature_set,
                ),
            )
        elif representation == "sequence":
            if candidate.lookback is None:
                raise FinalConfigurationSearchError(
                    f"Sequence family {family!r} has no lookback"
                )
            key = (outer_fold, candidate.lookback)
            split = self._cached(
                self._sequence_cache,
                key,
                5,
                lambda: self._sequence().get_final_search_split(
                    outer_fold,
                    candidate.lookback,
                ),
            )
        elif representation == "trajectory":
            key = (outer_fold,)
            split = self._cached(
                self._trajectory_cache,
                key,
                5,
                lambda: self._trajectory().get_final_search_split(outer_fold),
            )
        else:
            raise FinalConfigurationSearchError(
                f"Unsupported representation {representation!r}"
            )
        self._validate(split, outer_fold)
        return split

    def _validate(self, split: Any, outer_fold: int) -> None:
        training = split.training
        validation = split.validation
        if len(training) == 0 or len(validation) == 0:
            raise FinalConfigurationSearchError(
                f"Outer fold {outer_fold} produced an empty dataset"
            )
        if training.target is None or validation.target is None:
            raise FinalConfigurationSearchError("Final-search data have no RUL target")
        if training.sample_weights is None:
            raise FinalConfigurationSearchError("Training prefixes have no weights")
        training_ids = set(training.metadata["uav_id"].astype(str))
        validation_ids = set(validation.metadata["uav_id"].astype(str))
        overlap = sorted(training_ids & validation_ids)
        if overlap:
            raise FinalConfigurationSearchError(
                f"Outer fold {outer_fold} overlaps UAVs: {overlap[:5]}"
            )
        if len(training_ids) != 80 or len(validation_ids) != 20:
            raise FinalConfigurationSearchError(
                f"Outer fold {outer_fold} must contain 80/20 UAVs, found "
                f"{len(training_ids)}/{len(validation_ids)}"
            )
        training_uav_ids = training.metadata["uav_id"].astype(str)
        weight_sums = training.sample_weights.groupby(training_uav_ids).sum()
        if not np.allclose(
            weight_sums.to_numpy(dtype=float),
            np.ones(len(weight_sums), dtype=float),
            rtol=0,
            atol=1e-12,
        ):
            raise FinalConfigurationSearchError(
                f"Outer fold {outer_fold} does not weight training UAVs equally"
            )
        if "scenario" not in validation.metadata.columns:
            raise FinalConfigurationSearchError(
                "Development validation metadata have no scenario"
            )
        scenarios = int(validation.metadata["scenario"].nunique())
        if scenarios != self.expected_scenarios:
            raise FinalConfigurationSearchError(
                f"Expected {self.expected_scenarios} scenarios, found {scenarios}"
            )
        scenarios_per_uav = validation.metadata.groupby(
            validation.metadata["uav_id"].astype(str)
        )["scenario"].nunique()
        if not (scenarios_per_uav == self.expected_scenarios).all():
            raise FinalConfigurationSearchError(
                f"Outer fold {outer_fold} does not contain every scenario per UAV"
            )


class FinalConfigurationSearchRunner:
    """Run and resume the one selected-family Optuna study."""

    def __init__(self, run_number: int, output_dir: Path | None = None) -> None:
        self.run_number = run_number
        self.settings = load_resolved_phase_3_settings(run_number)
        self.settings_version = int(self.settings["settings_version"])
        if complete_manifest(1, run_number, self.settings_version) is None:
            raise FinalConfigurationSearchError("Step 1 is not complete")
        self.selection = read_json(
            selected_architecture_path(run_number),
            "selected architecture",
        )
        self.family = str(self.selection["selected_model_family"])
        self.representation = str(self.selection["representation"])
        self.phase_2_specification = read_json(
            configured_repository_path(
                self.settings,
                "phase_2_specification",
                PHASE_2_SPECIFICATION_PATH,
            ),
            "Phase 2 experiment specification",
        )
        self.phase_2_settings = self.phase_2_specification["settings"]
        self.architectures = self.phase_2_settings["architectures"]
        self.candidate_space = CandidateSpace(self.architectures)
        self.maximum_budget = int(self.settings["final_search"]["candidate_budget"])
        self.candidate_budget = self.candidate_space.candidate_budget(
            self.family,
            self.maximum_budget,
        )
        self.search_seed = int(self.settings["final_search"]["search_seed"])
        self.model_seed = int(self.settings["final_search"]["model_seed"])
        self.expected_scenarios = int(
            self.phase_2_settings["phase_1"]["expected_development_scenarios"]
        )
        self.output_dir = (
            step_directory(2, run_number=run_number)
            if output_dir is None
            else output_dir.resolve()
        )
        self.artifact_dir = self.output_dir / "artifacts"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.storage_path = self.checkpoint_dir / "final_search.sqlite3"
        self.status_path = self.output_dir / "final_search_status.json"
        self.manifest_output_path = self.output_dir / "final_search_manifest.json"
        self.selected_output_path = self.artifact_dir / "selected_configuration.json"
        self.log_root = (
            tensorboard_log_root(self.run_number)
            if output_dir is None
            else self.output_dir / "tensorboard_logs"
        )
        self.phase_2_specification_path = configured_repository_path(
            self.settings,
            "phase_2_specification",
            PHASE_2_SPECIFICATION_PATH,
        )
        self.tabular_manifest_path = configured_repository_path(
            self.settings,
            "tabular_manifest",
            PHASE_2_DIR
            / "2_tabular_data_adapter"
            / "artifacts"
            / "tabular_dataset_manifest.json",
        )
        self.sequence_manifest_path = configured_repository_path(
            self.settings,
            "sequence_manifest",
            PHASE_2_DIR
            / "3_sequence_data_adapter"
            / "artifacts"
            / "sequence_dataset_manifest.json",
        )
        self.trajectory_manifest_path = configured_repository_path(
            self.settings,
            "trajectory_manifest",
            PHASE_2_DIR
            / "3_trajectory_data_adapter"
            / "artifacts"
            / "trajectory_dataset_manifest.json",
        )
        self.factory = ModelAdapterFactory(self.phase_2_specification_path)

    def _clear_output(self) -> None:
        for path in (
            self.artifact_dir / "final_search_candidate_results.csv",
            self.artifact_dir / "final_search_fold_results.csv",
            self.artifact_dir / "final_search_oof_predictions.csv",
            self.selected_output_path,
            self.manifest_output_path,
            self.status_path,
        ):
            path.unlink(missing_ok=True)
        if self.checkpoint_dir.is_dir():
            for path in self.checkpoint_dir.glob("final_search.sqlite3*"):
                if path.is_file():
                    path.unlink()

    def _study(self) -> Study:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{self.storage_path.resolve().as_posix()}"
        return optuna.create_study(
            study_name=(
                f"phase_3_run_{self.run_number}__settings_{self.settings_version}__"
                f"{self.family}"
            ),
            storage=storage_url,
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.search_seed),
            pruner=optuna.pruners.NopPruner(),
        )

    @staticmethod
    def _records(
        study: Study,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        complete = [
            trial
            for trial in study.trials
            if trial.state == TrialState.COMPLETE
            and isinstance(trial.user_attrs.get("candidate_record"), dict)
        ]
        complete.sort(key=lambda trial: int(trial.user_attrs["candidate_number"]))
        candidates = [dict(trial.user_attrs["candidate_record"]) for trial in complete]
        folds = [
            dict(record)
            for trial in complete
            for record in trial.user_attrs["fold_records"]
        ]
        oof = [
            dict(record)
            for trial in complete
            for record in trial.user_attrs.get("oof_records", [])
        ]
        return candidates, folds, oof

    def _write_status(
        self,
        state: str,
        completed_candidates: int,
        error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "search_version": SEARCH_VERSION,
            "settings_version": self.settings_version,
            "phase_3_run_number": self.run_number,
            "model_family": self.family,
            "state": state,
            "candidate_budget": self.candidate_budget,
            "completed_candidates": completed_candidates,
        }
        if error:
            payload["error"] = error
        write_json(payload, self.status_path)

    def _checkpoint(self, study: Study, _: FrozenTrial | None = None) -> None:
        candidates, folds, oof = self._records(study)
        write_csv(
            candidates,
            CANDIDATE_COLUMNS,
            self.artifact_dir / "final_search_candidate_results.csv",
        )
        write_csv(
            folds,
            FOLD_COLUMNS,
            self.artifact_dir / "final_search_fold_results.csv",
        )
        write_csv(
            oof,
            OOF_COLUMNS,
            self.artifact_dir / "final_search_oof_predictions.csv",
        )
        self._write_status("running", len(candidates))

    @staticmethod
    def _training_iterations(
        family: str,
        fold_records: list[dict[str, Any]],
    ) -> int | None:
        if family not in EARLY_STOPPED_FAMILIES:
            return None
        values = [record["best_epoch_or_iteration"] for record in fold_records]
        if any(value is None for value in values):
            raise FinalConfigurationSearchError(
                f"Early-stopped family {family!r} has missing best durations"
            )
        median = float(np.median(np.asarray(values, dtype=np.float64)))
        return max(1, int(math.floor(median + 0.5)))

    def _evaluate_candidate(
        self,
        *,
        trial: Trial,
        candidate: ResolvedCandidate,
        candidate_number: int,
        folds: tuple[int, ...],
        split_repository: FinalSearchSplitRepository,
        monitors: dict[int, Any],
    ) -> tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        configuration_id = f"{self.family}__candidate_{candidate_number:03d}"
        fold_records: list[dict[str, Any]] = []
        oof_records: list[dict[str, Any]] = []
        for outer_fold in folds:
            split = split_repository.get(
                family=self.family,
                representation=self.representation,
                candidate=candidate,
                outer_fold=outer_fold,
            )
            context = TrainingRunContext(
                stage="step_5",
                model_family=self.family,
                representation=self.representation,
                outer_fold=outer_fold,
                seed=self.model_seed,
                configuration_id=configuration_id,
                candidate_number=candidate_number,
                inner_fold=outer_fold,
                feature_set=candidate.feature_set,
                lookback=candidate.lookback,
            )
            model = None
            with monitors[outer_fold].fit(context) as monitor:
                model = self.factory.create(
                    self.family,
                    candidate.hyperparameters,
                    seed=self.model_seed,
                    training_monitor=monitor,
                )
                try:
                    summary = model.fit(split.training, split.validation)
                    started = perf_counter()
                    predictions = model.predict(split.validation)
                    inference_seconds = perf_counter() - started
                    metrics = calculate_regression_metrics(
                        target_values(split.validation),
                        predictions,
                    )
                    if not all(np.isfinite(value) for value in metrics.values()):
                        raise FinalConfigurationSearchError(
                            f"{configuration_id} produced non-finite metrics"
                        )
                    if summary.best_validation_rmse is None or not np.isclose(
                        summary.best_validation_rmse,
                        metrics["rmse"],
                        rtol=1e-10,
                        atol=1e-10,
                    ):
                        raise FinalConfigurationSearchError(
                            "Adapter and final-search RMSE calculations disagree"
                        )
                    targets = target_values(split.validation)
                    metadata = split.validation.metadata.reset_index(drop=True)
                    for validation_row, (target, prediction) in enumerate(
                        zip(targets, predictions, strict=True)
                    ):
                        oof_records.append(
                            {
                                "settings_version": self.settings_version,
                                "phase_3_run_number": self.run_number,
                                "model_family": self.family,
                                "candidate_number": candidate_number,
                                "configuration_id": configuration_id,
                                "optuna_trial_number": trial.number,
                                "outer_fold": outer_fold,
                                "validation_row": validation_row,
                                "uav_id": str(metadata.loc[validation_row, "uav_id"]),
                                "scenario": _scenario_label(
                                    metadata.loc[validation_row, "scenario"]
                                ),
                                "cutoff": float(metadata.loc[validation_row, "cutoff"]),
                                "observed_rul": float(target),
                                "predicted_rul": float(prediction),
                                "residual": float(prediction - target),
                            }
                        )
                finally:
                    if model is not None:
                        model.detach_training_monitor()

            fold_records.append(
                {
                    "settings_version": self.settings_version,
                    "phase_3_run_number": self.run_number,
                    "model_family": self.family,
                    "candidate_number": candidate_number,
                    "configuration_id": configuration_id,
                    "optuna_trial_number": trial.number,
                    "outer_fold": outer_fold,
                    "feature_set": candidate.feature_set,
                    "lookback": candidate.lookback,
                    "model_seed": self.model_seed,
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "mae": metrics["mae"],
                    "bias": metrics["bias"],
                    **{
                        name: metrics[name]
                        for name in (
                            "overprediction_rate",
                            "mean_overprediction",
                            "root_mean_squared_overprediction",
                            "overprediction_q90",
                            "overprediction_q95",
                            "maximum_overprediction",
                            "underprediction_rate",
                            "mean_underprediction",
                            "root_mean_squared_underprediction",
                        )
                    },
                    "training_rows": len(split.training),
                    "training_uavs": int(
                        split.training.metadata["uav_id"].nunique()
                    ),
                    "validation_rows": len(split.validation),
                    "validation_uavs": int(
                        split.validation.metadata["uav_id"].nunique()
                    ),
                    "validation_scenarios": int(
                        split.validation.metadata["scenario"].nunique()
                    ),
                    "training_seconds": summary.training_seconds,
                    "inference_seconds": inference_seconds,
                    "epochs_or_iterations": summary.epochs_or_iterations,
                    "best_epoch_or_iteration": summary.best_epoch_or_iteration,
                    "trainable_parameters": summary.trainable_parameters,
                }
            )
            del model
            gc.collect()

        def values(name: str) -> np.ndarray:
            return np.asarray([record[name] for record in fold_records], dtype=float)

        candidate_record = {
            "settings_version": self.settings_version,
            "phase_3_run_number": self.run_number,
            "model_family": self.family,
            "candidate_number": candidate_number,
            "configuration_id": configuration_id,
            "optuna_trial_number": trial.number,
            "feature_set": candidate.feature_set,
            "lookback": candidate.lookback,
            "model_seed": self.model_seed,
            "mean_fold_rmse": float(np.mean(values("rmse"))),
            "fold_rmse_standard_deviation": float(np.std(values("rmse"))),
            "mean_fold_r2": float(np.mean(values("r2"))),
            "mean_fold_mae": float(np.mean(values("mae"))),
            "mean_fold_bias": float(np.mean(values("bias"))),
            "mean_fold_overprediction_rate": float(
                np.mean(values("overprediction_rate"))
            ),
            "mean_fold_mean_overprediction": float(
                np.mean(values("mean_overprediction"))
            ),
            "mean_fold_root_mean_squared_overprediction": float(
                np.mean(values("root_mean_squared_overprediction"))
            ),
            "mean_fold_underprediction_rate": float(
                np.mean(values("underprediction_rate"))
            ),
            "mean_fold_mean_underprediction": float(
                np.mean(values("mean_underprediction"))
            ),
            "mean_training_seconds": float(np.mean(values("training_seconds"))),
            "total_training_seconds": float(np.sum(values("training_seconds"))),
            "mean_inference_seconds": float(np.mean(values("inference_seconds"))),
            "final_training_iterations": self._training_iterations(
                self.family,
                fold_records,
            ),
            "trainable_parameters": fold_records[0]["trainable_parameters"],
            "configuration_json": candidate.canonical_json(),
            "hyperparameters_json": _stable_json(candidate.hyperparameters),
            "sampler_parameters_json": _stable_json(trial.params),
            "selected": False,
        }
        log_step_5_candidate(
            model_family=self.family,
            outer_fold=0,
            candidate_number=candidate_number,
            mean_inner_rmse=candidate_record["mean_fold_rmse"],
            hyperparameters=candidate.hyperparameters,
            log_root=self.log_root,
        )
        return candidate_record, fold_records, oof_records

    def _finish(self, study: Study, folds: tuple[int, ...]) -> dict[str, Any]:
        candidates, fold_records, oof_records = self._records(study)
        if len(candidates) != self.candidate_budget:
            raise FinalConfigurationSearchError(
                f"Completed {len(candidates)} of {self.candidate_budget} candidates"
            )
        selected = min(
            candidates,
            key=lambda record: (
                float(record["mean_fold_rmse"]),
                int(record["candidate_number"]),
            ),
        )
        for candidate in candidates:
            candidate["selected"] = (
                candidate["configuration_id"] == selected["configuration_id"]
            )
        write_csv(
            candidates,
            CANDIDATE_COLUMNS,
            self.artifact_dir / "final_search_candidate_results.csv",
        )
        write_csv(
            fold_records,
            FOLD_COLUMNS,
            self.artifact_dir / "final_search_fold_results.csv",
        )
        write_csv(
            oof_records,
            OOF_COLUMNS,
            self.artifact_dir / "final_search_oof_predictions.csv",
        )
        configuration = json.loads(str(selected["configuration_json"]))
        selected_payload = {
            "selection_version": 1,
            "settings_version": self.settings_version,
            "phase_2_run_number": self.settings["phase_2_run_number"],
            "phase_3_run_number": self.run_number,
            "model_family": self.family,
            "representation": self.representation,
            "configuration_id": selected["configuration_id"],
            "candidate_number": int(selected["candidate_number"]),
            "feature_set": configuration["feature_set"],
            "lookback": configuration["lookback"],
            "hyperparameters": configuration["hyperparameters"],
            "mean_fold_rmse": float(selected["mean_fold_rmse"]),
            "fold_rmse_standard_deviation": float(
                selected["fold_rmse_standard_deviation"]
            ),
            "final_training_iterations": (
                None
                if pd.isna(selected["final_training_iterations"])
                else int(selected["final_training_iterations"])
            ),
            "search_seed": self.search_seed,
            "model_seed": self.model_seed,
            "prediction_minimum": float(
                self.phase_2_settings["evaluation"]["prediction_minimum"]
            ),
            "selection_metric": "mean_fold_rmse",
            "selection_direction": "minimize",
            "tie_breaker": "candidate_number",
            "locked_data_loaded": False,
            "test_data_loaded": False,
        }
        write_json(selected_payload, self.selected_output_path)
        manifest = {
            "search_version": SEARCH_VERSION,
            "settings_version": self.settings_version,
            "phase_2_run_number": self.settings["phase_2_run_number"],
            "phase_3_run_number": self.run_number,
            "status": "complete",
            "model_family": self.family,
            "candidate_budget": self.candidate_budget,
            "completed_candidate_count": len(candidates),
            "outer_fold_labels": list(folds),
            "folds_per_candidate": len(folds),
            "development_scenarios": self.expected_scenarios,
            "sampler": "Optuna TPESampler",
            "search_seed": self.search_seed,
            "model_seed": self.model_seed,
            "selection_metric": "mean_fold_rmse",
            "selection_direction": "minimize",
            "tie_breaker": "candidate_number",
            "locked_data_loaded": False,
            "test_data_loaded": False,
            "libraries": {
                "numpy": version("numpy"),
                "optuna": version("optuna"),
                "pandas": version("pandas"),
            },
            "artifacts": {
                "candidate_results": "artifacts/final_search_candidate_results.csv",
                "fold_results": "artifacts/final_search_fold_results.csv",
                "oof_predictions": "artifacts/final_search_oof_predictions.csv",
                "selected_configuration": "artifacts/selected_configuration.json",
                "study_checkpoint": "checkpoints/final_search.sqlite3",
            },
        }
        write_json(manifest, self.manifest_output_path)
        self._write_status("complete", len(candidates))
        return selected_payload

    def run(self, *, force: bool = False) -> dict[str, Any]:
        if not force:
            complete = None
            if self.manifest_output_path.is_file():
                complete = read_json(
                    self.manifest_output_path,
                    "final-search manifest",
                )
            if (
                complete is not None
                and complete.get("status") == "complete"
                and complete.get("settings_version") == self.settings_version
                and complete.get("phase_3_run_number") == self.run_number
            ):
                return read_json(
                    self.selected_output_path,
                    "selected final configuration",
                )
        else:
            invalidate_downstream_manifests(2, self.run_number)
            self._clear_output()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = self._study()
        split_repository = FinalSearchSplitRepository(
            self.expected_scenarios,
            tabular_manifest_path=self.tabular_manifest_path,
            sequence_manifest_path=self.sequence_manifest_path,
            trajectory_manifest_path=self.trajectory_manifest_path,
        )
        folds = split_repository.outer_fold_labels(self.representation)
        if len(folds) != 5:
            raise FinalConfigurationSearchError(
                f"Expected five outer folds, observed {len(folds)}"
            )
        existing_candidates, _, _ = self._records(study)
        seen = {str(record["configuration_json"]) for record in existing_candidates}
        if len(existing_candidates) > self.candidate_budget:
            raise FinalConfigurationSearchError(
                "Checkpoint contains more candidates than the declared budget"
            )
        self._checkpoint(study)

        try:
            with ExitStack() as stack:
                monitors = {
                    fold: stack.enter_context(
                        create_study_monitor(
                            stage="step_5",
                            model_family=self.family,
                            outer_fold=fold,
                            log_root=self.log_root,
                        )
                    )
                    for fold in folds
                }

                def objective(trial: Trial) -> float:
                    candidate = self.candidate_space.resolve(self.family, trial)
                    canonical = candidate.canonical_json()
                    if canonical in seen:
                        raise optuna.TrialPruned("Duplicate resolved candidate")
                    candidates, _, _ = self._records(study)
                    next_number = max(
                        [int(record["candidate_number"]) for record in candidates],
                        default=0,
                    ) + 1
                    candidate_record, fold_records, oof_records = self._evaluate_candidate(
                        trial=trial,
                        candidate=candidate,
                        candidate_number=next_number,
                        folds=folds,
                        split_repository=split_repository,
                        monitors=monitors,
                    )
                    trial.set_user_attr("candidate_number", next_number)
                    trial.set_user_attr("candidate_record", candidate_record)
                    trial.set_user_attr("fold_records", fold_records)
                    trial.set_user_attr("oof_records", oof_records)
                    trial.set_user_attr("configuration_json", canonical)
                    seen.add(canonical)
                    return float(candidate_record["mean_fold_rmse"])

                maximum_attempts = self.candidate_budget * 20
                attempts = 0
                while len(self._records(study)[0]) < self.candidate_budget:
                    if attempts >= maximum_attempts:
                        raise FinalConfigurationSearchError(
                            f"Could not generate {self.candidate_budget} distinct candidates"
                        )
                    study.optimize(
                        objective,
                        n_trials=1,
                        n_jobs=1,
                        callbacks=[self._checkpoint],
                        gc_after_trial=True,
                        show_progress_bar=False,
                    )
                    attempts += 1
            return self._finish(study, folds)
        except (KeyboardInterrupt, SystemExit) as error:
            completed = len(self._records(study)[0])
            self._checkpoint(study)
            self._write_status(
                "interrupted",
                completed,
                str(error) or type(error).__name__,
            )
            raise
        except Exception as error:
            completed = len(self._records(study)[0])
            self._checkpoint(study)
            self._write_status("failed", completed, str(error))
            raise
