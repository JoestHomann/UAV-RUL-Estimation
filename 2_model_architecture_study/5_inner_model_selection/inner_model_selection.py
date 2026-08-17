"""Run leakage-safe automatic model selection within each architecture.

Every study is limited to one model family and one outer fold. Candidate
configurations are evaluated through the four inner UAV folds using only the
five development scenarios. Locked validation data is never loaded here.

The module writes detailed fold, candidate, and selected-configuration tables.
It deliberately does not compare families or declare an architecture winner.
"""

from __future__ import annotations

from collections import OrderedDict
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
DEFAULT_OUTPUT_DIR = STEP_DIR / "artifacts"
DEFAULT_SPECIFICATION_PATH = (
    PHASE_DIR
    / "1_architecture_study_settings"
    / "artifacts"
    / "experiment_specification.json"
)

# The numbered step folders are intentionally not Python package names. Add
# their exact locations once so this runner can reuse their public interfaces.
DEPENDENCY_DIRS = [
    PHASE_DIR,
    PHASE_DIR / "2_tabular_data_adapter",
    PHASE_DIR / "3_sequence_data_adapter",
    PHASE_DIR / "4_model_adapters",
]
for dependency_dir in DEPENDENCY_DIRS:
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from base import (  # noqa: E402
    ModelAdapterError,
    target_values,
)
from model_registry import (  # noqa: E402
    ModelAdapterFactory,
    load_experiment_specification,
)
from sequence_data_adapter import SequenceDataAdapter  # noqa: E402
from tabular_data_adapter import TabularDataAdapter  # noqa: E402
from tensorboard_monitoring import (  # noqa: E402
    TrainingRunContext,
    calculate_age_band_regression_metrics,
    calculate_regression_metrics,
    create_study_monitor,
    ensure_tensorboard_available,
    log_step_5_candidate,
    log_step_5_selection,
)

from candidate_space import CandidateSpace, ResolvedCandidate  # noqa: E402


RUNNER_VERSION = 1
EARLY_STOPPED_FAMILIES = {"xgboost", "mlp", "tcn", "lstm", "transformer"}

CANDIDATE_COLUMNS = [
    "settings_version",
    "model_family",
    "outer_fold",
    "candidate_number",
    "configuration_id",
    "optuna_trial_number",
    "feature_set",
    "lookback",
    "model_seed",
    "mean_inner_rmse",
    "inner_rmse_standard_deviation",
    "mean_training_seconds",
    "total_training_seconds",
    "mean_inference_seconds",
    "outer_retraining_iterations",
    "trainable_parameters",
    "configuration_json",
    "hyperparameters_json",
    "sampler_parameters_json",
    "selected_within_family",
]

FOLD_COLUMNS = [
    "settings_version",
    "model_family",
    "outer_fold",
    "candidate_number",
    "configuration_id",
    "optuna_trial_number",
    "inner_fold",
    "feature_set",
    "lookback",
    "model_seed",
    "rmse",
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

SELECTED_COLUMNS = [
    "settings_version",
    "model_family",
    "outer_fold",
    "configuration_id",
    "candidate_number",
    "feature_set",
    "lookback",
    "model_seed",
    "mean_inner_rmse",
    "inner_rmse_standard_deviation",
    "outer_retraining_iterations",
    "configuration_json",
    "hyperparameters_json",
    "selection_metric",
    "selection_direction",
]


class InnerModelSelectionError(ValueError):
    """Represent a settings, split, fitting, or artifact consistency failure."""


def _stable_json(value: Any) -> str:
    """Serialize nested configuration values consistently for CSV output."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Replace one generated JSON artifact only after writing it completely."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_csv(records: list[dict[str, Any]], columns: list[str], path: Path) -> None:
    """Write a stable-column CSV checkpoint through an atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records, columns=columns)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False, float_format="%.12g")
    temporary_path.replace(path)


class InnerSplitRepository:
    """Load and briefly cache only the inner-development splits Step 5 needs.

    Tabular splits are small enough to retain across one study. Sequence splits
    are larger, so a four-entry least-recently-used cache retains at most the
    four inner folds for the most recently evaluated lookback.
    """

    def __init__(self, expected_development_scenarios: int) -> None:
        self.expected_development_scenarios = expected_development_scenarios
        self._tabular_adapter: TabularDataAdapter | None = None
        self._sequence_adapter: SequenceDataAdapter | None = None
        self._tabular_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._sequence_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

    def get(
        self,
        *,
        family: str,
        representation: str,
        candidate: ResolvedCandidate,
        outer_fold: int,
        inner_fold: int,
    ) -> Any:
        """Return one split without providing any path to locked data."""

        if representation in {"none", "tabular"}:
            # The mean model needs target and weight carriers but ignores their
            # feature values. The smallest age-only table supplies those rows.
            feature_set = candidate.feature_set or "age_only"
            key = (outer_fold, inner_fold, feature_set)
            split = self._from_cache(
                self._tabular_cache,
                key,
                maximum_entries=16,
                loader=lambda: self._tabular().get_inner_selection_split(
                    outer_fold,
                    inner_fold,
                    feature_set,
                ),
            )
        elif representation == "sequence":
            if candidate.lookback is None:
                raise InnerModelSelectionError(
                    f"Sequence family {family!r} has no resolved lookback"
                )
            key = (outer_fold, inner_fold, candidate.lookback)
            split = self._from_cache(
                self._sequence_cache,
                key,
                maximum_entries=4,
                loader=lambda: self._sequence().get_inner_selection_split(
                    outer_fold,
                    inner_fold,
                    candidate.lookback,
                ),
            )
        else:
            raise InnerModelSelectionError(
                f"Family {family!r} has unsupported representation {representation!r}"
            )

        self._validate_split(split, outer_fold, inner_fold)
        return split

    def inner_fold_labels(
        self,
        representation: str,
        outer_fold: int,
    ) -> tuple[int, ...]:
        """Read fold labels from the adapter that owns the requested data."""

        if representation == "sequence":
            return self._sequence().inner_fold_labels(outer_fold)
        return self._tabular().inner_fold_labels(outer_fold)

    @staticmethod
    def _from_cache(
        cache: OrderedDict[tuple[Any, ...], Any],
        key: tuple[Any, ...],
        *,
        maximum_entries: int,
        loader: Callable[[], Any],
    ) -> Any:
        """Read one immutable split and bound memory used by cached sequences."""

        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        value = loader()
        cache[key] = value
        while len(cache) > maximum_entries:
            cache.popitem(last=False)
        return value

    def _tabular(self) -> TabularDataAdapter:
        """Construct the Step 2 adapter only if a tabular study requests it."""

        if self._tabular_adapter is None:
            self._tabular_adapter = TabularDataAdapter()
        return self._tabular_adapter

    def _sequence(self) -> SequenceDataAdapter:
        """Construct the Step 3 adapter only if a sequence study requests it."""

        if self._sequence_adapter is None:
            self._sequence_adapter = SequenceDataAdapter()
        return self._sequence_adapter

    def _validate_split(
        self,
        split: Any,
        outer_fold: int,
        inner_fold: int,
    ) -> None:
        """Confirm the fold is grouped, labelled, weighted, and development-only."""

        training = split.training
        validation = split.validation
        for name, dataset in (("training", training), ("validation", validation)):
            if len(dataset) == 0:
                raise InnerModelSelectionError(
                    f"Outer {outer_fold}, inner {inner_fold} has empty {name} data"
                )
            if dataset.target is None:
                raise InnerModelSelectionError(f"Inner {name} data has no RUL target")
            if "uav_id" not in dataset.metadata.columns:
                raise InnerModelSelectionError(f"Inner {name} data has no uav_id")
        if training.sample_weights is None:
            raise InnerModelSelectionError("Inner training data has no sample weights")

        training_uavs = set(training.metadata["uav_id"].astype(str))
        validation_uavs = set(validation.metadata["uav_id"].astype(str))
        overlap = sorted(training_uavs & validation_uavs)
        if overlap:
            raise InnerModelSelectionError(
                f"Inner training and validation UAVs overlap: {overlap[:5]}"
            )
        if "scenario" not in validation.metadata.columns:
            raise InnerModelSelectionError(
                "Inner validation metadata has no development scenario column"
            )
        observed_scenarios = int(validation.metadata["scenario"].nunique())
        if observed_scenarios != self.expected_development_scenarios:
            raise InnerModelSelectionError(
                "Inner validation scenario count does not match the settings: "
                f"expected {self.expected_development_scenarios}, "
                f"observed {observed_scenarios}"
            )


class StudyArtifactWriter:
    """Write inspectable checkpoints for one family and outer-fold study."""

    def __init__(
        self,
        output_dir: Path,
        *,
        settings_version: int,
        family: str,
        outer_fold: int,
        candidate_budget: int,
    ) -> None:
        self.output_dir = output_dir
        self.study_dir = output_dir / "studies"
        self.settings_version = settings_version
        self.family = family
        self.outer_fold = outer_fold
        self.candidate_budget = candidate_budget
        self.prefix = f"{family}__outer_{outer_fold:02d}"
        self.candidate_path = self.study_dir / f"{self.prefix}__candidates.csv"
        self.fold_path = self.study_dir / f"{self.prefix}__inner_folds.csv"
        self.selected_path = self.study_dir / f"{self.prefix}__selected.json"
        self.status_path = self.study_dir / f"{self.prefix}__status.json"

    def start(self) -> None:
        """Mark previous output stale before the new in-memory study begins."""

        _write_csv([], CANDIDATE_COLUMNS, self.candidate_path)
        _write_csv([], FOLD_COLUMNS, self.fold_path)
        self._write_status("running", completed_candidates=0)

    def checkpoint(self, study: Study, _: FrozenTrial) -> None:
        """Persist every successfully completed candidate evaluated so far."""

        candidate_records, fold_records = self._study_records(study)
        _write_csv(candidate_records, CANDIDATE_COLUMNS, self.candidate_path)
        _write_csv(fold_records, FOLD_COLUMNS, self.fold_path)
        self._write_status(
            "running",
            completed_candidates=len(candidate_records),
        )

    def finish(self, study: Study) -> dict[str, Any]:
        """Mark the best candidate within this family/fold study."""

        candidate_records, fold_records = self._study_records(study)
        if len(candidate_records) != self.candidate_budget:
            raise InnerModelSelectionError(
                f"Study {self.prefix} completed {len(candidate_records)} of "
                f"{self.candidate_budget} required candidates"
            )
        selected = min(
            candidate_records,
            key=lambda record: (
                record["mean_inner_rmse"],
                record["candidate_number"],
            ),
        )
        for record in candidate_records:
            record["selected_within_family"] = (
                record["configuration_id"] == selected["configuration_id"]
            )
        _write_csv(candidate_records, CANDIDATE_COLUMNS, self.candidate_path)
        _write_csv(fold_records, FOLD_COLUMNS, self.fold_path)

        selected_record = {
            key: selected[key]
            for key in SELECTED_COLUMNS
            if key in selected
        }
        selected_record["selection_metric"] = "mean_inner_rmse"
        selected_record["selection_direction"] = "minimize"
        _write_json(selected_record, self.selected_path)
        self._write_status(
            "complete",
            completed_candidates=len(candidate_records),
        )
        return selected_record

    def fail(self, message: str) -> None:
        """Make an interrupted or failed study impossible to treat as complete."""

        completed = 0
        if self.candidate_path.is_file():
            completed = len(pd.read_csv(self.candidate_path))
        self._write_status(
            "failed",
            completed_candidates=completed,
            error=message,
        )

    def _write_status(
        self,
        state: str,
        *,
        completed_candidates: int,
        error: str | None = None,
    ) -> None:
        """Write a small status record without timestamps or machine details."""

        payload: dict[str, Any] = {
            "runner_version": RUNNER_VERSION,
            "settings_version": self.settings_version,
            "model_family": self.family,
            "outer_fold": self.outer_fold,
            "state": state,
            "candidate_budget": self.candidate_budget,
            "completed_candidates": completed_candidates,
        }
        if error is not None:
            payload["error"] = error
        _write_json(payload, self.status_path)

    @staticmethod
    def _study_records(
        study: Study,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Collect records stored on complete Optuna trials in candidate order."""

        complete = [
            trial
            for trial in study.trials
            if trial.state == TrialState.COMPLETE
        ]
        complete.sort(key=lambda trial: trial.user_attrs["candidate_number"])
        candidate_records = [
            dict(trial.user_attrs["candidate_record"])
            for trial in complete
        ]
        fold_records = [
            dict(record)
            for trial in complete
            for record in trial.user_attrs["fold_records"]
        ]
        return candidate_records, fold_records


class InnerModelSelectionRunner:
    """Coordinate candidate search without crossing the locked-data boundary."""

    def __init__(
        self,
        specification_path: Path = DEFAULT_SPECIFICATION_PATH,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        self.specification_path = specification_path.resolve()
        self.output_dir = output_dir.resolve()
        # A direct Step 5 invocation receives the same fail-fast dependency
        # check as the top-level Phase 2 entry point.
        ensure_tensorboard_available()
        self.specification = load_experiment_specification(
            self.specification_path
        )
        self.settings = self.specification["settings"]
        self.architectures = self.settings["architectures"]
        self.candidate_space = CandidateSpace(self.architectures)
        self.factory = ModelAdapterFactory(self.specification_path)
        self.settings_version = int(self.settings["settings_version"])
        self.search_seed = int(self.settings["tuning"]["search_seed"])
        self.maximum_candidate_budget = int(
            self.settings["tuning"]["candidate_budget_per_architecture"]
        )
        self.outer_fold_count = int(
            self.settings["phase_1"]["expected_outer_folds"]
        )
        self.inner_fold_count = int(
            self.settings["phase_1"]["expected_inner_folds_per_outer_fold"]
        )
        self.development_scenarios = int(
            self.settings["phase_1"]["expected_development_scenarios"]
        )
        self.outer_fold_labels = list(TabularDataAdapter().outer_fold_labels())
        if len(self.outer_fold_labels) != self.outer_fold_count:
            raise InnerModelSelectionError(
                "Observed outer-fold label count does not match the settings: "
                f"expected {self.outer_fold_count}, "
                f"observed {len(self.outer_fold_labels)}"
            )

    @property
    def enabled_families(self) -> list[str]:
        """Return enabled families in the study order recorded by the settings."""

        study = self.settings["study"]
        configured_order = (
            study["architectures_to_run"]
            + study["conditional_architectures"]
            + study["optional_architectures"]
        )
        return [
            family
            for family in configured_order
            if self.settings["study"]["enabled"][family]
        ]

    def validate_request(
        self,
        families: list[str] | None,
        outer_folds: list[int] | None,
    ) -> tuple[list[str], list[int]]:
        """Resolve optional CLI filters without permitting disabled families."""

        selected_families = self.enabled_families if families is None else families
        if not selected_families:
            raise InnerModelSelectionError("At least one model family is required")
        if len(selected_families) != len(set(selected_families)):
            raise InnerModelSelectionError(
                "Requested model families contain duplicates"
            )
        for family in selected_families:
            if family not in self.architectures:
                raise InnerModelSelectionError(f"Unknown model family {family!r}")
            if not self.settings["study"]["enabled"][family]:
                raise InnerModelSelectionError(
                    f"Model family {family!r} is disabled in the settings"
                )

        selected_folds = (
            list(self.outer_fold_labels)
            if outer_folds is None
            else outer_folds
        )
        if len(selected_folds) != len(set(selected_folds)):
            raise InnerModelSelectionError("Requested outer folds contain duplicates")
        invalid = [
            fold
            for fold in selected_folds
            if fold not in self.outer_fold_labels
        ]
        if invalid:
            raise InnerModelSelectionError(
                f"Unknown outer folds {invalid}. Available: {self.outer_fold_labels}"
            )
        return selected_families, selected_folds

    def run(
        self,
        families: list[str] | None = None,
        outer_folds: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run every requested independent family/fold tuning study."""

        selected_families, selected_folds = self.validate_request(
            families,
            outer_folds,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for family in selected_families:
            for outer_fold in selected_folds:
                self.run_study(family, outer_fold)
        return self.consolidate_artifacts()

    def run_study(self, family: str, outer_fold: int) -> dict[str, Any]:
        """Tune one family on one outer fold and save its selected candidate."""

        self.validate_request([family], [outer_fold])
        architecture = self.architectures[family]
        candidate_budget = self.candidate_space.candidate_budget(
            family,
            self.maximum_candidate_budget,
        )
        writer = StudyArtifactWriter(
            self.output_dir,
            settings_version=self.settings_version,
            family=family,
            outer_fold=outer_fold,
            candidate_budget=candidate_budget,
        )
        writer.start()
        self.consolidate_artifacts()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.search_seed)
        study = optuna.create_study(
            study_name=f"{writer.prefix}__settings_{self.settings_version}",
            direction="minimize",
            sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
        )
        split_repository = InnerSplitRepository(self.development_scenarios)
        seen_configurations: set[str] = set()

        # One shared writer covers every candidate and inner fold in this
        # study, so the generated directory count stays fixed at one study
        # instead of growing with the candidate budget and inner-fold count.
        with create_study_monitor(
            stage="step_5",
            model_family=family,
            outer_fold=outer_fold,
        ) as study_monitor:

            def objective(trial: Trial) -> float:
                candidate = self.candidate_space.resolve(family, trial)
                canonical = candidate.canonical_json()
                if canonical in seen_configurations:
                    raise optuna.TrialPruned(
                        "Resolved candidate duplicates an earlier complete candidate"
                    )
                candidate_number = len(seen_configurations) + 1
                candidate_record, fold_records = self._evaluate_candidate(
                    trial=trial,
                    family=family,
                    architecture=architecture,
                    candidate=candidate,
                    candidate_number=candidate_number,
                    outer_fold=outer_fold,
                    split_repository=split_repository,
                    study_monitor=study_monitor,
                )
                trial.set_user_attr("candidate_number", candidate_number)
                trial.set_user_attr("candidate_record", candidate_record)
                trial.set_user_attr("fold_records", fold_records)
                trial.set_user_attr("configuration_json", canonical)
                seen_configurations.add(canonical)
                return float(candidate_record["mean_inner_rmse"])

            maximum_attempts = candidate_budget * 20
            attempts = 0
            try:
                while len(seen_configurations) < candidate_budget:
                    if attempts >= maximum_attempts:
                        raise InnerModelSelectionError(
                            f"Could not generate {candidate_budget} distinct "
                            f"candidates for {family!r}"
                        )
                    study.optimize(
                        objective,
                        n_trials=1,
                        n_jobs=1,
                        callbacks=[writer.checkpoint],
                        gc_after_trial=True,
                        show_progress_bar=False,
                    )
                    attempts += 1
            except Exception as error:
                writer.fail(str(error))
                self.consolidate_artifacts()
                raise

            selected = writer.finish(study)
            log_step_5_selection(
                model_family=family,
                outer_fold=outer_fold,
                selected_record=selected,
            )
        self.consolidate_artifacts()
        return selected

    def _evaluate_candidate(
        self,
        *,
        trial: Trial,
        family: str,
        architecture: dict[str, Any],
        candidate: ResolvedCandidate,
        candidate_number: int,
        outer_fold: int,
        split_repository: InnerSplitRepository,
        study_monitor: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Fit one resolved candidate on all four inner UAV folds."""

        configuration_id = (
            f"{family}__outer_{outer_fold:02d}__candidate_{candidate_number:03d}"
        )
        fold_records: list[dict[str, Any]] = []
        inner_fold_labels = split_repository.inner_fold_labels(
            architecture["representation"],
            outer_fold,
        )
        if len(inner_fold_labels) != self.inner_fold_count:
            raise InnerModelSelectionError(
                "Observed inner-fold label count does not match the settings: "
                f"expected {self.inner_fold_count}, observed {len(inner_fold_labels)}"
            )
        for inner_fold in inner_fold_labels:
            split = split_repository.get(
                family=family,
                representation=architecture["representation"],
                candidate=candidate,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
            )
            context = TrainingRunContext(
                stage="step_5",
                model_family=family,
                representation=architecture["representation"],
                outer_fold=outer_fold,
                seed=self.search_seed,
                configuration_id=configuration_id,
                candidate_number=candidate_number,
                inner_fold=inner_fold,
                feature_set=candidate.feature_set,
                lookback=candidate.lookback,
            )
            model = None
            with study_monitor.fit(context) as monitor:
                monitor.start_fit(
                    hyperparameters=candidate.hyperparameters,
                    training_data=split.training,
                    validation_data=split.validation,
                )
                model = self.factory.create(
                    family,
                    candidate.hyperparameters,
                    seed=self.search_seed,
                    training_monitor=monitor,
                )
                try:
                    summary = model.fit(split.training, split.validation)
                    prediction_started = perf_counter()
                    predictions = model.predict(split.validation)
                    inference_seconds = perf_counter() - prediction_started
                    observed_targets = target_values(split.validation)
                    development_metrics = calculate_regression_metrics(
                        observed_targets,
                        predictions,
                    )
                    monitored_metrics = {
                        **development_metrics,
                        **calculate_age_band_regression_metrics(
                            observed_targets,
                            predictions,
                            split.validation.metadata["cutoff"],
                        ),
                    }
                    rmse = development_metrics["rmse"]
                    if not np.isfinite(rmse):
                        raise InnerModelSelectionError(
                            f"Configuration {configuration_id} produced "
                            "non-finite RMSE"
                        )
                    if summary.best_validation_rmse is None or not np.isclose(
                        summary.best_validation_rmse,
                        rmse,
                        rtol=1e-10,
                        atol=1e-10,
                    ):
                        raise InnerModelSelectionError(
                            "Adapter and Step 5 validation RMSE calculations "
                            "disagree"
                        )
                    monitor.complete_fit(
                        training_summary=summary.to_dict(),
                        inference_seconds=inference_seconds,
                        evaluation_metrics=monitored_metrics,
                        prediction_rows=len(predictions),
                    )
                finally:
                    # Live writers must never become part of a fitted model or
                    # survive beyond this isolated fold fit.
                    model.detach_training_monitor()

            fold_records.append(
                {
                    "settings_version": self.settings_version,
                    "model_family": family,
                    "outer_fold": outer_fold,
                    "candidate_number": candidate_number,
                    "configuration_id": configuration_id,
                    "optuna_trial_number": trial.number,
                    "inner_fold": inner_fold,
                    "feature_set": candidate.feature_set,
                    "lookback": candidate.lookback,
                    "model_seed": self.search_seed,
                    "rmse": rmse,
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
                    "best_epoch_or_iteration": (
                        summary.best_epoch_or_iteration
                    ),
                    "trainable_parameters": summary.trainable_parameters,
                }
            )
            del model
            gc.collect()

        rmse_values = np.asarray(
            [record["rmse"] for record in fold_records],
            dtype=np.float64,
        )
        training_times = np.asarray(
            [record["training_seconds"] for record in fold_records],
            dtype=np.float64,
        )
        inference_times = np.asarray(
            [record["inference_seconds"] for record in fold_records],
            dtype=np.float64,
        )
        candidate_record = {
            "settings_version": self.settings_version,
            "model_family": family,
            "outer_fold": outer_fold,
            "candidate_number": candidate_number,
            "configuration_id": configuration_id,
            "optuna_trial_number": trial.number,
            "feature_set": candidate.feature_set,
            "lookback": candidate.lookback,
            "model_seed": self.search_seed,
            "mean_inner_rmse": float(np.mean(rmse_values)),
            "inner_rmse_standard_deviation": float(np.std(rmse_values)),
            "mean_training_seconds": float(np.mean(training_times)),
            "total_training_seconds": float(np.sum(training_times)),
            "mean_inference_seconds": float(np.mean(inference_times)),
            "outer_retraining_iterations": self._retraining_iterations(
                family,
                fold_records,
            ),
            "trainable_parameters": fold_records[0]["trainable_parameters"],
            "configuration_json": candidate.canonical_json(),
            "hyperparameters_json": _stable_json(candidate.hyperparameters),
            "sampler_parameters_json": _stable_json(trial.params),
            "selected_within_family": False,
        }
        log_step_5_candidate(
            model_family=family,
            outer_fold=outer_fold,
            candidate_number=candidate_number,
            candidate_record=candidate_record,
            hyperparameters=candidate.hyperparameters,
        )
        return candidate_record, fold_records

    @staticmethod
    def _retraining_iterations(
        family: str,
        fold_records: list[dict[str, Any]],
    ) -> int | None:
        """Convert four inner stopping points into the fixed outer duration."""

        if family not in EARLY_STOPPED_FAMILIES:
            return None
        values = [
            record["best_epoch_or_iteration"]
            for record in fold_records
            if record["best_epoch_or_iteration"] is not None
        ]
        if len(values) != len(fold_records):
            raise InnerModelSelectionError(
                f"Early-stopped family {family!r} did not record every best duration"
            )
        # The median of four integer durations can end in .5. Rounding halves
        # upward gives one explicit positive integer for outer retraining.
        median = float(np.median(np.asarray(values, dtype=np.float64)))
        return max(1, int(math.floor(median + 0.5)))

    def consolidate_artifacts(self) -> dict[str, Any]:
        """Combine completed studies and label the overall Step 5 status."""

        study_dir = self.output_dir / "studies"
        candidate_records: list[dict[str, Any]] = []
        fold_records: list[dict[str, Any]] = []
        selected_records: list[dict[str, Any]] = []
        complete_studies: list[str] = []
        incomplete_studies: list[str] = []

        for family in self.enabled_families:
            budget = self.candidate_space.candidate_budget(
                family,
                self.maximum_candidate_budget,
            )
            for outer_fold in self.outer_fold_labels:
                prefix = f"{family}__outer_{outer_fold:02d}"
                status_path = study_dir / f"{prefix}__status.json"
                status = self._read_status(status_path)
                valid = (
                    status is not None
                    and status.get("state") == "complete"
                    and status.get("runner_version") == RUNNER_VERSION
                    and status.get("settings_version") == self.settings_version
                    and status.get("candidate_budget") == budget
                )
                if not valid:
                    incomplete_studies.append(prefix)
                    continue

                candidates = pd.read_csv(
                    study_dir / f"{prefix}__candidates.csv"
                ).to_dict("records")
                folds = pd.read_csv(
                    study_dir / f"{prefix}__inner_folds.csv"
                ).to_dict("records")
                selected = json.loads(
                    (study_dir / f"{prefix}__selected.json").read_text(
                        encoding="utf-8"
                    )
                )
                if len(candidates) != budget:
                    raise InnerModelSelectionError(
                        f"Completed study {prefix} has {len(candidates)} candidates"
                    )
                if len(folds) != budget * self.inner_fold_count:
                    raise InnerModelSelectionError(
                        f"Completed study {prefix} has {len(folds)} fold rows"
                    )
                candidate_records.extend(candidates)
                fold_records.extend(folds)
                selected_records.append(selected)
                complete_studies.append(prefix)

        candidate_records.sort(
            key=lambda row: (
                row["model_family"],
                int(row["outer_fold"]),
                int(row["candidate_number"]),
            )
        )
        fold_records.sort(
            key=lambda row: (
                row["model_family"],
                int(row["outer_fold"]),
                int(row["candidate_number"]),
                int(row["inner_fold"]),
            )
        )
        selected_records.sort(
            key=lambda row: (row["model_family"], int(row["outer_fold"]))
        )
        _write_csv(
            candidate_records,
            CANDIDATE_COLUMNS,
            self.output_dir / "candidate_results.csv",
        )
        _write_csv(
            fold_records,
            FOLD_COLUMNS,
            self.output_dir / "inner_fold_results.csv",
        )
        _write_csv(
            selected_records,
            SELECTED_COLUMNS,
            self.output_dir / "selected_configurations.csv",
        )

        expected_studies = len(self.enabled_families) * len(self.outer_fold_labels)
        manifest = {
            "runner_version": RUNNER_VERSION,
            "settings_version": self.settings_version,
            "status": (
                "complete"
                if len(complete_studies) == expected_studies
                else "partial"
            ),
            "tuning_scope": "within_architecture",
            "automatic_within_family_selection": True,
            "automatic_architecture_selection": False,
            "primary_metric": "mean_inner_rmse",
            "direction": "minimize",
            "sampler": "Optuna TPESampler",
            "search_seed": self.search_seed,
            "maximum_candidate_budget": self.maximum_candidate_budget,
            "outer_fold_labels": self.outer_fold_labels,
            "inner_folds_per_outer_fold": self.inner_fold_count,
            "development_scenarios": self.development_scenarios,
            "enabled_families": self.enabled_families,
            "completed_studies": complete_studies,
            "incomplete_studies": incomplete_studies,
            "expected_study_count": expected_studies,
            "completed_study_count": len(complete_studies),
            "selected_configuration_rows": len(selected_records),
            "candidate_result_rows": len(candidate_records),
            "inner_fold_result_rows": len(fold_records),
            "locked_data_loaded": False,
            "test_data_loaded": False,
            "libraries": {
                "numpy": version("numpy"),
                "optuna": version("optuna"),
                "pandas": version("pandas"),
            },
            "artifacts": {
                "candidate_results": "candidate_results.csv",
                "inner_fold_results": "inner_fold_results.csv",
                "selected_configurations": "selected_configurations.csv",
                "study_checkpoints": "studies/",
            },
        }
        _write_json(
            manifest,
            self.output_dir / "selection_manifest.json",
        )
        return manifest

    @staticmethod
    def _read_status(path: Path) -> dict[str, Any] | None:
        """Return one status object, treating absent files as incomplete."""

        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise InnerModelSelectionError(
                f"Cannot read study status {path}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise InnerModelSelectionError(f"Study status {path} is not an object")
        return value
