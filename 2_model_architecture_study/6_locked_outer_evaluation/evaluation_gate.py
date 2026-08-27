"""Verify that inner selection is complete before locked data can be loaded.

This module intentionally imports no Step 2 or Step 3 data adapter. Its only
job is to validate the architecture study settings, the complete Step 5 manifest, and
the selected configurations. The evaluation runner constructs a data adapter
only after this gate returns a valid plan.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
PHASE_DIR = STEP_DIR.parent
MODEL_ADAPTER_DIR = PHASE_DIR / "4_model_adapters"
DEFAULT_SPECIFICATION_PATH = (
    PHASE_DIR
    / "1_architecture_study_settings"
    / "artifacts"
    / "experiment_specification.json"
)


if str(MODEL_ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_ADAPTER_DIR))
if str(PHASE_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE_DIR))

from run_layout import (  # noqa: E402
    STEP_5_DIRECTORY_NAME,
    step_directory_for_specification,
)

from model_registry import (  # noqa: E402
    ADAPTER_CLASSES,
    ModelAdapterFactory,
    load_experiment_specification,
)


def default_selection_manifest_path() -> Path:
    """Locate Step 5's manifest inside the current run folder.

    Resolved when it is called rather than at import time, so importing this
    gate does not require Step 1 to have run yet.
    """

    return (
        step_directory_for_specification(STEP_5_DIRECTORY_NAME)
        / "selection_manifest.json"
    )


def default_selected_configurations_path() -> Path:
    """Locate Step 5's selected configurations inside the current run folder."""

    return (
        step_directory_for_specification(STEP_5_DIRECTORY_NAME)
        / "selected_configurations.csv"
    )


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


class LockedEvaluationGateError(ValueError):
    """Explain why locked outer evaluation must not begin yet."""


@dataclass(frozen=True)
class SelectedConfiguration:
    """Hold one validated Step 5 selection for one family and outer fold."""

    settings_version: int
    model_family: str
    outer_fold: int
    configuration_id: str
    feature_set: str | None
    lookback: int | None
    hyperparameters: dict[str, Any]
    outer_retraining_iterations: int | None
    mean_inner_rmse: float


@dataclass(frozen=True)
class LockedEvaluationPlan:
    """Store all choices that must be fixed before locked data is opened."""

    settings: dict[str, Any]
    enabled_families: tuple[str, ...]
    outer_fold_labels: tuple[int, ...]
    retraining_seeds: tuple[int, ...]
    configurations: dict[tuple[str, int], SelectedConfiguration]

    def configuration(
        self,
        family: str,
        outer_fold: int,
    ) -> SelectedConfiguration:
        """Return one selected configuration using an explicit compound key."""

        try:
            return self.configurations[(family, outer_fold)]
        except KeyError as error:
            raise LockedEvaluationGateError(
                f"No selected configuration for {family!r}, outer fold {outer_fold}"
            ) from error

    def seeds_for(self, family: str) -> tuple[int, ...]:
        """Use three seeds only for model families with stochastic training."""

        if family not in ADAPTER_CLASSES:
            raise LockedEvaluationGateError(f"Unknown model family {family!r}")
        if ADAPTER_CLASSES[family].stochastic:
            return self.retraining_seeds
        return (self.retraining_seeds[0],)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    """Read one required object and provide a concise prerequisite error."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LockedEvaluationGateError(
            f"Cannot read {description} at {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise LockedEvaluationGateError(f"{description} must contain an object")
    return value


def _enabled_families(settings: dict[str, Any]) -> tuple[str, ...]:
    """Preserve the architecture order explicitly recorded in the settings."""

    study = settings["study"]
    order = (
        study["architectures_to_run"]
        + study["conditional_architectures"]
        + study["optional_architectures"]
    )
    return tuple(
        family
        for family in order
        if settings["study"]["enabled"][family]
    )


def _optional_text(value: Any) -> str | None:
    """Convert a CSV cell to optional text without turning NaN into a name."""

    if value is None or pd.isna(value):
        return None
    return str(value)


def _optional_positive_integer(value: Any) -> int | None:
    """Convert a blank CSV duration to None and reject fractional durations."""

    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        raise LockedEvaluationGateError(
            f"Outer retraining duration must be a positive integer, got {value!r}"
        )
    return int(numeric)


def _parse_selected_row(
    row: pd.Series,
    *,
    settings: dict[str, Any],
) -> SelectedConfiguration:
    """Parse and cross-check one selected-configuration CSV row."""

    family = str(row["model_family"])
    outer_fold = int(row["outer_fold"])
    try:
        configuration = json.loads(str(row["configuration_json"]))
        hyperparameters = json.loads(str(row["hyperparameters_json"]))
    except json.JSONDecodeError as error:
        raise LockedEvaluationGateError(
            f"Selected configuration for {family!r}, fold {outer_fold} has invalid JSON"
        ) from error
    if not isinstance(configuration, dict) or not isinstance(hyperparameters, dict):
        raise LockedEvaluationGateError(
            f"Selected configuration for {family!r}, fold {outer_fold} is not an object"
        )
    if configuration.get("hyperparameters") != hyperparameters:
        raise LockedEvaluationGateError(
            f"Configuration and hyperparameter JSON disagree for {family!r}"
        )

    feature_set = configuration.get("feature_set")
    lookback_value = configuration.get("lookback")
    lookback = int(lookback_value) if lookback_value is not None else None
    architecture = settings["architectures"][family]
    if feature_set is not None and feature_set not in architecture["feature_sets"]:
        raise LockedEvaluationGateError(
            f"Selected feature set {feature_set!r} is invalid for {family!r}"
        )
    if lookback is not None and lookback not in architecture["lookbacks"]:
        raise LockedEvaluationGateError(
            f"Selected lookback {lookback!r} is invalid for {family!r}"
        )
    representation = architecture["representation"]
    if representation == "none" and (feature_set is not None or lookback is not None):
        raise LockedEvaluationGateError(
            f"No-input family {family!r} has a feature set or lookback"
        )
    if representation == "tabular" and (
        feature_set is None or lookback is not None
    ):
        raise LockedEvaluationGateError(
            f"Tabular family {family!r} has inconsistent representation fields"
        )
    if representation == "sequence" and (
        feature_set is not None or lookback is None
    ):
        raise LockedEvaluationGateError(
            f"Sequence family {family!r} has inconsistent representation fields"
        )
    if representation == "trajectory" and (
        feature_set is not None or lookback is not None
    ):
        raise LockedEvaluationGateError(
            f"Trajectory family {family!r} has inconsistent representation fields"
        )
    if _optional_text(row["feature_set"]) != feature_set:
        raise LockedEvaluationGateError(
            f"Feature-set columns disagree for {family!r}, fold {outer_fold}"
        )

    csv_lookback = _optional_positive_integer(row["lookback"])
    if csv_lookback != lookback:
        raise LockedEvaluationGateError(
            f"Lookback columns disagree for {family!r}, fold {outer_fold}"
        )
    duration = _optional_positive_integer(row["outer_retraining_iterations"])
    if family in EARLY_STOPPED_FAMILIES and duration is None:
        raise LockedEvaluationGateError(
            f"Selected early-stopped family {family!r} has no retraining duration"
        )
    if family not in EARLY_STOPPED_FAMILIES and duration is not None:
        raise LockedEvaluationGateError(
            f"Non-early-stopped family {family!r} has a retraining duration"
        )

    mean_inner_rmse = float(row["mean_inner_rmse"])
    if not np.isfinite(mean_inner_rmse) or mean_inner_rmse < 0:
        raise LockedEvaluationGateError(
            f"Selected mean inner RMSE is invalid for {family!r}, fold {outer_fold}"
        )
    _validate_hyperparameter_values(family, hyperparameters, architecture["search"])
    expected_prefix = f"{family}__outer_{outer_fold:02d}__candidate_"
    configuration_id = str(row["configuration_id"])
    if not configuration_id.startswith(expected_prefix):
        raise LockedEvaluationGateError(
            f"Configuration ID {configuration_id!r} has an unexpected format"
        )
    return SelectedConfiguration(
        settings_version=int(row["settings_version"]),
        model_family=family,
        outer_fold=outer_fold,
        configuration_id=configuration_id,
        feature_set=feature_set,
        lookback=lookback,
        hyperparameters=hyperparameters,
        outer_retraining_iterations=duration,
        mean_inner_rmse=mean_inner_rmse,
    )


def _validate_hyperparameter_values(
    family: str,
    hyperparameters: dict[str, Any],
    search: dict[str, dict[str, Any]],
) -> None:
    """Require every resolved value to remain inside the recorded search space."""

    if set(hyperparameters) != set(search):
        raise LockedEvaluationGateError(
            f"Selected hyperparameter names differ from the settings for {family!r}"
        )
    for name, definition in search.items():
        value = hyperparameters[name]
        kind = definition["kind"]
        valid = False
        if kind == "fixed":
            valid = value == definition["value"]
        elif kind == "categorical":
            valid = value in definition["values"]
        elif kind == "categorical_integer_sequences":
            try:
                supplied_sequence = list(value)
            except TypeError:
                supplied_sequence = []
            valid = supplied_sequence in [
                list(item) for item in definition["values"]
            ]
        elif kind in {"uniform", "log_uniform"}:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = float("nan")
            valid = (
                np.isfinite(numeric)
                and float(definition["low"]) <= numeric
                and numeric <= float(definition["high"])
            )
        if not valid:
            raise LockedEvaluationGateError(
                f"Selected value {value!r} for {family}.{name} is outside the settings"
            )


def _manifest_artifact_path(
    manifest: dict[str, Any],
    manifest_path: Path,
    artifact_name: str,
) -> Path:
    """Resolve one declared Step 5 artifact inside its own artifact directory."""

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise LockedEvaluationGateError("Step 5 manifest has no artifacts object")
    relative_value = artifacts.get(artifact_name)
    if not isinstance(relative_value, str):
        raise LockedEvaluationGateError(
            f"Step 5 manifest does not declare artifact {artifact_name!r}"
        )
    relative_path = Path(relative_value)
    if relative_path.is_absolute():
        raise LockedEvaluationGateError("Step 5 artifact paths must be relative")
    artifact_root = manifest_path.resolve().parent
    resolved = (artifact_root / relative_path).resolve()
    try:
        resolved.relative_to(artifact_root)
    except ValueError as error:
        raise LockedEvaluationGateError(
            f"Step 5 artifact path escapes its directory: {relative_value}"
        ) from error
    return resolved


def _selected_flag(value: Any) -> bool:
    """Interpret the generated Boolean CSV cell without accepting other text."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise LockedEvaluationGateError(
        f"Candidate selected flag has invalid value {value!r}"
    )


def _verify_step5_result_tables(
    *,
    settings: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    selected_path: Path,
    selected_table: pd.DataFrame,
    configurations: dict[tuple[str, int], SelectedConfiguration],
) -> None:
    """Recalculate Step 5 selections and stopping durations from detail rows."""

    declared_selected_path = _manifest_artifact_path(
        manifest,
        manifest_path,
        "selected_configurations",
    )
    if declared_selected_path != selected_path.resolve():
        raise LockedEvaluationGateError(
            "Selected configuration path differs from the Step 5 manifest"
        )
    candidate_path = _manifest_artifact_path(
        manifest,
        manifest_path,
        "candidate_results",
    )
    fold_path = _manifest_artifact_path(
        manifest,
        manifest_path,
        "inner_fold_results",
    )
    try:
        candidates = pd.read_csv(candidate_path)
        folds = pd.read_csv(fold_path)
    except (OSError, pd.errors.ParserError) as error:
        raise LockedEvaluationGateError(
            f"Cannot read detailed Step 5 results: {error}"
        ) from error

    candidate_columns = {
        "model_family",
        "outer_fold",
        "candidate_number",
        "configuration_id",
        "mean_inner_rmse",
        "outer_retraining_iterations",
        "configuration_json",
        "hyperparameters_json",
        "selected_within_family",
    }
    fold_columns = {
        "model_family",
        "outer_fold",
        "configuration_id",
        "inner_fold",
        "rmse",
        "best_epoch_or_iteration",
    }
    if not candidate_columns.issubset(candidates.columns):
        raise LockedEvaluationGateError(
            "Step 5 candidate results are missing required columns"
        )
    if not fold_columns.issubset(folds.columns):
        raise LockedEvaluationGateError(
            "Step 5 inner-fold results are missing required columns"
        )
    if len(candidates) != int(manifest["candidate_result_rows"]):
        raise LockedEvaluationGateError(
            "Step 5 candidate-result row count differs from its manifest"
        )
    if len(folds) != int(manifest["inner_fold_result_rows"]):
        raise LockedEvaluationGateError(
            "Step 5 inner-fold row count differs from its manifest"
        )
    if candidates.duplicated(["configuration_id"]).any():
        raise LockedEvaluationGateError(
            "Step 5 candidate results contain duplicate configuration IDs"
        )
    candidate_keys = set(
        zip(
            candidates["model_family"].astype(str),
            candidates["outer_fold"].astype(int),
            strict=True,
        )
    )
    if candidate_keys != set(configurations):
        raise LockedEvaluationGateError(
            "Step 5 candidate family/fold keys differ from selected configurations"
        )

    inner_fold_count = int(
        settings["phase_1"]["expected_inner_folds_per_outer_fold"]
    )
    maximum_budget = int(
        settings["tuning"]["candidate_budget_per_architecture"]
    )
    if set(folds["configuration_id"]) != set(candidates["configuration_id"]):
        raise LockedEvaluationGateError(
            "Step 5 candidate and inner-fold configuration IDs differ"
        )
    fold_counts = folds.groupby("configuration_id", sort=False).size()
    if not (fold_counts == inner_fold_count).all():
        raise LockedEvaluationGateError(
            "A Step 5 candidate does not have exactly four inner-fold results"
        )

    for key, selected in configurations.items():
        family, outer_fold = key
        group = candidates.loc[
            (candidates["model_family"] == family)
            & (candidates["outer_fold"].astype(int) == outer_fold)
        ].copy()
        architecture = settings["architectures"][family]
        alternatives = max(
            len(architecture["feature_sets"]),
            len(architecture["lookbacks"]),
            1,
        )
        expected_budget = (
            maximum_budget
            if architecture["search"] or alternatives > 1
            else 1
        )
        if len(group) != expected_budget:
            raise LockedEvaluationGateError(
                f"Step 5 candidate count is wrong for {family!r}, fold {outer_fold}"
            )
        flags = group["selected_within_family"].map(_selected_flag)
        if int(flags.sum()) != 1:
            raise LockedEvaluationGateError(
                f"Step 5 must flag one selected candidate for {family!r}, "
                f"fold {outer_fold}"
            )
        best = group.sort_values(
            ["mean_inner_rmse", "candidate_number"],
            kind="stable",
        ).iloc[0]
        if str(best["configuration_id"]) != selected.configuration_id:
            raise LockedEvaluationGateError(
                f"Step 5 selected candidate is not the minimum for {family!r}, "
                f"fold {outer_fold}"
            )
        if not _selected_flag(
            group.loc[
                group["configuration_id"] == selected.configuration_id,
                "selected_within_family",
            ].iloc[0]
        ):
            raise LockedEvaluationGateError(
                f"Step 5 selected flag disagrees for {selected.configuration_id!r}"
            )

        selected_folds = folds.loc[
            folds["configuration_id"] == selected.configuration_id
        ]
        fold_rmse = selected_folds["rmse"].to_numpy(dtype=np.float64)
        recalculated_rmse = float(np.mean(fold_rmse))
        if not np.isclose(recalculated_rmse, selected.mean_inner_rmse):
            raise LockedEvaluationGateError(
                "Step 5 mean RMSE does not match folds for "
                f"{selected.configuration_id!r}"
            )
        if family in EARLY_STOPPED_FAMILIES:
            durations = selected_folds["best_epoch_or_iteration"].to_numpy(
                dtype=np.float64
            )
            if not np.isfinite(durations).all():
                raise LockedEvaluationGateError(
                    f"Step 5 stopping durations are incomplete for {family!r}"
                )
            median = float(np.median(durations))
            recalculated_duration = max(1, int(math.floor(median + 0.5)))
            if recalculated_duration != selected.outer_retraining_iterations:
                raise LockedEvaluationGateError(
                    f"Step 5 median duration disagrees for {family!r}, "
                    f"fold {outer_fold}"
                )

        selected_row = selected_table.loc[
            (selected_table["model_family"] == family)
            & (selected_table["outer_fold"].astype(int) == outer_fold)
        ].iloc[0]
        if json.loads(str(best["configuration_json"])) != json.loads(
            str(selected_row["configuration_json"])
        ):
            raise LockedEvaluationGateError(
                f"Step 5 selected configuration details disagree for {family!r}"
            )


def build_locked_evaluation_plan(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    selection_manifest_path: Path | None = None,
    selected_configurations_path: Path | None = None,
) -> LockedEvaluationPlan:
    """Open the locked-evaluation gate only for a complete matching Step 5."""

    if selection_manifest_path is None:
        selection_manifest_path = default_selection_manifest_path()
    if selected_configurations_path is None:
        selected_configurations_path = default_selected_configurations_path()
    specification = load_experiment_specification(specification_path)
    settings = specification["settings"]
    settings_version = int(settings["settings_version"])
    manifest = _read_json(selection_manifest_path, "Step 5 selection manifest")

    completed = int(manifest.get("completed_study_count", -1))
    expected = int(manifest.get("expected_study_count", -1))
    if manifest.get("status") != "complete" or completed != expected:
        raise LockedEvaluationGateError(
            "Step 5 is incomplete, so locked validation data must remain closed: "
            f"completed {completed}/{expected} family/fold studies"
        )
    if manifest.get("settings_version") != settings_version:
        raise LockedEvaluationGateError(
            "Step 5 and the current architecture study settings versions differ"
        )
    if manifest.get("locked_data_loaded") is not False:
        raise LockedEvaluationGateError(
            "Step 5 reports locked-data access and cannot authorize evaluation"
        )
    if manifest.get("test_data_loaded") is not False:
        raise LockedEvaluationGateError(
            "Step 5 reports test-data access and cannot authorize evaluation"
        )

    enabled_families = _enabled_families(settings)
    if tuple(manifest.get("enabled_families", ())) != enabled_families:
        raise LockedEvaluationGateError(
            "Step 5 enabled families differ from the current settings"
        )
    outer_fold_labels = tuple(
        int(value) for value in manifest.get("outer_fold_labels", ())
    )
    expected_outer_folds = int(
        settings["phase_1"]["expected_outer_folds"]
    )
    if len(outer_fold_labels) != expected_outer_folds:
        raise LockedEvaluationGateError(
            "Step 5 outer-fold label count differs from the current settings"
        )
    expected_keys = {
        (family, outer_fold)
        for family in enabled_families
        for outer_fold in outer_fold_labels
    }
    if expected != len(expected_keys):
        raise LockedEvaluationGateError(
            "Step 5 expected study count does not match families and folds"
        )

    required_columns = {
        "settings_version",
        "model_family",
        "outer_fold",
        "configuration_id",
        "feature_set",
        "lookback",
        "mean_inner_rmse",
        "outer_retraining_iterations",
        "configuration_json",
        "hyperparameters_json",
        "selection_metric",
        "selection_direction",
    }
    try:
        selected_table = pd.read_csv(selected_configurations_path)
    except (OSError, pd.errors.ParserError) as error:
        raise LockedEvaluationGateError(
            f"Cannot read selected configurations: {error}"
        ) from error
    missing_columns = sorted(required_columns - set(selected_table.columns))
    if missing_columns:
        raise LockedEvaluationGateError(
            f"Selected configurations are missing columns {missing_columns}"
        )
    if len(selected_table) != len(expected_keys):
        raise LockedEvaluationGateError(
            "Selected configuration row count does not cover every family/fold"
        )
    if selected_table.duplicated(["model_family", "outer_fold"]).any():
        raise LockedEvaluationGateError(
            "Selected configurations contain duplicate family/fold keys"
        )

    configurations: dict[tuple[str, int], SelectedConfiguration] = {}
    factory = ModelAdapterFactory(specification_path)
    validation_seed = int(settings["tuning"]["retraining_seeds"][0])
    for _, row in selected_table.iterrows():
        family = str(row["model_family"])
        if family not in enabled_families:
            raise LockedEvaluationGateError(
                f"Selected table contains disabled or unknown family {family!r}"
            )
        selected = _parse_selected_row(row, settings=settings)
        if selected.settings_version != settings_version:
            raise LockedEvaluationGateError(
                f"Selected configuration {selected.configuration_id!r} has "
                "the wrong settings version"
            )
        if row["selection_metric"] != "mean_inner_rmse":
            raise LockedEvaluationGateError("Unexpected Step 5 selection metric")
        if row["selection_direction"] != "minimize":
            raise LockedEvaluationGateError("Unexpected Step 5 selection direction")
        key = (selected.model_family, selected.outer_fold)
        if key not in expected_keys:
            raise LockedEvaluationGateError(f"Unexpected selected key {key}")
        factory.create(
            selected.model_family,
            selected.hyperparameters,
            seed=validation_seed,
            training_iterations=selected.outer_retraining_iterations,
        )
        configurations[key] = selected
    if set(configurations) != expected_keys:
        missing = sorted(expected_keys - set(configurations))
        raise LockedEvaluationGateError(
            f"Selected configurations do not cover keys {missing}"
        )

    _verify_step5_result_tables(
        settings=settings,
        manifest=manifest,
        manifest_path=selection_manifest_path,
        selected_path=selected_configurations_path,
        selected_table=selected_table,
        configurations=configurations,
    )

    retraining_seeds = tuple(
        int(seed) for seed in settings["tuning"]["retraining_seeds"]
    )
    if not retraining_seeds or len(retraining_seeds) != len(set(retraining_seeds)):
        raise LockedEvaluationGateError("Retraining seeds must be non-empty and unique")
    return LockedEvaluationPlan(
        settings=settings,
        enabled_families=enabled_families,
        outer_fold_labels=outer_fold_labels,
        retraining_seeds=retraining_seeds,
        configurations=configurations,
    )
