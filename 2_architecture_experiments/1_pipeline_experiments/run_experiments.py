"""Run declarative Phase 1, Phase 2, and Phase 3 experiment definitions."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from typing import Any

import pandas as pd

MANAGER_DIR = Path(__file__).resolve().parent
if str(MANAGER_DIR) not in sys.path:
    sys.path.insert(0, str(MANAGER_DIR))

from experiment_paths import (
    artifact_directory,
    gallery_directory,
    pipeline_owner,
    pipeline_run_name,
    repository_path,
    run_directory,
)
from experiment_config import ExperimentConfigError, read_experiment_config
from promote_calibrated_ensemble import (
    PromotionError,
    run_promotion as run_calibrated_ensemble_promotion,
)


ARCHITECTURE_EXPERIMENTS_ROOT = MANAGER_DIR.parent
REPOSITORY_ROOT = ARCHITECTURE_EXPERIMENTS_ROOT.parent
DEFAULT_CONFIG_PATH = MANAGER_DIR / "pipeline_experiments.toml"
EXPERIMENTS_DIR = MANAGER_DIR / "experiments"
# Kept as an injectable alias for tests and compatibility helpers.
RUNS_DIR = EXPERIMENTS_DIR
PHASE_1_ROOT = REPOSITORY_ROOT / "1_dataset_construction"
PHASE_3_ROOT = REPOSITORY_ROOT / "3_final_model_training_and_inference"
STAGES = ("phase1", "phase2", "phase3")
PHASE_2_SCOPES = ("selection_only", "complete")
FAULT_MODE_STRATEGIES = ("none", "indicator", "experts")
SIGNAL_COMPRESSION_STRATEGIES = (
    "none",
    "median_only",
    "pca_only",
    "individual_plus_median",
    "individual_plus_pca",
)
FEATURE_CATALOG_METADATA_COLUMNS = (
    "feature_name",
    "channel",
    "channel_role",
    "statistic",
    "window",
)


class ExperimentManagerError(ValueError):
    """Explain an invalid experiment definition or failed pipeline stage."""


def _read_config(path: Path) -> dict[str, Any]:
    try:
        payload = read_experiment_config(path)
    except ExperimentConfigError as error:
        raise ExperimentManagerError(f"Cannot read experiment catalog: {error}") from error
    if not isinstance(payload, dict):
        raise ExperimentManagerError("Experiment catalog must be a TOML table")
    return payload


def _repo_path(value: str, *, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExperimentManagerError(f"{description} must be a non-empty path")
    path = repository_path(REPOSITORY_ROOT, value)
    try:
        path.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ExperimentManagerError(f"{description} escapes the repository") from error
    return path


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ExperimentManagerError(f"Path is outside the repository: {path}") from error


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    table = config.get("paths")
    if not isinstance(table, dict):
        raise ExperimentManagerError("The catalog needs a [paths] table")
    return {
        key: _repo_path(value, description=f"paths.{key}")
        for key, value in table.items()
    }


def _experiments(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = config.get("experiments", {})
    if not isinstance(value, dict):
        raise ExperimentManagerError("experiments must be a TOML table")
    result: dict[str, dict[str, Any]] = {}
    for name, experiment in value.items():
        if not isinstance(name, str) or not name or not isinstance(experiment, dict):
            raise ExperimentManagerError("Every experiment must be a named TOML table")
        result[name] = experiment
    return result


def _experiment(config: dict[str, Any], name: str) -> dict[str, Any]:
    experiment = _experiments(config).get(name)
    if experiment is None:
        available = ", ".join(sorted(_experiments(config)))
        raise ExperimentManagerError(f"Unknown experiment {name!r}; available: {available}")
    return experiment


def _experiment_groups(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = config.get("experiment_groups", {})
    if not isinstance(value, dict):
        raise ExperimentManagerError("experiment_groups must be a TOML table")
    result: dict[str, dict[str, Any]] = {}
    known_experiments = set(_experiments(config))
    for name, group in value.items():
        if not isinstance(name, str) or not name or not isinstance(group, dict):
            raise ExperimentManagerError("Every experiment group must be a named TOML table")
        members = group.get("experiments")
        if (
            not isinstance(members, list)
            or not members
            or not all(isinstance(item, str) and item for item in members)
            or len(members) != len(set(members))
        ):
            raise ExperimentManagerError(
                f"experiment_groups.{name}.experiments must be a non-empty unique string list"
            )
        unknown = sorted(set(members) - known_experiments)
        if unknown:
            raise ExperimentManagerError(
                f"experiment_groups.{name} references unknown experiments: {unknown}"
            )
        control = group.get("control")
        if not isinstance(control, str) or control not in members:
            raise ExperimentManagerError(
                f"experiment_groups.{name}.control must name one group experiment"
            )
        reporter = group.get("reporter")
        if not isinstance(reporter, str) or not reporter:
            raise ExperimentManagerError(
                f"experiment_groups.{name}.reporter must be a non-empty path"
            )
        result[name] = group
    return result


def _experiment_group(config: dict[str, Any], name: str) -> dict[str, Any]:
    group = _experiment_groups(config).get(name)
    if group is None:
        available = ", ".join(sorted(_experiment_groups(config)))
        raise ExperimentManagerError(
            f"Unknown experiment group {name!r}; available: {available}"
        )
    return group


def _experiment_workflows(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = config.get("experiment_workflows", {})
    if not isinstance(value, dict):
        raise ExperimentManagerError("experiment_workflows must be a TOML table")
    groups = _experiment_groups(config)
    required_group_fields = (
        "feature_group",
        "cap_group",
        "ensemble_group",
        "safety_group",
    )
    result: dict[str, dict[str, Any]] = {}
    for name, workflow in value.items():
        if not isinstance(name, str) or not name or not isinstance(workflow, dict):
            raise ExperimentManagerError("Every workflow must be a named TOML table")
        for field in required_group_fields:
            group_name = workflow.get(field)
            if not isinstance(group_name, str) or group_name not in groups:
                raise ExperimentManagerError(
                    f"experiment_workflows.{name}.{field} must name a group"
                )
        tolerance = workflow.get("safety_r2_tolerance", 0.005)
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not 0.0 <= float(tolerance) < 1.0
        ):
            raise ExperimentManagerError(
                f"experiment_workflows.{name}.safety_r2_tolerance must be in [0, 1)"
            )
        result[name] = workflow
    return result


def _conditional_calibration_workflows(
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate post-Phase-3 development-only safety calibration workflows."""

    value = config.get("conditional_calibration_workflows", {})
    if not isinstance(value, dict):
        raise ExperimentManagerError(
            "conditional_calibration_workflows must be a TOML table"
        )
    result: dict[str, dict[str, Any]] = {}
    for name, workflow in value.items():
        if not isinstance(name, str) or not name or not isinstance(workflow, dict):
            raise ExperimentManagerError(
                "Every conditional calibration workflow must be a named TOML table"
            )
        source_run = workflow.get("source_phase_3_run")
        if isinstance(source_run, bool) or not isinstance(source_run, int) or source_run <= 0:
            raise ExperimentManagerError(
                f"conditional_calibration_workflows.{name}.source_phase_3_run "
                "must be a positive integer"
            )
        quantiles = workflow.get("quantiles")
        if (
            not isinstance(quantiles, list)
            or not quantiles
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not 0.5 <= float(item) < 1.0
                for item in quantiles
            )
        ):
            raise ExperimentManagerError(
                f"conditional_calibration_workflows.{name}.quantiles must be in [0.5, 1.0)"
            )
        result[name] = workflow
    return result


def run_conditional_calibration_workflow(
    name: str,
    config_path: Path,
    config: dict[str, Any],
) -> None:
    """Dispatch one cross-fitted conditional safety calibration run."""

    if name not in _conditional_calibration_workflows(config):
        available = ", ".join(sorted(_conditional_calibration_workflows(config)))
        raise ExperimentManagerError(
            f"Unknown conditional calibration workflow {name!r}; available: {available}"
        )
    _run_command(
        [
            sys.executable,
            str(MANAGER_DIR / "conditional_safety_calibration.py"),
            "--config",
            str(config_path),
            "--run",
            name,
        ],
        label=f"{name}: conditional safety calibration",
    )
    _collect_figures(name)


def _target_submission_workflows(
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate fixed-architecture target-policy submission workflows."""

    value = config.get("target_submission_workflows", {})
    if not isinstance(value, dict):
        raise ExperimentManagerError("target_submission_workflows must be a TOML table")
    result: dict[str, dict[str, Any]] = {}
    required_variants = {"hard_cap_125", "raw", "weighted_raw", "soft_tail"}
    for name, workflow in value.items():
        if not isinstance(name, str) or not name or not isinstance(workflow, dict):
            raise ExperimentManagerError(
                "Every target submission workflow must be a named TOML table"
            )
        source_run = workflow.get("source_phase_3_run")
        if isinstance(source_run, bool) or not isinstance(source_run, int) or source_run <= 0:
            raise ExperimentManagerError(
                f"target_submission_workflows.{name}.source_phase_3_run "
                "must be a positive integer"
            )
        variants = workflow.get("variants")
        if not isinstance(variants, dict) or set(variants) != required_variants:
            raise ExperimentManagerError(
                f"target_submission_workflows.{name}.variants must define exactly "
                f"{sorted(required_variants)}"
            )
        result[name] = workflow
    return result


def run_target_submission_workflow(
    name: str,
    config_path: Path,
    config: dict[str, Any],
    *,
    force: bool,
) -> None:
    """Dispatch one fixed-architecture target-policy submission experiment."""

    if name not in _target_submission_workflows(config):
        available = ", ".join(sorted(_target_submission_workflows(config)))
        raise ExperimentManagerError(
            f"Unknown target submission workflow {name!r}; available: {available}"
        )
    command = [
        sys.executable,
        str(MANAGER_DIR / "target_policy_submissions.py"),
        "--config",
        str(config_path),
        "--run",
        name,
    ]
    if force:
        command.append("--force")
    _run_command(command, label=f"{name}: target-policy submissions")
    _collect_figures(name)


def _promotions(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = config.get("promotions", {})
    if not isinstance(value, dict):
        raise ExperimentManagerError("promotions must be a TOML table")
    workflows = _experiment_workflows(config)
    result: dict[str, dict[str, Any]] = {}
    for name, promotion in value.items():
        if not isinstance(name, str) or not name or not isinstance(promotion, dict):
            raise ExperimentManagerError("Every promotion must be a named TOML table")
        workflow = promotion.get("workflow")
        if not isinstance(workflow, str) or workflow not in workflows:
            raise ExperimentManagerError(
                f"promotions.{name}.workflow must name an experiment workflow"
            )
        ensemble_group = promotion.get("ensemble_group")
        if not isinstance(ensemble_group, str) or ensemble_group not in _experiment_groups(config):
            raise ExperimentManagerError(
                f"promotions.{name}.ensemble_group must name an experiment group"
            )
        result[name] = promotion
    return result


def run_promotion(
    name: str,
    config: dict[str, Any],
    *,
    force: bool,
) -> dict[str, Any]:
    """Run or resume one catalogued locked confirmation policy."""

    _promotions(config)[name]
    configured_workers = _configured_max_workers(config)
    max_workers = (
        max(1, os.cpu_count() or 1)
        if configured_workers == "auto"
        else int(configured_workers)
    )
    try:
        manifest = run_calibrated_ensemble_promotion(
            config,
            name,
            force=force,
            max_workers=max_workers,
        )
    except PromotionError as error:
        raise ExperimentManagerError(str(error)) from error
    workflow = str(_promotions(config)[name]["workflow"])
    _collect_existing_figures(config, workflow)
    return manifest


def _configured_max_workers(config: dict[str, Any]) -> int | str:
    """Read the global execution worker limit from the experiment catalog."""

    execution = config.get("execution")
    if not isinstance(execution, dict):
        raise ExperimentManagerError("The catalog needs an [execution] table")
    value = execution.get("max_workers")
    if value == "auto":
        return value
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ExperimentManagerError(
            "execution.max_workers must be a positive integer or 'auto'"
        )
    return value


def _phase2_scope(experiment: dict[str, Any]) -> str:
    """Return the requested Phase 2 protocol boundary."""

    scope = experiment.get("phase_2_scope", "complete")
    if scope not in PHASE_2_SCOPES:
        allowed = ", ".join(PHASE_2_SCOPES)
        raise ExperimentManagerError(
            f"phase_2_scope must be one of: {allowed}"
        )
    return str(scope)


def _run_dir(
    name: str,
    specification: dict[str, Any] | None = None,
) -> Path:
    try:
        return artifact_directory(RUNS_DIR, name, specification)
    except ValueError as error:
        raise ExperimentManagerError(str(error)) from error


def _state_path(name: str, experiment: dict[str, Any]) -> Path:
    return _run_dir(name, experiment) / "experiment_status.json"


def _phase3_figure_dir(experiment: dict[str, Any]) -> Path | None:
    if not experiment.get("phase_3_enabled", False):
        return None
    run_number = experiment.get("phase_3_run_number")
    if isinstance(run_number, bool) or not isinstance(run_number, int):
        return None
    return (
        PHASE_3_ROOT
        / "runs"
        / f"run_{run_number}"
        / "7_post_run_reporting"
        / "figures"
    )


def _collect_figures(
    name: str,
    experiment: dict[str, Any] | None = None,
    *,
    related_experiments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Refresh one run's flat figure gallery without moving canonical outputs."""

    figure_dir = gallery_directory(RUNS_DIR, name, experiment)
    owner_dir = figure_dir.parent
    manifest_path = figure_dir / "figure_manifest.json"
    figure_dir.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[Path, str]] = []
    if owner_dir.is_dir():
        for path in sorted(owner_dir.rglob("*.png")):
            if figure_dir in path.parents:
                continue
            if any(
                parent.name == "figures"
                and (parent / "figure_manifest.json").is_file()
                for parent in path.parents
                if parent != figure_dir
            ):
                continue
            relative_parent = path.relative_to(owner_dir).parent
            label = "__".join(
                part for part in relative_parent.parts if part != "figures"
            )
            sources.append((path, label or "run"))

    phase3_specs = [experiment] if experiment is not None else []
    phase3_specs.extend(related_experiments or [])
    observed_phase3_runs: set[int] = set()
    for phase3_spec in phase3_specs:
        run_number = phase3_spec.get("phase_3_run_number")
        if isinstance(run_number, bool) or not isinstance(run_number, int):
            continue
        if run_number in observed_phase3_runs:
            continue
        observed_phase3_runs.add(run_number)
        phase3_dir = _phase3_figure_dir(phase3_spec)
        if phase3_dir is not None and phase3_dir.is_dir():
            sources.extend(
                (path, f"phase3_run_{run_number}")
                for path in sorted(phase3_dir.glob("*.png"))
            )

    name_counts: dict[str, int] = {}
    for source, _ in sources:
        key = source.name.casefold()
        name_counts[key] = name_counts.get(key, 0) + 1

    previous_files: set[str] = set()
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_files = {
                str(item["file"])
                for item in previous.get("figures", [])
                if isinstance(item, dict) and isinstance(item.get("file"), str)
            }
        except (OSError, json.JSONDecodeError, AttributeError):
            previous_files = set()

    records: list[dict[str, str]] = []
    current_files: set[str] = set()
    for source, label in sources:
        filename = source.name
        if name_counts[source.name.casefold()] > 1:
            filename = f"{label}__{source.name}"
        destination = figure_dir / filename
        shutil.copy2(source, destination)
        relative_destination = destination.relative_to(owner_dir).as_posix()
        current_files.add(relative_destination)
        records.append(
            {
                "file": relative_destination,
                "source": _repo_relative(source),
            }
        )

    for relative_path in previous_files - current_files:
        stale_path = (owner_dir / relative_path).resolve()
        try:
            stale_path.relative_to(figure_dir.resolve())
        except ValueError:
            continue
        if stale_path.is_file():
            stale_path.unlink()

    manifest = {
        "status": "complete",
        "pipeline_experiment": pipeline_owner(name, experiment)[0],
        "pipeline_run": pipeline_run_name(name, experiment),
        "figures": records,
    }
    _write_json(manifest, manifest_path)
    print(f"{name}: collected {len(records)} plot(s) in {figure_dir}", flush=True)
    return manifest


def _collect_existing_figures(config: dict[str, Any], name: str | None = None) -> None:
    experiments = _experiments(config)
    groups = _experiment_groups(config)
    promotions = _promotions(config)
    definitions = config.get("run_definitions", {})
    if not isinstance(definitions, dict):
        definitions = {}
    workflows = _experiment_workflows(config)
    conditional_workflows = _conditional_calibration_workflows(config)

    def owned_by(owner: tuple[str | None, str]) -> list[dict[str, Any]]:
        return [
            experiment
            for experiment_name, experiment in experiments.items()
            if pipeline_owner(experiment_name, experiment) == owner
        ]

    if name is not None:
        if name in experiments:
            experiment = experiments[name]
            owner = pipeline_owner(name, experiment)
            _collect_figures(
                name,
                experiment,
                related_experiments=owned_by(owner),
            )
            return
        if name in groups:
            group = groups[name]
            owner = pipeline_owner(name, group)
            _collect_figures(
                name,
                group,
                related_experiments=owned_by(owner),
            )
            return
        if name in promotions:
            workflow = str(promotions[name]["workflow"])
            workflow_spec = workflows[workflow]
            _collect_figures(
                workflow,
                workflow_spec,
                related_experiments=owned_by(
                    pipeline_owner(workflow, workflow_spec)
                ),
            )
            return
        parent_specification = workflows.get(name)
        if not isinstance(parent_specification, dict):
            parent_specification = conditional_workflows.get(name)
        if not isinstance(parent_specification, dict):
            candidate = definitions.get(name)
            parent_specification = candidate if isinstance(candidate, dict) else None
        if isinstance(parent_specification, dict) and run_directory(
            RUNS_DIR,
            name,
            parent_specification,
        ).is_dir():
            owner = pipeline_owner(name, parent_specification)
            _collect_figures(
                name,
                parent_specification,
                related_experiments=owned_by(owner),
            )
            return
        raise ExperimentManagerError(f"Unknown experiment or group {name!r}")

    owners: dict[tuple[str | None, str], tuple[str, dict[str, Any]]] = {}
    for experiment_name, experiment in experiments.items():
        owners.setdefault(
            pipeline_owner(experiment_name, experiment),
            (experiment_name, experiment),
        )
    for workflow_name, workflow in workflows.items():
        owners.setdefault(
            pipeline_owner(workflow_name, workflow),
            (workflow_name, workflow),
        )
    for workflow_name, workflow in conditional_workflows.items():
        owners.setdefault(
            pipeline_owner(workflow_name, workflow),
            (workflow_name, workflow),
        )
    for definition_name, definition in definitions.items():
        if isinstance(definition, dict):
            owners.setdefault(
                pipeline_owner(definition_name, definition),
                (definition_name, definition),
            )
    for owner, (run_name, specification) in sorted(
        owners.items(),
        key=lambda item: (str(item[0][0]), item[0][1]),
    ):
        if not run_directory(RUNS_DIR, run_name, specification).is_dir():
            continue
        _collect_figures(
            run_name,
            specification,
            related_experiments=owned_by(owner),
        )


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentManagerError(f"Cannot read {description} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExperimentManagerError(f"{description} must contain a JSON object")
    return value


def _run_command(command: list[str], *, label: str) -> None:
    print("", flush=True)
    print("=" * 78, flush=True)
    print(label, flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    print("=" * 78, flush=True)
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if completed.returncode != 0:
        raise ExperimentManagerError(f"{label} failed with exit code {completed.returncode}")


def _load_interface(experiment: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    phase1_run = experiment.get("phase_1_run_name")
    variant = experiment.get("prefix_variant")
    if not isinstance(phase1_run, str) or not phase1_run:
        raise ExperimentManagerError("phase_1_run_name must be configured")
    if not isinstance(variant, str) or not variant:
        raise ExperimentManagerError("prefix_variant must be configured")
    path = PHASE_1_ROOT / "runs" / phase1_run / variant / "phase_2_interface.json"
    return path, _read_json(path, "Phase 1 interface")


def _csv_observation(path: Path) -> tuple[int, int, int]:
    try:
        table = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise ExperimentManagerError(f"Cannot inspect Phase 1 CSV {path}: {error}") from error
    feature_columns = sum(str(column).startswith("feature__") for column in table.columns)
    return len(table), len(table.columns), feature_columns


def _update_feature_catalog_contract(
    phase1: dict[str, Any],
    expected_sets: dict[str, Any],
) -> None:
    """Bind the template artifact contract to this Phase 1 feature catalog."""

    try:
        catalog = phase1["artifacts"]["feature_catalog"]
    except (KeyError, TypeError) as error:
        raise ExperimentManagerError(
            "Pipeline Phase 2 settings have no feature_catalog artifact contract"
        ) from error
    catalog["required_columns"] = [
        *FEATURE_CATALOG_METADATA_COLUMNS,
        *expected_sets,
    ]


def _phase2_settings(
    config: dict[str, Any],
    experiment: dict[str, Any],
    interface: dict[str, Any],
    interface_path: Path,
) -> dict[str, Any]:
    paths = _paths(config)
    settings_path = paths["phase_2_settings"]
    try:
        settings_path.relative_to(MANAGER_DIR.resolve())
    except ValueError as error:
        raise ExperimentManagerError(
            "paths.phase_2_settings must point inside 1_pipeline_experiments"
        ) from error
    try:
        with settings_path.open("rb") as stream:
            settings = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ExperimentManagerError(
            f"Cannot read pipeline experiment Phase 2 settings: {error}"
        ) from error

    models = experiment.get("architectures")
    if not isinstance(models, list) or not models or not all(isinstance(item, str) for item in models):
        raise ExperimentManagerError("architectures must be a non-empty list of model family names")
    known_models = set(settings["architectures"])
    unknown_models = sorted(set(models) - known_models)
    if unknown_models:
        raise ExperimentManagerError(f"Unknown Phase 2 architectures: {unknown_models}")
    feature_set = experiment.get("feature_set")
    expected_sets = interface.get("expected_feature_sets")
    if not isinstance(expected_sets, dict):
        raise ExperimentManagerError("Phase 1 interface has no expected_feature_sets object")
    if not isinstance(feature_set, str) or feature_set not in expected_sets:
        raise ExperimentManagerError(
            f"feature_set must be one of {sorted(expected_sets)}"
        )

    try:
        settings["settings_version"] = int(experiment["phase_2_settings_version"])
        settings["run_number"] = int(experiment["phase_2_run_number"])
    except (KeyError, TypeError, ValueError) as error:
        raise ExperimentManagerError(
            "phase_2_settings_version and phase_2_run_number must be integers"
        ) from error
    settings["execution"]["max_workers"] = _configured_max_workers(config)
    settings["study"]["enabled"] = {
        family: family in models for family in settings["architectures"]
    }
    settings["tuning"]["candidate_budget_per_architecture"] = int(
        experiment.get("candidate_budget", settings["tuning"]["candidate_budget_per_architecture"])
    )
    settings["tuning"]["search_seed"] = int(
        experiment.get("search_seed", settings["tuning"]["search_seed"])
    )
    settings["tuning"]["retraining_seeds"] = list(
        experiment.get("retraining_seeds", settings["tuning"]["retraining_seeds"])
    )

    sequence_lookbacks = experiment.get("sequence_lookbacks")
    if sequence_lookbacks is not None:
        if (
            not isinstance(sequence_lookbacks, list)
            or not sequence_lookbacks
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in sequence_lookbacks
            )
            or len(set(sequence_lookbacks)) != len(sequence_lookbacks)
        ):
            raise ExperimentManagerError(
                "sequence_lookbacks must be a non-empty list of unique positive integers"
            )
        settings["representations"]["sequence_lookbacks"] = list(sequence_lookbacks)
        for architecture in settings["architectures"].values():
            if architecture.get("representation") in {"sequence", "heterogeneous"}:
                architecture["lookbacks"] = list(sequence_lookbacks)

    sequence_channels = experiment.get("sequence_channels")
    if sequence_channels is not None:
        if (
            not isinstance(sequence_channels, list)
            or not sequence_channels
            or not all(isinstance(value, str) and value for value in sequence_channels)
            or len(set(sequence_channels)) != len(sequence_channels)
        ):
            raise ExperimentManagerError(
                "sequence_channels must be a non-empty list of unique names"
            )
        settings["representations"]["sequence_channels"] = list(sequence_channels)
        settings["representations"]["sequence_channel_count"] = len(sequence_channels)

    neural_override = experiment.get("neural_training")
    if neural_override is not None:
        expected_neural_fields = {
            "batch_size",
            "maximum_epochs",
            "early_stopping_patience",
            "gradient_clip_global_norm",
        }
        if not isinstance(neural_override, dict) or set(neural_override) - expected_neural_fields:
            raise ExperimentManagerError(
                "neural_training contains unsupported fields"
            )
        settings["neural_training"].update(copy.deepcopy(neural_override))

    fixed_hyperparameters = experiment.get("fixed_hyperparameters", {})
    if not isinstance(fixed_hyperparameters, dict):
        raise ExperimentManagerError("fixed_hyperparameters must be a table")
    for family, values in fixed_hyperparameters.items():
        architecture = settings["architectures"].get(family)
        if not isinstance(architecture, dict) or family not in models:
            raise ExperimentManagerError(
                f"Fixed hyperparameters reference unavailable family {family!r}"
            )
        search = architecture.get("search")
        if not isinstance(values, dict) or not isinstance(search, dict):
            raise ExperimentManagerError(
                f"fixed_hyperparameters.{family} must be a table"
            )
        unknown = sorted(set(values) - set(search))
        if unknown:
            raise ExperimentManagerError(
                f"Unknown fixed hyperparameters for {family}: {unknown}"
            )
        for parameter, value in values.items():
            search[parameter] = {"kind": "fixed", "value": copy.deepcopy(value)}
    strategy_defaults = {
        "fault_mode_strategy": "none",
        "signal_compression_strategy": "none",
    }
    for parameter, default in strategy_defaults.items():
        value = str(experiment.get(parameter, default))
        allowed = (
            FAULT_MODE_STRATEGIES
            if parameter == "fault_mode_strategy"
            else SIGNAL_COMPRESSION_STRATEGIES
        )
        if value not in allowed:
            raise ExperimentManagerError(
                f"{parameter} must be one of: {', '.join(allowed)}"
            )
        for family in ("extra_trees", "xgboost"):
            settings["architectures"][family]["search"][parameter] = {
                "kind": "fixed",
                "value": value,
            }

    target_profiles = config.get("target_profiles", {})
    prediction_profiles = config.get("prediction_profiles", {})
    target_name = experiment.get("target_profile", "raw")
    prediction_name = experiment.get("prediction_profile", "symmetric")
    try:
        settings["target"] = copy.deepcopy(target_profiles[target_name])
        settings["prediction_policy"] = copy.deepcopy(prediction_profiles[prediction_name])
    except (KeyError, TypeError) as error:
        raise ExperimentManagerError(
            f"Unknown target or prediction profile: {target_name!r}, {prediction_name!r}"
        ) from error

    settings["evaluation"]["metrics"] = list(
        experiment.get(
            "metrics",
            [
                "r2",
                "rmse",
                "mae",
                "bias",
                "overprediction_rate",
                "mean_overprediction",
                "root_mean_squared_overprediction",
                "underprediction_rate",
                "mean_underprediction",
            ],
        )
    )
    settings["evaluation"]["reported_groups"] = list(
        experiment.get(
            "reported_groups",
            ["overall", "scenario", "outer_fold", "age_band", "lifetime_quantile", "rul_band"],
        )
    )

    phase1 = settings["phase_1"]
    phase1["expected_feature_sets"] = copy.deepcopy(expected_sets)
    phase1["expected_generated_features"] = int(interface["expected_generated_features"])
    _update_feature_catalog_contract(phase1, expected_sets)
    for key in (
        "expected_prefixes_per_training_uav",
        "minimum_prefixes_per_training_uav",
        "maximum_prefixes_per_training_uav",
    ):
        phase1.pop(key, None)
    for key in (
        "expected_prefixes_per_training_uav",
        "minimum_prefixes_per_training_uav",
        "maximum_prefixes_per_training_uav",
    ):
        if key in interface:
            phase1[key] = interface[key]

    artifacts = interface.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ExperimentManagerError("Phase 1 interface has no artifacts object")
    phase1["expected_training_uavs"] = 100
    phase1["expected_outer_folds"] = 5
    phase1["expected_inner_folds_per_outer_fold"] = 4
    for name, artifact in artifacts.items():
        if name not in phase1["artifacts"] or not isinstance(artifact, str):
            raise ExperimentManagerError(f"Phase 1 interface has an invalid artifact {name!r}")
        target = phase1["artifacts"][name]
        target["path"] = artifact
        path = _repo_path(artifact, description=f"Phase 1 artifact {name}")
        if target["format"] == "csv":
            rows, columns, feature_columns = _csv_observation(path)
            target["rows"] = rows
            target["total_columns"] = columns
            if "feature_columns" in target:
                target["feature_columns"] = feature_columns

    scenario_name = experiment.get("scenario_profile")
    scenario_profiles = config.get("scenario_profiles")
    if (
        not isinstance(scenario_name, str)
        or not isinstance(scenario_profiles, dict)
        or not isinstance(scenario_profiles.get(scenario_name), dict)
    ):
        raise ExperimentManagerError(
            f"Unknown scenario_profile: {scenario_name!r}"
        )
    scenario_config = _read_json(
        _repo_path(
            artifacts["scenario_config"],
            description="Phase 1 scenario config",
        ),
        "Phase 1 scenario config",
    )
    expected_scenario = scenario_profiles[scenario_name]
    for key in (
        "assignment",
        "development_scenarios",
        "locked_scenarios",
        "seed",
        "minimum_rul",
        "maximum_rul",
    ):
        expected_value = expected_scenario.get(key)
        if scenario_config.get(key) != expected_value:
            raise ExperimentManagerError(
                f"Phase 1 scenario config does not match "
                f"scenario_profiles.{scenario_name}.{key}: "
                f"observed {scenario_config.get(key)!r}, "
                f"expected {expected_value!r}"
            )

    outer = pd.read_csv(_repo_path(artifacts["outer_folds"], description="outer folds"))
    inner = pd.read_csv(_repo_path(artifacts["inner_folds"], description="inner folds"))
    development = pd.read_csv(
        _repo_path(artifacts["development_scenarios"], description="development scenarios")
    )
    locked = pd.read_csv(
        _repo_path(artifacts["locked_scenarios"], description="locked scenarios")
    )
    phase1["expected_training_uavs"] = int(outer["uav_id"].nunique())
    phase1["expected_outer_folds"] = int(outer["outer_fold"].nunique())
    per_outer = inner.groupby("outer_fold")["inner_fold"].nunique()
    if per_outer.empty or per_outer.nunique() != 1:
        raise ExperimentManagerError("Inner-fold counts are not equal across outer folds")
    phase1["expected_inner_folds_per_outer_fold"] = int(per_outer.iloc[0])
    phase1["expected_development_scenarios"] = int(development["scenario"].nunique())
    phase1["expected_locked_scenarios"] = int(locked["scenario"].nunique())

    required_values = phase1["artifacts"]["verification_report"]["required_json_values"]
    required_values["development_scenarios"] = phase1["expected_development_scenarios"]
    required_values["locked_scenarios"] = phase1["expected_locked_scenarios"]
    required_values = phase1["artifacts"]["scenario_config"]["required_json_values"]
    required_values["development_scenarios"] = phase1["expected_development_scenarios"]
    required_values["locked_scenarios"] = phase1["expected_locked_scenarios"]
    prefix_values = phase1["artifacts"]["training_prefix_config"]["required_json_values"]
    prefix_values.pop("cutoffs_per_uav", None)
    if "expected_prefixes_per_training_uav" in phase1:
        prefix_values["cutoffs_per_uav"] = phase1["expected_prefixes_per_training_uav"]

    settings["representations"]["tabular_feature_sets"] = list(expected_sets)
    for family, architecture in settings["architectures"].items():
        if architecture["representation"] == "tabular":
            cycle_baseline_set = (
                "age_only" if "age_only" in expected_sets else feature_set
            )
            architecture["feature_sets"] = (
                [cycle_baseline_set]
                if family == "cycle_only_baseline"
                else [feature_set]
            )

    return settings


def _phase1_interface_path(experiment: dict[str, Any]) -> Path:
    phase1_run = experiment.get("phase_1_run_name")
    variant = experiment.get("prefix_variant")
    return PHASE_1_ROOT / "runs" / str(phase1_run) / str(variant) / "phase_2_interface.json"


def _run_phase1(
    name: str,
    config: dict[str, Any],
    experiment: dict[str, Any],
) -> None:
    paths = _paths(config)
    # Phase 1 reads TOML/JSON directly and does not resolve the pipeline
    # experiment ``include`` chain. Persist the fully merged catalog so the
    # exact profile, scenario, and prefix definitions used by this cell remain
    # reviewable with its artifacts.
    resolved_settings_path = (
        artifact_directory(RUNS_DIR, name, experiment)
        / "resolved_phase_1_settings.json"
    )
    _write_json(config, resolved_settings_path)
    command = [
        sys.executable,
        str(paths["phase_1_runner"]),
        "--settings",
        str(resolved_settings_path),
        "--profile",
        str(experiment["phase_1_profile"]),
        "--run-name",
        str(experiment["phase_1_run_name"]),
        "--prefix-variant",
        str(experiment["prefix_variant"]),
    ]
    if isinstance(experiment.get("scenario_profile"), str):
        command.extend(["--scenario-profile", experiment["scenario_profile"]])
    if experiment.get("phase_1_mode", "rebuild") == "reuse":
        command.append("--refresh-interface")
    _run_command(command, label=f"{name}: Phase 1")
    interface_path = _phase1_interface_path(experiment)
    _read_json(interface_path, "Phase 1 interface")


def _phase2_paths(
    name: str,
    experiment: dict[str, Any] | None = None,
) -> dict[str, Path]:
    root = _run_dir(name, experiment) / "phase2"
    return {
        "root": root,
        "settings": root / "phase_2_settings.json",
        "specification": root / "1_architecture_study_settings" / "artifacts" / "experiment_specification.json",
        "tabular_manifest": root / "2_tabular_data_adapter" / "artifacts" / "tabular_dataset_manifest.json",
        "sequence_manifest": root / "3_sequence_data_adapter" / "artifacts" / "sequence_dataset_manifest.json",
        "trajectory_manifest": root / "3_trajectory_data_adapter" / "artifacts" / "trajectory_dataset_manifest.json",
        "registry": root / "4_model_adapters" / "artifacts" / "model_registry.json",
    }


def _run_phase2(
    name: str,
    config: dict[str, Any],
    experiment: dict[str, Any],
    *,
    force: bool,
) -> None:
    paths = _paths(config)
    interface_path, interface = _load_interface(experiment)
    phase2 = _phase2_paths(name, experiment)
    settings = _phase2_settings(config, experiment, interface, interface_path)
    _write_json(settings, phase2["settings"])

    _run_command(
        [sys.executable, str(paths["phase_2_settings_builder"]), "--settings", str(phase2["settings"]), "--output-dir", str(phase2["specification"].parent)],
        label=f"{name}: Phase 2 Step 1 settings",
    )
    _run_command(
        [sys.executable, str(paths["tabular_adapter_builder"]), "--specification", str(phase2["specification"]), "--output-dir", str(phase2["tabular_manifest"].parent)],
        label=f"{name}: Phase 2 Step 2 tabular adapter",
    )
    _run_command(
        [
            sys.executable,
            str(paths["sequence_adapter_builder"]),
            "--specification",
            str(phase2["specification"]),
            "--output-dir",
            str(phase2["sequence_manifest"].parent),
            "--test-endpoints",
            _repo_relative(
                PHASE_1_ROOT
                / "runs"
                / str(experiment["phase_1_run_name"])
                / "3_test_like_validation_scenarios"
                / "artifacts"
                / "test_endpoints.csv"
            ),
        ],
        label=f"{name}: Phase 2 Step 3 sequence adapter",
    )
    _run_command(
        [sys.executable, str(paths["trajectory_adapter_builder"]), "--specification", str(phase2["specification"]), "--sequence-manifest", str(phase2["sequence_manifest"]), "--sequence-report", str(phase2["sequence_manifest"].parent / "copy_verification.json"), "--output-dir", str(phase2["trajectory_manifest"].parent)],
        label=f"{name}: Phase 2 Step 3b trajectory adapter",
    )
    _run_command(
        [sys.executable, str(paths["model_registry_builder"]), "--specification", str(phase2["specification"]), "--output-dir", str(phase2["registry"].parent)],
        label=f"{name}: Phase 2 Step 4 model registry",
    )

    scope = _phase2_scope(experiment)
    run_root = phase2["root"]
    step6 = run_root / "6_locked_outer_evaluation"
    step7 = run_root / "7_architecture_comparison"
    through_step = "5" if scope == "selection_only" else "6"
    phase_2_command = [
        sys.executable,
        str(paths["phase_2_orchestrator"]),
        "--specification",
        str(phase2["specification"]),
        "--from-step",
        "5",
        "--through-step",
        through_step,
        "--tabular-manifest",
        str(phase2["tabular_manifest"]),
        "--sequence-manifest",
        str(phase2["sequence_manifest"]),
        "--trajectory-manifest",
        str(phase2["trajectory_manifest"]),
        "--model-registry",
        str(phase2["registry"]),
        "--run-root",
        str(run_root),
    ]
    if force:
        phase_2_command.append("--force")
    _run_command(
        phase_2_command,
        label=(
            f"{name}: Phase 2 Step 5 development selection"
            if scope == "selection_only"
            else f"{name}: Phase 2 Steps 5-6 selection and locked evaluation"
        ),
    )
    if scope == "selection_only":
        print(
            f"{name}: selection_only gate reached; locked Steps 6-7 were not run",
            flush=True,
        )
        return

    _run_command(
        [
            sys.executable,
            str(paths["phase_2_comparison"]),
            "--specification",
            str(phase2["specification"]),
            "--locked-manifest",
            str(step6 / "locked_evaluation_manifest.json"),
            "--output-dir",
            str(step7),
        ],
        label=f"{name}: Phase 2 Step 7 comparison",
    )


def _phase3_settings(name: str, config: dict[str, Any], experiment: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    selected = experiment.get("phase_3_selected_model_family")
    architectures = experiment.get("architectures", [])
    if selected not in architectures:
        raise ExperimentManagerError(
            "phase_3_selected_model_family must be one of the Phase 2 architectures"
        )
    phase2 = _phase2_paths(name, experiment)
    payload = {
        "settings_version": int(experiment.get("phase_3_settings_version", 1)),
        "run_number": int(experiment["phase_3_run_number"]),
        "phase_2_run_number": int(experiment["phase_2_run_number"]),
        "selected_model_family": selected,
        "phase_2_run_root": _repo_relative(phase2["root"]),
        "phase_2_specification": _repo_relative(phase2["specification"]),
        "phase_2_model_registry": _repo_relative(phase2["registry"]),
        "tabular_manifest": _repo_relative(phase2["tabular_manifest"]),
        "sequence_manifest": _repo_relative(phase2["sequence_manifest"]),
        "trajectory_manifest": _repo_relative(phase2["trajectory_manifest"]),
        "final_search": {
            "candidate_budget": int(experiment.get("phase_3_candidate_budget", 50)),
            "search_seed": int(experiment.get("phase_3_search_seed", 13)),
            "model_seed": int(experiment.get("phase_3_model_seed", 13)),
        },
    }
    path = _run_dir(name, experiment) / "phase3_settings.json"
    _write_json(payload, path)
    return path, payload


def _run_phase3(name: str, config: dict[str, Any], experiment: dict[str, Any], force: bool) -> None:
    paths = _paths(config)
    settings_path, _ = _phase3_settings(name, config, experiment)
    command = [
        sys.executable,
        str(paths["phase_3_runner"]),
        "--settings",
        str(settings_path),
    ]
    if force:
        command.append("--force")
    _run_command(command, label=f"{name}: Phase 3")


def _state(name: str, experiment: dict[str, Any]) -> dict[str, Any]:
    path = _state_path(name, experiment)
    if path.is_file():
        return _read_json(path, "experiment status")
    return {
        "experiment": name,
        "phase_1_run_name": experiment.get("phase_1_run_name"),
        "phase_2_run_number": experiment.get("phase_2_run_number"),
        "phase_3_run_number": experiment.get("phase_3_run_number"),
        "stages": {},
    }


def _mark_stage(name: str, experiment: dict[str, Any], stage: str, status: str) -> None:
    payload = _state(name, experiment)
    payload.setdefault("stages", {})[stage] = status
    _write_json(payload, _state_path(name, experiment))


def _stage_complete(name: str, experiment: dict[str, Any], stage: str) -> bool:
    state = _state(name, experiment)
    if state.get("stages", {}).get(stage) != "complete":
        return False
    if stage == "phase1":
        return _phase1_interface_path(experiment).is_file()
    if stage == "phase2":
        root = _phase2_paths(name, experiment)["root"]
        if _phase2_scope(experiment) == "selection_only":
            path = root / "5_inner_model_selection" / "selection_manifest.json"
        else:
            path = root / "7_architecture_comparison" / "comparison_manifest.json"
    else:
        path = PHASE_3_ROOT / "runs" / f"run_{int(experiment['phase_3_run_number'])}" / "7_post_run_reporting" / "report_manifest.json"
    return path.is_file()


def _resolve_stage(value: Any, *, key: str, default: str) -> str:
    stage = default if value is None else value
    if stage not in STAGES:
        allowed = ", ".join(STAGES)
        raise ExperimentManagerError(f"{key} must be one of: {allowed}")
    return stage


def run_experiment(
    name: str,
    config: dict[str, Any],
    *,
    from_stage: str,
    through_stage: str,
    force: bool,
) -> None:
    experiment = _experiment(config, name)
    from_stage = _resolve_stage(from_stage, key="from_stage", default="phase1")
    through_stage = _resolve_stage(through_stage, key="through_stage", default="phase2")
    first = STAGES.index(from_stage)
    last = STAGES.index(through_stage)
    if first > last:
        raise ExperimentManagerError("from_stage cannot come after through_stage")
    if not experiment.get("enabled", True):
        raise ExperimentManagerError(f"Experiment {name!r} is disabled in the catalog")

    for stage in STAGES[first : last + 1]:
        if stage == "phase3" and not experiment.get("phase_3_enabled", False):
            raise ExperimentManagerError(
                f"{name!r} has phase_3_enabled = false; enable it before running Phase 3"
            )
        if not force and _stage_complete(name, experiment, stage):
            print(f"{name}: {stage} already complete; resuming after it")
            owner = pipeline_owner(name, experiment)
            related = [
                specification
                for experiment_name, specification in _experiments(config).items()
                if pipeline_owner(experiment_name, specification) == owner
            ]
            _collect_figures(
                name,
                experiment,
                related_experiments=related,
            )
            continue
        _mark_stage(name, experiment, stage, "running")
        try:
            if stage == "phase1":
                _run_phase1(name, config, experiment)
            elif stage == "phase2":
                _run_phase2(name, config, experiment, force=force)
            else:
                _run_phase3(name, config, experiment, force)
        except KeyboardInterrupt:
            _mark_stage(name, experiment, stage, "interrupted")
            raise
        except Exception:
            _mark_stage(name, experiment, stage, "failed")
            raise
        _mark_stage(name, experiment, stage, "complete")
        owner = pipeline_owner(name, experiment)
        related = [
            specification
            for experiment_name, specification in _experiments(config).items()
            if pipeline_owner(experiment_name, specification) == owner
        ]
        _collect_figures(
            name,
            experiment,
            related_experiments=related,
        )
        print(f"{name}: {stage} complete")


def run_experiment_group(
    name: str,
    config_path: Path,
    config: dict[str, Any],
    *,
    force: bool,
    report_config_path: Path | None = None,
) -> None:
    """Run an explicit ordered experiment group and then build its report."""

    group = _experiment_group(config, name)
    for experiment_name in group["experiments"]:
        experiment = _experiment(config, experiment_name)
        from_stage = _resolve_stage(
            experiment.get("from_stage"),
            key=f"experiments.{experiment_name}.from_stage",
            default="phase1",
        )
        through_stage = _resolve_stage(
            experiment.get("through_stage"),
            key=f"experiments.{experiment_name}.through_stage",
            default="phase2",
        )
        run_experiment(
            experiment_name,
            config,
            from_stage=from_stage,
            through_stage=through_stage,
            force=force,
        )

    reporter = _repo_path(
        group["reporter"],
        description=f"experiment_groups.{name}.reporter",
    )
    _run_command(
        [
            sys.executable,
            str(reporter),
            "--config",
            str(report_config_path or config_path),
            "--group",
            name,
            "--output-dir",
            str(_run_dir(name, group) / "reporting"),
        ],
        label=f"{name}: paired ablation report",
    )
    owner = pipeline_owner(name, group)
    related = [
        specification
        for experiment_name, specification in _experiments(config).items()
        if pipeline_owner(experiment_name, specification) == owner
    ]
    _collect_figures(name, group, related_experiments=related)


def _reporting_directory(config: dict[str, Any], group_name: str) -> Path:
    group = _experiment_group(config, group_name)
    return _run_dir(group_name, group) / "reporting"


def _read_report_table(path: Path, required: set[str]) -> pd.DataFrame:
    try:
        table = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise ExperimentManagerError(f"Cannot read workflow report {path}: {error}") from error
    missing = sorted(required - set(table.columns))
    if missing:
        raise ExperimentManagerError(f"Workflow report {path} is missing {missing}")
    if table.empty:
        raise ExperimentManagerError(f"Workflow report {path} is empty")
    return table


def _accuracy_winner(config: dict[str, Any], group_name: str) -> dict[str, Any]:
    """Select one experiment by equal-weight model-family development accuracy."""

    path = _reporting_directory(config, group_name) / "paired_summary.csv"
    table = _read_report_table(
        path,
        {"experiment", "model_family", "mean_r2", "mean_rmse"},
    )
    ranking = (
        table.groupby("experiment", as_index=False)
        .agg(
            model_families=("model_family", "nunique"),
            mean_r2=("mean_r2", "mean"),
            mean_rmse=("mean_rmse", "mean"),
        )
        .sort_values(
            ["mean_r2", "mean_rmse", "experiment"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    winner = ranking.iloc[0]
    return {
        "experiment": str(winner["experiment"]),
        "mean_r2": float(winner["mean_r2"]),
        "mean_rmse": float(winner["mean_rmse"]),
        "model_families": int(winner["model_families"]),
        "selection_rule": (
            "highest equal-weight mean development R2 across model families; "
            "then lowest mean RMSE; then lexical experiment name"
        ),
        "summary": _repo_relative(path),
    }


def _safety_winner(
    table: pd.DataFrame,
    *,
    identity_column: str,
    r2_tolerance: float,
) -> dict[str, Any]:
    required = {
        identity_column,
        "mean_r2",
        "mean_rmse",
        "mean_bias",
        "mean_overprediction_rate",
        "mean_rms_overprediction",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ExperimentManagerError(f"Safety comparison is missing {missing}")
    best_r2 = float(table["mean_r2"].max())
    minimum_r2 = best_r2 - float(r2_tolerance)
    eligible = table.loc[table["mean_r2"] >= minimum_r2].copy()
    eligible = eligible.sort_values(
        [
            "mean_rms_overprediction",
            "mean_overprediction_rate",
            "mean_rmse",
            "mean_r2",
            identity_column,
        ],
        ascending=[True, True, True, False, True],
    )
    winner = eligible.iloc[0]
    return {
        "candidate": str(winner[identity_column]),
        "mean_r2": float(winner["mean_r2"]),
        "mean_rmse": float(winner["mean_rmse"]),
        "mean_bias": float(winner["mean_bias"]),
        "mean_overprediction_rate": float(winner["mean_overprediction_rate"]),
        "mean_rms_overprediction": float(winner["mean_rms_overprediction"]),
        "best_observed_r2": best_r2,
        "minimum_eligible_r2": minimum_r2,
        "r2_tolerance": float(r2_tolerance),
        "selection_rule": (
            "within the R2 tolerance of the best candidate, minimize RMS "
            "overprediction; then overprediction rate, RMSE, and maximize R2"
        ),
    }


def _selected_prediction_summary(
    config: dict[str, Any],
    experiment_name: str,
    *,
    model_family: str,
) -> dict[str, Any]:
    """Calculate pooled outer-fold metrics from one selected Step 5 model."""

    experiment = _experiment(config, experiment_name)
    path = (
        _phase2_paths(experiment_name, experiment)["root"]
        / "5_inner_model_selection"
        / "selected_inner_predictions.csv.gz"
    )
    table = _read_report_table(
        path,
        {"outer_fold", "model_family", "observed_rul", "predicted_rul"},
    )
    families = sorted(table["model_family"].astype(str).unique())
    if model_family not in families:
        raise ExperimentManagerError(
            f"Safety experiment {experiment_name} has no {model_family!r} "
            f"predictions; observed {families}"
        )
    table = table.loc[table["model_family"].astype(str) == model_family].copy()
    fold_records: list[dict[str, float]] = []
    for _, rows in table.groupby("outer_fold", sort=True):
        targets = rows["observed_rul"].to_numpy(dtype=float)
        predictions = rows["predicted_rul"].to_numpy(dtype=float)
        residual = predictions - targets
        positive = residual.clip(min=0.0)
        denominator = float(((targets - targets.mean()) ** 2).sum())
        if denominator <= 0.0:
            raise ExperimentManagerError(
                f"Safety experiment {experiment_name} has a constant outer-fold target"
            )
        fold_records.append(
            {
                "mean_r2": 1.0 - float((residual**2).sum()) / denominator,
                "mean_rmse": float((residual**2).mean() ** 0.5),
                "mean_bias": float(residual.mean()),
                "mean_overprediction_rate": float((residual > 0.0).mean()),
                "mean_rms_overprediction": float((positive**2).mean() ** 0.5),
            }
        )
    folds = pd.DataFrame.from_records(fold_records)
    return {
        "experiment": experiment_name,
        "model_family": model_family,
        "outer_folds": len(fold_records),
        **{column: float(folds[column].mean()) for column in fold_records[0]},
        "predictions": _repo_relative(path),
    }


def _write_resolved_workflow_catalog(
    config: dict[str, Any],
    path: Path,
) -> None:
    _write_json(config, path)


def run_experiment_workflow(
    name: str,
    config_path: Path,
    source_config: dict[str, Any],
    *,
    force: bool,
) -> dict[str, Any]:
    """Run a PE_3-style staged selection and propagate each winner."""

    workflows = _experiment_workflows(source_config)
    if name not in workflows:
        available = ", ".join(sorted(workflows))
        raise ExperimentManagerError(
            f"Unknown experiment workflow {name!r}; available: {available}"
        )
    config = copy.deepcopy(source_config)
    workflow = workflows[name]
    workflow_dir = run_directory(RUNS_DIR, name, workflow) / "workflow"
    resolved_path = workflow_dir / "resolved_catalog.json"
    manifest_path = workflow_dir / "selection_manifest.json"
    tolerance = float(workflow.get("safety_r2_tolerance", 0.005))
    selections: dict[str, Any] = {}

    feature_group_name = str(workflow["feature_group"])
    _write_resolved_workflow_catalog(config, resolved_path)
    run_experiment_group(
        feature_group_name,
        config_path,
        config,
        force=force,
        report_config_path=resolved_path,
    )
    feature_selection = _accuracy_winner(config, feature_group_name)
    feature_experiment = _experiment(config, feature_selection["experiment"])
    selected_feature = str(feature_experiment["feature_set"])
    feature_selection["feature_set"] = selected_feature
    selections["feature"] = feature_selection

    cap_group_name = str(workflow["cap_group"])
    cap_group = _experiment_group(config, cap_group_name)
    cap_treatments = [
        experiment_name
        for experiment_name in cap_group["experiments"]
        if experiment_name != cap_group["control"]
    ]
    cap_group["control"] = feature_selection["experiment"]
    cap_group["experiments"] = [
        feature_selection["experiment"],
        *cap_treatments,
    ]
    for experiment_name in cap_treatments:
        _experiment(config, experiment_name)["feature_set"] = selected_feature
    _write_resolved_workflow_catalog(config, resolved_path)
    run_experiment_group(
        cap_group_name,
        config_path,
        config,
        force=force,
        report_config_path=resolved_path,
    )
    cap_selection = _accuracy_winner(config, cap_group_name)
    cap_experiment = _experiment(config, cap_selection["experiment"])
    selected_target = str(cap_experiment["target_profile"])
    cap_selection["target_profile"] = selected_target
    cap_selection["feature_set"] = selected_feature
    selections["target_cap"] = cap_selection

    ensemble_group_name = str(workflow["ensemble_group"])
    ensemble_group = _experiment_group(config, ensemble_group_name)
    ensemble_group["control"] = cap_selection["experiment"]
    ensemble_group["experiments"] = [cap_selection["experiment"]]
    _write_resolved_workflow_catalog(config, resolved_path)
    run_experiment_group(
        ensemble_group_name,
        config_path,
        config,
        force=force,
        report_config_path=resolved_path,
    )
    ensemble_summary_path = _reporting_directory(
        config,
        ensemble_group_name,
    ) / "summary.csv"
    ensemble_summary = _read_report_table(
        ensemble_summary_path,
        {
            "method",
            "mean_r2",
            "mean_rmse",
            "mean_bias",
            "mean_overprediction_rate",
            "mean_rms_overprediction",
        },
    )
    ensemble_accuracy = ensemble_summary.sort_values(
        ["mean_r2", "mean_rmse", "method"],
        ascending=[False, True, True],
    ).iloc[0]
    selections["ensemble_accuracy"] = {
        "method": str(ensemble_accuracy["method"]),
        "mean_r2": float(ensemble_accuracy["mean_r2"]),
        "mean_rmse": float(ensemble_accuracy["mean_rmse"]),
        "source_experiment": cap_selection["experiment"],
        "summary": _repo_relative(ensemble_summary_path),
        "selection_rule": "highest development R2, then lowest RMSE",
    }

    safety_group_name = str(workflow["safety_group"])
    safety_group = _experiment_group(config, safety_group_name)
    severity_treatments = [
        experiment_name
        for experiment_name in safety_group["experiments"]
        if experiment_name != safety_group["control"]
    ]
    safety_group["control"] = cap_selection["experiment"]
    safety_group["experiments"] = [
        cap_selection["experiment"],
        *severity_treatments,
    ]
    for experiment_name in severity_treatments:
        experiment = _experiment(config, experiment_name)
        experiment["feature_set"] = selected_feature
        experiment["target_profile"] = selected_target
    _write_resolved_workflow_catalog(config, resolved_path)
    run_experiment_group(
        safety_group_name,
        config_path,
        config,
        force=force,
        report_config_path=resolved_path,
    )
    generic_safety_summary_path = _reporting_directory(
        config,
        safety_group_name,
    ) / "paired_summary.csv"
    _read_report_table(
        generic_safety_summary_path,
        {"experiment", "mean_r2", "mean_rmse"},
    )
    safety_summary = pd.DataFrame.from_records(
        [
            _selected_prediction_summary(
                config,
                experiment_name,
                model_family="xgboost",
            )
            for experiment_name in safety_group["experiments"]
        ]
    )
    safety_summary_path = workflow_dir / "severity_candidate_summary.csv"
    safety_summary.to_csv(safety_summary_path, index=False)
    safety_selection = _safety_winner(
        safety_summary,
        identity_column="experiment",
        r2_tolerance=tolerance,
    )
    selections["severity_loss"] = {
        **safety_selection,
        "summary": _repo_relative(safety_summary_path),
    }

    ensemble_candidates = ensemble_summary.copy()
    ensemble_candidates["candidate"] = "ensemble:" + ensemble_candidates["method"]
    severity_candidates = safety_summary.copy()
    severity_candidates["candidate"] = "loss:" + severity_candidates["experiment"]
    final_candidates = pd.concat(
        [
            ensemble_candidates[
                [
                    "candidate",
                    "mean_r2",
                    "mean_rmse",
                    "mean_bias",
                    "mean_overprediction_rate",
                    "mean_rms_overprediction",
                ]
            ],
            severity_candidates[
                [
                    "candidate",
                    "mean_r2",
                    "mean_rmse",
                    "mean_bias",
                    "mean_overprediction_rate",
                    "mean_rms_overprediction",
                ]
            ],
        ],
        ignore_index=True,
    )
    final_selection = _safety_winner(
        final_candidates,
        identity_column="candidate",
        r2_tolerance=tolerance,
    )
    selections["final"] = final_selection

    _write_resolved_workflow_catalog(config, resolved_path)
    manifest = {
        "status": "complete",
        "workflow": name,
        "pipeline_experiment": pipeline_owner(name, workflow)[0],
        "pipeline_run": pipeline_run_name(name, workflow),
        "uses_locked_evaluation": False,
        "safety_r2_tolerance": tolerance,
        "selections": selections,
        "resolved_catalog": _repo_relative(resolved_path),
    }
    _write_json(manifest, manifest_path)
    _collect_existing_figures(config, name)
    promotion_name = workflow.get("promotion")
    if promotion_name is not None:
        if not isinstance(promotion_name, str) or promotion_name not in _promotions(config):
            raise ExperimentManagerError(
                f"experiment_workflows.{name}.promotion must name a promotion"
            )
        run_promotion(promotion_name, config, force=force)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def print_status(config: dict[str, Any]) -> None:
    for name, experiment in sorted(_experiments(config).items()):
        state = _state(name, experiment)
        stages = [
            f"{stage}={state.get('stages', {}).get(stage, 'not_started')}"
            for stage in STAGES
        ]
        print(
            f"{name}: phase2_scope={_phase2_scope(experiment)}, "
            + ", ".join(stages)
        )
    workflows = _experiment_workflows(config)
    for name, workflow in sorted(workflows.items()):
        manifest = (
            run_directory(RUNS_DIR, name, workflow)
            / "workflow"
            / "selection_manifest.json"
        )
        status = "complete" if manifest.is_file() else "not_started"
        print(f"{name}: workflow={status}")
    for name, workflow in sorted(_conditional_calibration_workflows(config).items()):
        manifest = (
            run_directory(RUNS_DIR, name, workflow)
            / "conditional_calibration_manifest.json"
        )
        status = "complete" if manifest.is_file() else "not_started"
        print(f"{name}: conditional_calibration={status}")
    for name, workflow in sorted(_target_submission_workflows(config).items()):
        manifest = (
            run_directory(RUNS_DIR, name, workflow)
            / "target_submission_manifest.json"
        )
        status = "complete" if manifest.is_file() else "not_started"
        print(f"{name}: target_submissions={status}")
    for name, promotion in sorted(_promotions(config).items()):
        workflow_name = str(promotion["workflow"])
        workflow = workflows[workflow_name]
        manifest = (
            run_directory(RUNS_DIR, workflow_name, workflow)
            / name
            / "locked_confirmation_manifest.json"
        )
        status = "complete" if manifest.is_file() else "not_started"
        print(f"{name}: locked_confirmation={status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run", dest="run_name")
    parser.add_argument("--group", dest="group_name")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--collect-figures",
        action="store_true",
        help="Refresh top-level figure galleries without rerunning experiments.",
    )
    parser.add_argument("--from-stage", choices=STAGES)
    parser.add_argument("--through-stage", choices=STAGES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        config_path = args.config.resolve()
        config = _read_config(config_path)
        if args.list:
            for name, experiment in sorted(_experiments(config).items()):
                print(f"{name}: {'enabled' if experiment.get('enabled', True) else 'disabled'}")
            for name, group in sorted(_experiment_groups(config).items()):
                print(f"{name}: group ({len(group['experiments'])} experiments)")
            for name in sorted(_experiment_workflows(config)):
                print(f"{name}: automatic workflow")
            for name in sorted(_conditional_calibration_workflows(config)):
                print(f"{name}: conditional calibration workflow")
            for name in sorted(_target_submission_workflows(config)):
                print(f"{name}: target submission workflow")
            for name, promotion in sorted(_promotions(config).items()):
                print(f"{name}: locked promotion for {promotion['workflow']}")
            return
        if args.status:
            print_status(config)
            return
        if args.collect_figures:
            if args.run_name and args.group_name:
                parser.error("declare either --run or --group, not both")
            _collect_existing_figures(config, args.run_name or args.group_name)
            return
        if args.run_name and args.group_name:
            parser.error("declare either --run or --group, not both")
        if not args.run_name and not args.group_name:
            parser.error("--run or --group is required unless --list or --status is used")
        if args.group_name:
            run_experiment_group(
                args.group_name,
                config_path,
                config,
                force=args.force,
            )
            return
        if args.run_name in _experiment_workflows(config):
            if args.from_stage is not None or args.through_stage is not None:
                parser.error("workflow runs do not accept --from-stage or --through-stage")
            run_experiment_workflow(
                args.run_name,
                config_path,
                config,
                force=args.force,
            )
            return
        if args.run_name in _conditional_calibration_workflows(config):
            if args.from_stage is not None or args.through_stage is not None:
                parser.error(
                    "conditional calibration workflows do not accept "
                    "--from-stage or --through-stage"
                )
            run_conditional_calibration_workflow(
                args.run_name,
                config_path,
                config,
            )
            return
        if args.run_name in _target_submission_workflows(config):
            if args.from_stage is not None or args.through_stage is not None:
                parser.error(
                    "target submission workflows do not accept "
                    "--from-stage or --through-stage"
                )
            run_target_submission_workflow(
                args.run_name,
                config_path,
                config,
                force=args.force,
            )
            return
        if args.run_name in _promotions(config):
            if args.from_stage is not None or args.through_stage is not None:
                parser.error("promotion runs do not accept --from-stage or --through-stage")
            run_promotion(args.run_name, config, force=args.force)
            return
        experiment = _experiment(config, args.run_name)
        from_stage = _resolve_stage(
            args.from_stage if args.from_stage is not None else experiment.get("from_stage"),
            key="from_stage",
            default="phase1",
        )
        through_stage = _resolve_stage(
            args.through_stage
            if args.through_stage is not None
            else experiment.get("through_stage"),
            key="through_stage",
            default="phase2",
        )
        run_experiment(
            args.run_name,
            config,
            from_stage=from_stage,
            through_stage=through_stage,
            force=args.force,
        )
    except ExperimentManagerError as error:
        print(f"Pipeline experiment stopped:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
