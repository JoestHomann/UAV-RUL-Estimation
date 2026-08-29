"""Run declarative Phase 1, Phase 2, and Phase 3 experiment definitions."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any

import pandas as pd


MANAGER_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MANAGER_DIR.parent
DEFAULT_CONFIG_PATH = MANAGER_DIR / "pipeline_experiments.toml"
RUNS_DIR = MANAGER_DIR / "runs"
PHASE_1_ROOT = REPOSITORY_ROOT / "1_dataset_construction"
PHASE_2_ROOT = REPOSITORY_ROOT / "2_model_architecture_study"
PHASE_3_ROOT = REPOSITORY_ROOT / "3_final_model_training_and_inference"
STAGES = ("phase1", "phase2", "phase3")


class ExperimentManagerError(ValueError):
    """Explain an invalid experiment definition or failed pipeline stage."""


def _read_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ExperimentManagerError(f"Cannot read experiment catalog: {error}") from error
    if not isinstance(payload, dict):
        raise ExperimentManagerError("Experiment catalog must be a TOML table")
    return payload


def _repo_path(value: str, *, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExperimentManagerError(f"{description} must be a non-empty path")
    path = (REPOSITORY_ROOT / value).resolve()
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
    value = config.get("experiments")
    if not isinstance(value, dict) or not value:
        raise ExperimentManagerError("The catalog needs at least one [experiments.*] table")
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


def _run_dir(name: str) -> Path:
    return RUNS_DIR / name


def _state_path(name: str) -> Path:
    return _run_dir(name) / "experiment_status.json"


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
            "paths.phase_2_settings must point inside pipeline_experiments"
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
            architecture["feature_sets"] = ["age_only"] if family == "cycle_only_baseline" else [feature_set]

    return settings


def _phase1_interface_path(experiment: dict[str, Any]) -> Path:
    phase1_run = experiment.get("phase_1_run_name")
    variant = experiment.get("prefix_variant")
    return PHASE_1_ROOT / "runs" / str(phase1_run) / str(variant) / "phase_2_interface.json"


def _run_phase1(name: str, config_path: Path, config: dict[str, Any], experiment: dict[str, Any]) -> None:
    paths = _paths(config)
    command = [
        sys.executable,
        str(paths["phase_1_runner"]),
        "--settings",
        str(config_path),
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


def _phase2_paths(name: str) -> dict[str, Path]:
    root = _run_dir(name) / "phase2"
    return {
        "root": root,
        "settings": root / "phase_2_settings.json",
        "specification": root / "1_architecture_study_settings" / "artifacts" / "experiment_specification.json",
        "tabular_manifest": root / "2_tabular_data_adapter" / "artifacts" / "tabular_dataset_manifest.json",
        "sequence_manifest": root / "3_sequence_data_adapter" / "artifacts" / "sequence_dataset_manifest.json",
        "trajectory_manifest": root / "3_trajectory_data_adapter" / "artifacts" / "trajectory_dataset_manifest.json",
        "registry": root / "4_model_adapters" / "artifacts" / "model_registry.json",
    }


def _phase2_run_root(experiment: dict[str, Any]) -> Path:
    return PHASE_2_ROOT / "runs" / f"run_{int(experiment['phase_2_run_number'])}"


def _run_phase2(name: str, config: dict[str, Any], experiment: dict[str, Any]) -> None:
    paths = _paths(config)
    interface_path, interface = _load_interface(experiment)
    phase2 = _phase2_paths(name)
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

    run_root = _phase2_run_root(experiment)
    step5 = run_root / "5_inner_model_selection"
    step6 = run_root / "6_locked_outer_evaluation"
    step7 = run_root / "7_architecture_comparison"
    _run_command(
        [
            sys.executable,
            str(paths["phase_2_orchestrator"]),
            "--specification",
            str(phase2["specification"]),
            "--from-step",
            "5",
            "--through-step",
            "6",
            "--tabular-manifest",
            str(phase2["tabular_manifest"]),
            "--sequence-manifest",
            str(phase2["sequence_manifest"]),
            "--trajectory-manifest",
            str(phase2["trajectory_manifest"]),
            "--model-registry",
            str(phase2["registry"]),
        ],
        label=f"{name}: Phase 2 Steps 5-6 parallel selection and evaluation",
    )
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
    phase2 = _phase2_paths(name)
    payload = {
        "settings_version": int(experiment.get("phase_3_settings_version", 1)),
        "run_number": int(experiment["phase_3_run_number"]),
        "phase_2_run_number": int(experiment["phase_2_run_number"]),
        "selected_model_family": selected,
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
    path = _run_dir(name) / "phase3_settings.json"
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
    path = _state_path(name)
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
    _write_json(payload, _state_path(name))


def _stage_complete(name: str, experiment: dict[str, Any], stage: str) -> bool:
    state = _state(name, experiment)
    if state.get("stages", {}).get(stage) != "complete":
        return False
    if stage == "phase1":
        return _phase1_interface_path(experiment).is_file()
    if stage == "phase2":
        path = _phase2_run_root(experiment) / "7_architecture_comparison" / "comparison_manifest.json"
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
    config_path: Path,
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
            continue
        _mark_stage(name, experiment, stage, "running")
        try:
            if stage == "phase1":
                _run_phase1(name, config_path, config, experiment)
            elif stage == "phase2":
                _run_phase2(name, config, experiment)
            else:
                _run_phase3(name, config, experiment, force)
        except KeyboardInterrupt:
            _mark_stage(name, experiment, stage, "interrupted")
            raise
        except Exception:
            _mark_stage(name, experiment, stage, "failed")
            raise
        _mark_stage(name, experiment, stage, "complete")
        print(f"{name}: {stage} complete")


def print_status(config: dict[str, Any]) -> None:
    for name, experiment in sorted(_experiments(config).items()):
        state = _state(name, experiment)
        stages = [
            f"{stage}={state.get('stages', {}).get(stage, 'not_started')}"
            for stage in STAGES
        ]
        print(f"{name}: " + ", ".join(stages))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run", dest="run_name")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--status", action="store_true")
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
            return
        if args.status:
            print_status(config)
            return
        if not args.run_name:
            parser.error("--run is required unless --list or --status is used")
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
            config_path,
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
