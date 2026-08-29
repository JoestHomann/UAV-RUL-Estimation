"""Build and rerun configurable feature-engineering experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SETTINGS = SCRIPT_DIR / "feature_engineering_experiments.toml"
ALLOWED_MODELS = {"xgboost", "extra_trees"}
ALLOWED_PHASE_1_MODES = {"reuse", "rebuild"}
ALLOWED_FEATURE_PROFILES = {"legacy", "extended"}
ALLOWED_PREFIX_STRATEGIES = {"empirical", "stratified_empirical"}
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class WorkflowPaths:
    phase_1_settings: Path
    phase_1_runner: Path
    diagnostics_runner: Path
    train_csv: Path
    anomaly_priority: Path


@dataclass(frozen=True)
class DiagnosticSettings:
    feature_sets: tuple[str, ...]
    models: tuple[str, ...]
    permutation_repetitions: int
    skip_permutation: bool
    seed: int
    dpi: int
    model_parameters: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class PrefixVariantSettings:
    strategy: str
    cutoffs_per_uav: int
    seed: int


@dataclass(frozen=True)
class Experiment:
    name: str
    enabled: bool
    profile: str
    phase_1_run_name: str
    phase_1_mode: str
    prefix_variants: tuple[str, ...]
    profile_feature_sets: tuple[str, ...]
    output_root: Path
    diagnostics: DiagnosticSettings


@dataclass(frozen=True)
class LauncherSettings:
    settings_version: int
    paths: WorkflowPaths
    prefix_variants: dict[str, PrefixVariantSettings]
    experiments: dict[str, Experiment]


def require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a TOML table")
    return value


def reject_unknown_keys(
    table: dict[str, Any],
    allowed: set[str],
    field: str,
) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unknown settings {unknown}")


def repository_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    supplied = Path(value)
    if supplied.is_absolute():
        raise ValueError(f"{field} must be repository-relative")
    resolved = (REPOSITORY_ROOT / supplied).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{field} escapes the repository") from error
    return resolved


def string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty TOML array")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must contain non-empty strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def positive_integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def load_settings(path: Path = DEFAULT_SETTINGS) -> LauncherSettings:
    """Load and strictly validate the editable experiment registry."""

    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Cannot read experiment settings {path}: {error}") from error

    reject_unknown_keys(
        payload,
        {
            "settings_version",
            "paths",
            "profiles",
            "scenario_profiles",
            "prefix_variants",
            "runs",
        },
        "root",
    )
    settings_version = positive_integer(
        payload.get("settings_version"),
        "settings_version",
    )
    path_table = require_mapping(payload.get("paths"), "paths")
    required_paths = {
        "phase_1_settings",
        "phase_1_runner",
        "diagnostics_runner",
        "train_csv",
        "anomaly_priority",
    }
    reject_unknown_keys(path_table, required_paths, "paths")
    missing_paths = sorted(required_paths - set(path_table))
    if missing_paths:
        raise ValueError(f"paths is missing required settings {missing_paths}")
    paths = WorkflowPaths(
        **{
            name: repository_path(path_table[name], f"paths.{name}")
            for name in required_paths
        }
    )

    profile_tables = require_mapping(payload.get("profiles"), "profiles")
    prefix_variant_tables = require_mapping(
        payload.get("prefix_variants"),
        "prefix_variants",
    )
    prefix_variants_by_name: dict[str, PrefixVariantSettings] = {}
    for variant_name, raw_variant in prefix_variant_tables.items():
        variant = require_mapping(
            raw_variant,
            f"prefix_variants.{variant_name}",
        )
        reject_unknown_keys(
            variant,
            {"strategy", "cutoffs_per_uav", "seed"},
            f"prefix_variants.{variant_name}",
        )
        strategy = variant.get("strategy")
        if strategy not in ALLOWED_PREFIX_STRATEGIES:
            raise ValueError(
                f"prefix_variants.{variant_name}.strategy must be one of "
                f"{sorted(ALLOWED_PREFIX_STRATEGIES)}"
            )
        prefix_variants_by_name[variant_name] = PrefixVariantSettings(
            strategy=strategy,
            cutoffs_per_uav=positive_integer(
                variant.get("cutoffs_per_uav"),
                f"prefix_variants.{variant_name}.cutoffs_per_uav",
            ),
            seed=integer(
                variant.get("seed"),
                f"prefix_variants.{variant_name}.seed",
            ),
        )
    run_tables = require_mapping(payload.get("runs"), "runs")
    if not run_tables:
        raise ValueError("runs must declare at least one named experiment")
    experiments: dict[str, Experiment] = {}
    run_keys = {
        "enabled",
        "profile",
        "phase_1_run_name",
        "phase_1_mode",
        "prefix_variants",
        "output_root",
        "diagnostics",
        "model_parameters",
    }
    diagnostic_keys = {
        "feature_sets",
        "models",
        "permutation_repetitions",
        "skip_permutation",
        "seed",
        "dpi",
    }
    for name, raw_run in run_tables.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("run names must be non-empty strings")
        run = require_mapping(raw_run, f"runs.{name}")
        reject_unknown_keys(run, run_keys, f"runs.{name}")
        profile = run.get("profile")
        if not isinstance(profile, str) or not profile.strip():
            raise ValueError(f"runs.{name}.profile must be a non-empty string")
        if profile not in profile_tables:
            raise ValueError(f"runs.{name} references unknown profile {profile!r}")
        phase_1_run_name = run.get("phase_1_run_name")
        if not isinstance(phase_1_run_name, str) or not RUN_NAME_PATTERN.fullmatch(
            phase_1_run_name
        ):
            raise ValueError(
                f"runs.{name}.phase_1_run_name must start with a letter and "
                "contain only letters, digits, underscores, or hyphens"
            )
        profile_table = require_mapping(
            profile_tables[profile],
            f"profiles.{profile}",
        )
        feature_profile = profile_table.get("feature_profile")
        if feature_profile not in ALLOWED_FEATURE_PROFILES:
            raise ValueError(
                f"profiles.{profile}.feature_profile must be one of "
                f"{sorted(ALLOWED_FEATURE_PROFILES)}"
            )
        profile_feature_sets = string_tuple(
            profile_table.get("feature_sets"),
            f"profiles.{profile}.feature_sets",
        )
        profile_variants = string_tuple(
            profile_table.get("prefix_variants"),
            f"profiles.{profile}.prefix_variants",
        )
        prefix_variants = string_tuple(
            run.get("prefix_variants"),
            f"runs.{name}.prefix_variants",
        )
        unknown_variants = sorted(set(prefix_variants) - set(profile_variants))
        if unknown_variants:
            raise ValueError(
                f"runs.{name} requests variants absent from profile {profile!r}: "
                f"{unknown_variants}"
            )
        undefined_variants = sorted(
            set(prefix_variants) - set(prefix_variants_by_name)
        )
        if undefined_variants:
            raise ValueError(
                f"runs.{name} references undefined prefix variants "
                f"{undefined_variants}"
            )
        diagnostics = require_mapping(
            run.get("diagnostics"),
            f"runs.{name}.diagnostics",
        )
        reject_unknown_keys(
            diagnostics,
            diagnostic_keys,
            f"runs.{name}.diagnostics",
        )
        models = string_tuple(
            diagnostics.get("models"),
            f"runs.{name}.diagnostics.models",
        )
        unknown_models = sorted(set(models) - ALLOWED_MODELS)
        if unknown_models:
            raise ValueError(f"runs.{name} contains unknown models {unknown_models}")
        feature_sets = string_tuple(
            diagnostics.get("feature_sets"),
            f"runs.{name}.diagnostics.feature_sets",
        )
        unknown_feature_sets = sorted(set(feature_sets) - set(profile_feature_sets))
        if unknown_feature_sets:
            raise ValueError(
                f"runs.{name} requests feature sets absent from profile "
                f"{profile!r}: {unknown_feature_sets}"
            )

        raw_parameters = require_mapping(
            run.get("model_parameters"),
            f"runs.{name}.model_parameters",
        )
        reject_unknown_keys(
            raw_parameters,
            ALLOWED_MODELS,
            f"runs.{name}.model_parameters",
        )
        model_parameters = {
            model: require_mapping(
                raw_parameters.get(model, {}),
                f"runs.{name}.model_parameters.{model}",
            )
            for model in models
        }
        if any("random_state" in values for values in model_parameters.values()):
            raise ValueError(
                f"runs.{name} must use diagnostics.seed instead of random_state"
            )

        phase_1_mode = run.get("phase_1_mode")
        if phase_1_mode not in ALLOWED_PHASE_1_MODES:
            raise ValueError(
                f"runs.{name}.phase_1_mode must be one of "
                f"{sorted(ALLOWED_PHASE_1_MODES)}"
            )
        enabled = run.get("enabled")
        skip_permutation = diagnostics.get("skip_permutation")
        if not isinstance(enabled, bool) or not isinstance(skip_permutation, bool):
            raise ValueError(
                f"runs.{name}.enabled and diagnostics.skip_permutation "
                "must be Boolean"
            )
        experiments[name] = Experiment(
            name=name,
            enabled=enabled,
            profile=profile,
            phase_1_run_name=phase_1_run_name,
            phase_1_mode=phase_1_mode,
            prefix_variants=prefix_variants,
            profile_feature_sets=profile_feature_sets,
            output_root=repository_path(
                run.get("output_root"),
                f"runs.{name}.output_root",
            ),
            diagnostics=DiagnosticSettings(
                feature_sets=feature_sets,
                models=models,
                permutation_repetitions=positive_integer(
                    diagnostics.get("permutation_repetitions"),
                    f"runs.{name}.diagnostics.permutation_repetitions",
                ),
                skip_permutation=skip_permutation,
                seed=positive_integer(
                    diagnostics.get("seed"),
                    f"runs.{name}.diagnostics.seed",
                ),
                dpi=positive_integer(
                    diagnostics.get("dpi"),
                    f"runs.{name}.diagnostics.dpi",
                ),
                model_parameters=model_parameters,
            ),
        )
    return LauncherSettings(
        settings_version=settings_version,
        paths=paths,
        prefix_variants=prefix_variants_by_name,
        experiments=experiments,
    )


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def execute(command: list[str], *, dry_run: bool) -> None:
    print(f"\n{command_text(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def interface_path(settings: LauncherSettings, experiment: Experiment, variant: str) -> Path:
    return (
        settings.paths.phase_1_runner.parent
        / "runs"
        / experiment.phase_1_run_name
        / variant
        / "phase_2_interface.json"
    )


def fallback_artifacts(
    settings: LauncherSettings,
    experiment: Experiment,
    variant: str,
) -> dict[str, str]:
    """Predict deterministic paths for a dry run before Phase 1 exists."""

    variant_root = interface_path(settings, experiment, variant).parent
    phase_1_root = settings.paths.phase_1_runner.parent
    return {
        "training_features": str(
            variant_root
            / "5_prefix_feature_engineering"
            / "artifacts"
            / "training_features.csv.gz"
        ),
        "development_features": str(
            variant_root
            / "5_prefix_feature_engineering"
            / "artifacts"
            / "development_validation_features.csv.gz"
        ),
        "test_features": str(
            variant_root
            / "5_prefix_feature_engineering"
            / "artifacts"
            / "test_features.csv.gz"
        ),
        "feature_catalog": str(
            variant_root
            / "6_feature_sets"
            / "artifacts"
            / "feature_catalog.csv"
        ),
        "outer_folds": str(
            phase_1_root
            / "2_UAV_grouped_validation_folds"
            / "artifacts"
            / "outer_folds.csv"
        ),
    }


def load_interface(
    settings: LauncherSettings,
    experiment: Experiment,
    variant: str,
    *,
    dry_run: bool,
    allow_stale: bool = False,
) -> dict[str, Any]:
    path = interface_path(settings, experiment, variant)
    if not path.is_file():
        if dry_run:
            return {"artifacts": fallback_artifacts(settings, experiment, variant)}
        raise FileNotFoundError(
            f"Missing {path}. Set phase_1_mode = 'rebuild' or use "
            "--rebuild-phase1."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read Phase 1 interface {path}: {error}") from error
    artifacts = require_mapping(payload.get("artifacts"), f"{path}: artifacts")
    required = {
        "training_features",
        "development_features",
        "test_features",
        "feature_catalog",
        "outer_folds",
        "training_prefix_config",
    }
    missing = sorted(required - set(artifacts))
    if missing:
        raise ValueError(f"Phase 1 interface {path} is missing artifacts {missing}")
    expected_sets = require_mapping(
        payload.get("expected_feature_sets"),
        f"{path}: expected_feature_sets",
    )
    if not allow_stale and set(expected_sets) != set(experiment.profile_feature_sets):
        raise ValueError(
            f"Phase 1 interface {path} does not match the profile feature sets. "
            "Rebuild Phase 1 before running diagnostics."
        )
    unknown = sorted(set(experiment.diagnostics.feature_sets) - set(expected_sets))
    if unknown:
        raise ValueError(
            f"Run {experiment.name}, variant {variant} requests feature sets "
            f"absent from its Phase 1 catalog: {unknown}"
        )
    if not allow_stale:
        prefix_config_path = artifact_path(
            artifacts["training_prefix_config"],
            "training_prefix_config",
        )
        prefix_config = json.loads(prefix_config_path.read_text(encoding="utf-8"))
        configured = settings.prefix_variants[variant]
        expected_prefix_values = {
            "strategy": configured.strategy,
            "cutoffs_per_uav": configured.cutoffs_per_uav,
            "seed": configured.seed,
        }
        mismatches = {
            name: {"configured": value, "artifact": prefix_config.get(name)}
            for name, value in expected_prefix_values.items()
            if prefix_config.get(name) != value
        }
        if mismatches:
            raise ValueError(
                f"Phase 1 interface {path} is stale for prefix variant "
                f"{variant!r}: {mismatches}. Rebuild Phase 1."
            )
    return payload


def artifact_path(value: Any, field: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a path string")
    path = Path(value)
    if path.is_absolute():
        try:
            path.resolve().relative_to(REPOSITORY_ROOT.resolve())
        except ValueError as error:
            raise ValueError(f"{field} points outside the repository") from error
        return path.resolve()
    return repository_path(value, field)


def diagnostic_command(
    settings: LauncherSettings,
    experiment: Experiment,
    variant: str,
    interface: dict[str, Any],
) -> list[str]:
    diagnostics = experiment.diagnostics
    artifacts = require_mapping(interface.get("artifacts"), "interface.artifacts")
    command = [
        sys.executable,
        str(settings.paths.diagnostics_runner),
        "--training-features",
        str(artifact_path(artifacts["training_features"], "training_features")),
        "--development-features",
        str(artifact_path(artifacts["development_features"], "development_features")),
        "--test-features",
        str(artifact_path(artifacts["test_features"], "test_features")),
        "--feature-catalog",
        str(artifact_path(artifacts["feature_catalog"], "feature_catalog")),
        "--outer-folds",
        str(artifact_path(artifacts["outer_folds"], "outer_folds")),
        "--anomaly-priority",
        str(settings.paths.anomaly_priority),
        "--train-csv",
        str(settings.paths.train_csv),
        "--feature-sets",
        *diagnostics.feature_sets,
        "--models",
        *diagnostics.models,
        "--permutation-repetitions",
        str(diagnostics.permutation_repetitions),
        "--seed",
        str(diagnostics.seed),
        "--dpi",
        str(diagnostics.dpi),
        "--output-dir",
        str(experiment.output_root / variant),
    ]
    if "xgboost" in diagnostics.models:
        command.extend(
            [
                "--xgboost-parameters-json",
                json.dumps(diagnostics.model_parameters["xgboost"]),
            ]
        )
    if "extra_trees" in diagnostics.models:
        command.extend(
            [
                "--extra-trees-parameters-json",
                json.dumps(diagnostics.model_parameters["extra_trees"]),
            ]
        )
    if diagnostics.skip_permutation:
        command.append("--skip-permutation")
    return command


def write_launcher_manifest(
    settings_path: Path,
    settings: LauncherSettings,
    experiment: Experiment,
    commands: list[list[str]],
    status: str,
) -> Path:
    experiment.output_root.mkdir(parents=True, exist_ok=True)
    path = experiment.output_root / "feature_engineering_run_manifest.json"
    payload = {
        "status": status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings_version": settings.settings_version,
        "settings": settings_path.resolve().relative_to(REPOSITORY_ROOT).as_posix(),
        "run": experiment.name,
        "phase_1_profile": experiment.profile,
        "phase_1_run_name": experiment.phase_1_run_name,
        "prefix_variants": list(experiment.prefix_variants),
        "commands": [command_text(command) for command in commands],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def run_experiment(
    settings_path: Path,
    settings: LauncherSettings,
    experiment: Experiment,
    *,
    rebuild_phase_1: bool,
    phase_1_only: bool,
    dry_run: bool,
) -> None:
    print(f"\n=== Feature-engineering experiment: {experiment.name} ===")
    commands: list[list[str]] = []
    rebuild = rebuild_phase_1 or experiment.phase_1_mode == "rebuild"
    if rebuild:
        phase_1_command = [
            sys.executable,
            str(settings.paths.phase_1_runner),
            "--settings",
            str(settings.paths.phase_1_settings),
            "--profile",
            experiment.profile,
            "--run-name",
            experiment.phase_1_run_name,
            "--prefix-variant",
            *experiment.prefix_variants,
        ]
        commands.append(phase_1_command)
        execute(phase_1_command, dry_run=dry_run)

    if not phase_1_only:
        for variant in experiment.prefix_variants:
            interface = load_interface(
                settings,
                experiment,
                variant,
                dry_run=dry_run,
                allow_stale=dry_run and rebuild,
            )
            command = diagnostic_command(settings, experiment, variant, interface)
            commands.append(command)
            execute(command, dry_run=dry_run)

    if not dry_run:
        manifest = write_launcher_manifest(
            settings_path,
            settings,
            experiment,
            commands,
            "complete",
        )
        print(f"Completed {experiment.name}; manifest: {manifest}")


def selected_experiments(
    settings: LauncherSettings,
    requested: list[str] | None,
) -> list[Experiment]:
    if requested:
        unknown = sorted(set(requested) - set(settings.experiments))
        if unknown:
            raise ValueError(f"Unknown experiment runs {unknown}")
        return [settings.experiments[name] for name in requested]
    selected = [item for item in settings.experiments.values() if item.enabled]
    if not selected:
        raise ValueError("No enabled runs; enable one or pass --run NAME")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--run", nargs="+", dest="runs")
    parser.add_argument("--list", action="store_true", dest="list_runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-phase1", action="store_true")
    parser.add_argument("--phase1-only", action="store_true")
    args = parser.parse_args()

    settings_path = args.settings.resolve()
    settings = load_settings(settings_path)
    if args.list_runs:
        for experiment in settings.experiments.values():
            state = "enabled" if experiment.enabled else "disabled"
            variants = ", ".join(experiment.prefix_variants)
            print(f"{experiment.name}: {state}; {experiment.profile}; {variants}")
        return

    for experiment in selected_experiments(settings, args.runs):
        run_experiment(
            settings_path,
            settings,
            experiment,
            rebuild_phase_1=args.rebuild_phase1,
            phase_1_only=args.phase1_only,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
