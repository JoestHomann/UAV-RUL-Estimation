"""Build Phase 3's resolved settings and manual architecture selection."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tomllib

import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
PHASE_DIR = STEP_DIR.parent
for dependency_dir in (PHASE_DIR, STEP_DIR):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from phase_3_common import (  # noqa: E402
    Phase3Error,
    artifacts_directory,
    manifest_path,
    read_optional_json,
    repository_relative,
    configured_repository_path,
    selected_architecture_path,
    step_directory,
    write_json,
)
from verify_phase_3_settings import (  # noqa: E402
    DEFAULT_SETTINGS_PATH,
    SettingsError,
    load_and_verify_settings,
    PROMOTED_ENSEMBLE_FAMILY,
    PROMOTED_STACK_FAMILY,
    RESIDUAL_ENSEMBLE_FAMILY,
)


def _promoted_ensemble_specification(
    settings: object,
    verification: object,
    artifact_dir: Path,
) -> tuple[dict[str, object], str]:
    """Build a run-local composite search from development-selected components."""

    phase_2_specification = copy.deepcopy(verification.phase_2_specification)
    phase_2_settings = phase_2_specification["settings"]
    source_root = verification.manifest_paths["selection"].parents[1]
    configurations_path = (
        source_root / "5_inner_model_selection" / "selected_configurations.csv"
    )
    try:
        selected = pd.read_csv(configurations_path)
    except (OSError, pd.errors.ParserError) as error:
        raise Phase3Error(
            f"Cannot read promoted component configurations: {error}"
        ) from error
    components: dict[str, list[dict[str, object]]] = {}
    for family in ("extra_trees", "xgboost"):
        rows = selected.loc[selected["model_family"].astype(str) == family].sort_values(
            "outer_fold"
        )
        if len(rows) != 5:
            raise Phase3Error(
                f"Promoted ensemble requires five selected {family} configurations"
            )
        values = []
        for row in rows.to_dict("records"):
            try:
                hyperparameters = json.loads(str(row["hyperparameters_json"]))
            except (json.JSONDecodeError, TypeError) as error:
                raise Phase3Error(
                    f"Cannot parse selected {family} hyperparameters"
                ) from error
            iterations = row.get("outer_retraining_iterations")
            values.append(
                {
                    "source_outer_fold": int(row["outer_fold"]),
                    "source_configuration_id": str(row["configuration_id"]),
                    "hyperparameters": hyperparameters,
                    "training_iterations": (
                        int(iterations)
                        if family == "xgboost" and not pd.isna(iterations)
                        else None
                    ),
                }
            )
        if family == "xgboost" and any(
            value["training_iterations"] is None for value in values
        ):
            raise Phase3Error("Selected XGBoost configurations lack fixed durations")
        components[family] = values

    component_path = artifact_dir / "promoted_component_configurations.json"
    write_json(components, component_path)
    promotion = verification.manifests["promotion"]
    calibration_summary_value = promotion.get("calibration_fit_summary")
    if not isinstance(calibration_summary_value, str):
        raise Phase3Error("Promotion manifest has no calibration fit summary")
    calibration_summary_path = configured_repository_path(
        {"calibration_summary": calibration_summary_value},
        "calibration_summary",
        Path("calibration_fit_summary.json"),
    )
    calibration_summary = read_optional_json(
        calibration_summary_path,
        "calibration fit summary",
    )
    if not isinstance(calibration_summary, dict):
        raise Phase3Error("Calibration fit summary is unavailable")
    calibrator_value = calibration_summary.get("model")
    if not isinstance(calibrator_value, str):
        raise Phase3Error("Calibration fit summary has no frozen model")
    contract = verification.manifests["promotion_contract"]
    weight = float(contract["xgboost_weight"])
    feature_set = str(contract["feature_set"])
    architecture = {
        "status": "included",
        "representation": "tabular",
        "feature_sets": [feature_set],
        "lookbacks": [],
        "variants": ["frozen_calibrated_50_50_blend"],
        "early_stopping_patience": None,
        "search": {
            "extra_trees_configuration_index": {
                "kind": "categorical",
                "values": list(range(len(components["extra_trees"]))),
            },
            "xgboost_configuration_index": {
                "kind": "categorical",
                "values": list(range(len(components["xgboost"]))),
            },
            "component_configurations_path": {
                "kind": "fixed",
                "value": repository_relative(component_path),
            },
            "residual_calibrator_path": {
                "kind": "fixed",
                "value": calibrator_value,
            },
            "xgboost_weight": {"kind": "fixed", "value": weight},
        },
    }
    phase_2_settings["architectures"][PROMOTED_ENSEMBLE_FAMILY] = architecture
    phase_2_settings["study"]["enabled"][PROMOTED_ENSEMBLE_FAMILY] = True
    generated_path = artifact_dir / "promoted_phase_2_specification.json"
    write_json(phase_2_specification, generated_path)
    return phase_2_specification, repository_relative(generated_path)


def _promoted_stack_specification(
    verification: object,
    artifact_dir: Path,
) -> tuple[dict[str, object], str]:
    """Inject one fully frozen heterogeneous stack candidate."""

    phase_2_specification = copy.deepcopy(verification.phase_2_specification)
    phase_2_settings = phase_2_specification["settings"]
    contract = verification.manifests["promotion_contract"]
    hyperparameters = contract.get("adapter_hyperparameters")
    if not isinstance(hyperparameters, dict):
        raise Phase3Error("Promoted stack adapter hyperparameters are missing")
    architecture = {
        "status": "included",
        "representation": "heterogeneous",
        "feature_sets": [],
        "lookbacks": [],
        "variants": [str(contract.get("method"))],
        "early_stopping_patience": None,
        "search": {
            name: {"kind": "fixed", "value": value}
            for name, value in hyperparameters.items()
        },
    }
    phase_2_settings["architectures"][PROMOTED_STACK_FAMILY] = architecture
    phase_2_settings["study"]["enabled"][PROMOTED_STACK_FAMILY] = True
    generated_path = artifact_dir / "promoted_stack_phase_2_specification.json"
    write_json(phase_2_specification, generated_path)
    return phase_2_specification, repository_relative(generated_path)


def _residual_ensemble_specification(
    settings: object,
    verification: object,
    artifact_dir: Path,
) -> tuple[dict[str, object], str]:
    """Build the frozen, internally cross-fitted PE_11 deployment contract."""

    phase_2_specification = copy.deepcopy(verification.phase_2_specification)
    phase_2_settings = phase_2_specification["settings"]
    definition_path = verification.manifest_paths["experiment_definition"]
    try:
        with definition_path.open("rb") as stream:
            definition = tomllib.load(stream)
        workflow = definition["bagging_residual_workflows"]["PE_11"]
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise Phase3Error(f"Cannot read the PE_11 experiment definition: {error}") from error

    selected_path = configured_repository_path(
        {"selected": workflow["selected_configurations"]},
        "selected",
        Path("selected_configurations.csv"),
    )
    try:
        selected = pd.read_csv(selected_path)
    except (OSError, pd.errors.ParserError) as error:
        raise Phase3Error(f"Cannot read PE_11 component configurations: {error}") from error

    ensemble_settings = settings.residual_corrected_ensemble
    if ensemble_settings is None:
        raise Phase3Error("Residual ensemble settings are missing")

    components: dict[str, dict[str, object]] = {}
    feature_set: str | None = None
    choices = {
        "extra_trees": ensemble_settings.extra_trees_configuration_index,
        "xgboost": ensemble_settings.xgboost_configuration_index,
    }
    for family, outer_fold in choices.items():
        rows = selected.loc[
            selected["model_family"].astype(str).eq(family)
            & selected["outer_fold"].astype(int).eq(int(outer_fold))
        ]
        if len(rows) != 1:
            raise Phase3Error(
                f"PE_11 source has no unique {family} configuration for fold {outer_fold}"
            )
        row = rows.iloc[0]
        try:
            hyperparameters = json.loads(str(row["hyperparameters_json"]))
        except (json.JSONDecodeError, TypeError) as error:
            raise Phase3Error(f"Cannot parse the selected {family} configuration") from error
        row_feature_set = str(row["feature_set"])
        if feature_set is None:
            feature_set = row_feature_set
        elif row_feature_set != feature_set:
            raise Phase3Error("PE_11 component feature sets disagree")
        component: dict[str, object] = {
            "source_outer_fold": int(outer_fold),
            "source_configuration_id": str(row["configuration_id"]),
            "hyperparameters": hyperparameters,
        }
        if family == "xgboost":
            iterations = row.get("outer_retraining_iterations")
            if pd.isna(iterations) or int(iterations) <= 0:
                raise Phase3Error("Selected PE_11 XGBoost configuration has no duration")
            component["training_iterations"] = int(iterations)
        components[family] = component

    development = (
        phase_2_specification.get("phase_1_verification", {})
        .get("artifacts", {})
        .get("development_features", {})
    )
    calibration_path = development.get("path")
    if not isinstance(calibration_path, str):
        raise Phase3Error("PE_11 source has no development calibration table")
    contract = {
        "contract_version": 1,
        "method": "residual_corrected",
        "source_experiment": "PE_11",
        "member_seeds": [int(value) for value in workflow["seeds"]],
        "internal_folds": int(ensemble_settings.internal_folds),
        "xgboost_weight_grid": [
            float(value) for value in workflow["xgboost_weight_grid"]
        ],
        "residual_features": [str(value) for value in workflow["residual_features"]],
        "residual_model": {
            "maximum_iterations": int(workflow["residual_maximum_iterations"]),
            "maximum_leaf_nodes": int(workflow["residual_maximum_leaf_nodes"]),
            "minimum_samples_leaf": 20,
            "l2_regularization": float(workflow["residual_l2_regularization"]),
            "learning_rate": 0.05,
            "seed": 13,
        },
        "calibration_features_path": calibration_path,
        "components": components,
        "provenance": {
            "pe11_winner_manifest": repository_relative(
                verification.manifest_paths["pe11_promotion"]
            ),
            "pe12_winner_manifest": repository_relative(
                verification.manifest_paths["pe12_confirmation"]
            ),
            "experiment_definition": repository_relative(definition_path),
            "selected_configurations": repository_relative(selected_path),
        },
    }
    contract_path = artifact_dir / "residual_ensemble_contract.json"
    write_json(contract, contract_path)
    architecture = {
        "status": "included",
        "representation": "tabular",
        "feature_sets": [feature_set],
        "lookbacks": [],
        "variants": ["pe11_residual_corrected"],
        "early_stopping_patience": None,
        "search": {
            "ensemble_contract_path": {
                "kind": "fixed",
                "value": repository_relative(contract_path),
            }
        },
    }
    phase_2_settings["architectures"][RESIDUAL_ENSEMBLE_FAMILY] = architecture
    phase_2_settings["study"]["enabled"][RESIDUAL_ENSEMBLE_FAMILY] = True
    generated_path = artifact_dir / "residual_ensemble_phase_2_specification.json"
    write_json(phase_2_specification, generated_path)
    return phase_2_specification, repository_relative(generated_path)


def build_selection(settings_path: Path = DEFAULT_SETTINGS_PATH) -> dict[str, object]:
    settings, verification = load_and_verify_settings(settings_path)
    run_number = settings.run_number
    selection_path = selected_architecture_path(run_number)
    artifact_dir = artifacts_directory(1, run_number)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    phase_2_specification = verification.phase_2_specification
    generated_specification_path = None
    if settings.selected_model_family == PROMOTED_ENSEMBLE_FAMILY:
        phase_2_specification, generated_specification_path = (
            _promoted_ensemble_specification(settings, verification, artifact_dir)
        )
    elif settings.selected_model_family == PROMOTED_STACK_FAMILY:
        phase_2_specification, generated_specification_path = (
            _promoted_stack_specification(verification, artifact_dir)
        )
    elif settings.selected_model_family == RESIDUAL_ENSEMBLE_FAMILY:
        phase_2_specification, generated_specification_path = (
            _residual_ensemble_specification(settings, verification, artifact_dir)
        )
    phase_2_settings = phase_2_specification["settings"]
    architecture = phase_2_settings["architectures"][settings.selected_model_family]
    selection = {
        "selection_version": 1,
        "settings_version": settings.settings_version,
        "phase_2_run_number": settings.phase_2_run_number,
        "phase_2_settings_version": verification.settings_version,
        "phase_3_run_number": run_number,
        "selected_model_family": settings.selected_model_family,
        "representation": architecture["representation"],
        "feature_sets": architecture["feature_sets"],
        "lookbacks": architecture["lookbacks"],
        "primary_metric": "mean_fold_rmse",
        "prediction_minimum": phase_2_settings["evaluation"]["prediction_minimum"],
        "locked_results_used_for_architecture_selection": (
            settings.selected_model_family != RESIDUAL_ENSEMBLE_FAMILY
        ),
        "locked_results_used_for_configuration_tuning": False,
        "test_data_loaded": False,
        "selection_status": "approved",
    }
    source_settings = settings.model_dump(mode="json", exclude_none=True)
    resolved_settings = copy.deepcopy(source_settings)
    if generated_specification_path is not None:
        resolved_settings["phase_2_specification"] = generated_specification_path
    resolved = {
        "settings_source": repository_relative(settings_path),
        "settings": resolved_settings,
        "phase_2_verification": verification.to_dict(),
        "phase_2_settings": phase_2_settings,
    }
    if generated_specification_path is not None:
        resolved["source_settings"] = source_settings
    existing_selection = read_optional_json(selection_path, "selected architecture")
    existing_resolved = read_optional_json(
        artifact_dir / "resolved_phase_3_settings.json",
        "resolved Phase 3 settings",
    )
    step_2_dir = step_directory(2, run_number=run_number)
    step_2_started = step_2_dir.is_dir() and any(
        path.is_file() for path in step_2_dir.rglob("*")
    )
    if step_2_started and (
        existing_selection != selection or existing_resolved != resolved
    ):
        raise Phase3Error(
            "Phase 3 settings cannot change after Step 2 has written output; "
            "start a new run"
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(resolved, artifact_dir / "resolved_phase_3_settings.json")
    write_json(selection, selection_path)
    manifest = {
        "manifest_version": 1,
        "settings_version": settings.settings_version,
        "phase_2_run_number": settings.phase_2_run_number,
        "phase_3_run_number": run_number,
        "status": "complete",
        "locked_results_used_for_architecture_selection": (
            settings.selected_model_family != RESIDUAL_ENSEMBLE_FAMILY
        ),
        "locked_results_used_for_configuration_tuning": False,
        "test_data_loaded": False,
        "artifacts": {
            "resolved_settings": "artifacts/resolved_phase_3_settings.json",
            "selected_architecture": "artifacts/selected_architecture.json",
        },
    }
    write_json(manifest, manifest_path(1, run_number))
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH)
    args = parser.parse_args()
    try:
        selection = build_selection(args.settings)
    except (SettingsError, Phase3Error) as error:
        print(f"Architecture selection build failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("Phase 3 architecture selection built")
    print(f"Selected family: {selection['selected_model_family']}")
    print(f"Phase 3 run: {selection['phase_3_run_number']}")


if __name__ == "__main__":
    main()
