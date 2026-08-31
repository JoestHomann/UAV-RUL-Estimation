"""Freeze the selected architecture and configuration into a training contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np


STEP_DIR = Path(__file__).resolve().parent
PHASE_DIR = STEP_DIR.parent
REPOSITORY_ROOT = PHASE_DIR.parent
PHASE_2_DIR = REPOSITORY_ROOT / "2_model_architecture_study"
for dependency_dir in (
    PHASE_DIR,
    PHASE_2_DIR / "2_tabular_data_adapter",
    PHASE_2_DIR / "3_sequence_data_adapter",
    PHASE_2_DIR / "3_trajectory_data_adapter",
    PHASE_2_DIR / "4_model_adapters",
):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from model_registry import EXPECTED_HYPERPARAMETERS  # noqa: E402
from sequence_data_adapter import SequenceDataAdapter  # noqa: E402
from tabular_data_adapter import TabularDataAdapter  # noqa: E402
from trajectory_data_adapter import TrajectoryDataAdapter  # noqa: E402
from phase_3_common import (  # noqa: E402
    PHASE_2_SPECIFICATION_PATH,
    Phase3Error,
    complete_manifest,
    configured_repository_path,
    invalidate_downstream_manifests,
    load_resolved_phase_3_settings,
    manifest_path,
    read_json,
    repository_relative,
    require_current_settings,
    selected_architecture_path,
    selected_configuration_path,
    training_contract_path,
    write_json,
)
from phase_3_run_layout import SETTINGS_PATH  # noqa: E402


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


class FinalTrainingContractError(Phase3Error):
    """Explain an incomplete or internally inconsistent frozen contract."""


def _training_view(
    representation: str,
    feature_set: str | None,
    lookback: int | None,
    *,
    tabular_manifest_path: Path | None = None,
    sequence_manifest_path: Path | None = None,
    trajectory_manifest_path: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load training-only metadata and ordered inputs for the selected family."""

    if representation in {"none", "tabular"}:
        selected_feature_set = feature_set or "age_only"
        dataset = (
            TabularDataAdapter(tabular_manifest_path)
            if tabular_manifest_path is not None
            else TabularDataAdapter()
        ).load_training(selected_feature_set)
        schema = {
            "representation": representation,
            "feature_set": selected_feature_set,
            "feature_names": list(dataset.features.columns),
            "lookback": None,
            "channel_names": [],
            "side_feature_names": [],
        }
        return dataset, schema
    if representation == "sequence":
        if lookback is None:
            raise FinalTrainingContractError("Sequence configuration has no lookback")
        dataset = (
            SequenceDataAdapter(sequence_manifest_path)
            if sequence_manifest_path is not None
            else SequenceDataAdapter()
        ).load_training(lookback)
        schema = {
            "representation": representation,
            "feature_set": None,
            "feature_names": [],
            "lookback": lookback,
            "channel_names": list(dataset.channel_names),
            "side_feature_names": list(dataset.side_feature_names),
        }
        return dataset, schema
    if representation == "trajectory":
        dataset = (
            TrajectoryDataAdapter(trajectory_manifest_path)
            if trajectory_manifest_path is not None
            else TrajectoryDataAdapter()
        ).load_training()
        schema = {
            "representation": representation,
            "feature_set": None,
            "feature_names": [],
            "lookback": None,
            "channel_names": list(dataset.channel_names),
            "side_feature_names": list(dataset.side_feature_names),
        }
        return dataset, schema
    raise FinalTrainingContractError(
        f"Unsupported selected representation {representation!r}"
    )


def _verify_training_rows(
    dataset: Any,
    expected_rows: int,
    expected_uavs: int,
) -> list[str]:
    if dataset.target is None or dataset.sample_weights is None:
        raise FinalTrainingContractError("Final training view lacks targets or weights")
    if len(dataset) != expected_rows:
        raise FinalTrainingContractError(
            f"Expected {expected_rows} training rows, found {len(dataset)}"
        )
    uav_ids = dataset.metadata["uav_id"].astype(str)
    unique_uavs = sorted(uav_ids.unique())
    if len(unique_uavs) != expected_uavs:
        raise FinalTrainingContractError(
            f"Expected {expected_uavs} training UAVs, found {len(unique_uavs)}"
        )
    weight_sums = dataset.sample_weights.groupby(uav_ids).sum().to_numpy(dtype=float)
    if not np.allclose(weight_sums, np.ones_like(weight_sums), rtol=0, atol=1e-12):
        raise FinalTrainingContractError(
            "Training prefix weights do not give every UAV equal total influence"
        )
    return unique_uavs


def build_contract(run_number: int, *, force: bool = False) -> dict[str, Any]:
    settings = load_resolved_phase_3_settings(run_number)
    settings_version = int(settings["settings_version"])
    if not force and complete_manifest(3, run_number, settings_version) is not None:
        return read_json(training_contract_path(run_number), "final training contract")
    if force:
        invalidate_downstream_manifests(3, run_number)
    for prerequisite in (1, 2):
        if complete_manifest(prerequisite, run_number, settings_version) is None:
            raise FinalTrainingContractError(
                f"Phase 3 Step {prerequisite} is not complete"
            )

    architecture = read_json(
        selected_architecture_path(run_number),
        "selected architecture",
    )
    configuration = read_json(
        selected_configuration_path(run_number),
        "selected configuration",
    )
    if architecture["selected_model_family"] != configuration["model_family"]:
        raise FinalTrainingContractError(
            "Selected architecture and configuration identify different families"
        )
    family = str(configuration["model_family"])
    representation = str(configuration["representation"])
    hyperparameters = configuration.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        raise FinalTrainingContractError("Selected configuration has no hyperparameters")
    if set(hyperparameters) != EXPECTED_HYPERPARAMETERS[family]:
        raise FinalTrainingContractError(
            f"Selected {family} hyperparameters do not match the adapter contract"
        )

    iterations = configuration.get("final_training_iterations")
    if family in EARLY_STOPPED_FAMILIES:
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
            raise FinalTrainingContractError(
                f"Early-stopped family {family!r} has no fixed training duration"
            )
    elif iterations is not None:
        raise FinalTrainingContractError(
            f"Non-iterative family {family!r} unexpectedly has a training duration"
        )

    phase_2_specification_path = configured_repository_path(
        settings,
        "phase_2_specification",
        PHASE_2_SPECIFICATION_PATH,
    )
    phase_2_specification = read_json(
        phase_2_specification_path,
        "Phase 2 experiment specification",
    )
    phase_2_settings = phase_2_specification["settings"]
    expected_uavs = int(phase_2_settings["phase_1"]["expected_training_uavs"])
    training_observation = (
        phase_2_specification.get("phase_1_verification", {})
        .get("artifacts", {})
        .get("training_features", {})
    )
    expected_rows = training_observation.get("rows")
    if isinstance(expected_rows, bool) or not isinstance(expected_rows, int):
        raise FinalTrainingContractError(
            "Phase 2 specification does not record the observed training row count"
        )
    tabular_manifest_path = configured_repository_path(
        settings,
        "tabular_manifest",
        PHASE_2_DIR
        / "2_tabular_data_adapter"
        / "artifacts"
        / "tabular_dataset_manifest.json",
    )
    sequence_manifest_path = configured_repository_path(
        settings,
        "sequence_manifest",
        PHASE_2_DIR
        / "3_sequence_data_adapter"
        / "artifacts"
        / "sequence_dataset_manifest.json",
    )
    trajectory_manifest_path = configured_repository_path(
        settings,
        "trajectory_manifest",
        PHASE_2_DIR
        / "3_trajectory_data_adapter"
        / "artifacts"
        / "trajectory_dataset_manifest.json",
    )
    dataset, input_schema = _training_view(
        representation,
        configuration.get("feature_set"),
        configuration.get("lookback"),
        tabular_manifest_path=tabular_manifest_path,
        sequence_manifest_path=sequence_manifest_path,
        trajectory_manifest_path=trajectory_manifest_path,
    )
    training_uav_ids = _verify_training_rows(dataset, expected_rows, expected_uavs)

    verification = phase_2_specification.get("phase_1_verification", {})
    test_observation = verification.get("artifacts", {}).get("test_features", {})
    expected_test_rows = test_observation.get("rows")
    if isinstance(expected_test_rows, bool) or not isinstance(expected_test_rows, int):
        raise FinalTrainingContractError(
            "Phase 2 specification does not record the expected test row count"
        )

    contract = {
        "contract_version": 1,
        "settings_version": settings_version,
        "phase_2_run_number": settings["phase_2_run_number"],
        "phase_3_run_number": run_number,
        "model_family": family,
        "representation": representation,
        "configuration_id": configuration["configuration_id"],
        "input_schema": input_schema,
        "hyperparameters": hyperparameters,
        "data_manifests": {
            "tabular": repository_relative(tabular_manifest_path),
            "sequence": repository_relative(sequence_manifest_path),
            "trajectory": repository_relative(trajectory_manifest_path),
        },
        "target": phase_2_settings.get(
            "target",
            {"mode": "raw", "maximum_rul": None},
        ),
        "prediction_policy": phase_2_settings.get(
            "prediction_policy",
            {
                "loss": "symmetric_rmse",
                "overprediction_weight": 1.0,
                "quantile": 0.5,
                "severity_scale": 10.0,
                "calibration": "none",
                "safety_offset": 0.0,
                "non_overprediction_coverage": 0.5,
            },
        ),
        "preprocessing": {
            "algorithm": (
                "robust_channel_scaler"
                if representation in {"sequence", "trajectory"}
                else "selected_model_adapter"
            ),
            "telemetry_algorithm": (
                "robust_channel_scaler"
                if representation in {"sequence", "trajectory"}
                else None
            ),
            "side_feature_algorithm": (
                "model_adapter_robust_scaler"
                if representation == "sequence"
                else None
            ),
            "fit_scope": "all_training_uavs",
            "separate_artifact": representation in {"sequence", "trajectory"},
        },
        "training": {
            "row_count": expected_rows,
            "uav_count": expected_uavs,
            "uav_ids": training_uav_ids,
            "sample_weighting": "each UAV has total prefix weight 1.0",
            "iterations_or_epochs": iterations,
            "model_seed": int(settings["final_search"]["model_seed"]),
        },
        "prediction_minimum": float(
            phase_2_settings["evaluation"]["prediction_minimum"]
        ),
        "serialization": {
            "model_format": "trusted local joblib artifact",
            "model_filename": "final_model.joblib",
            "preprocessor_filename": (
                "final_preprocessor.joblib"
                if representation in {"sequence", "trajectory"}
                else None
            ),
        },
        "test_contract": {
            "expected_rows": expected_test_rows,
            "metadata_columns": ["sample_id", "uav_id", "cutoff"],
            "prediction_columns": [
                "sample_id",
                "uav_id",
                "cutoff",
                "model_family",
                "configuration_id",
                "model_seed",
                "RUL",
            ],
            "submission_columns": ["id", "RUL"],
            "submission_identifier_mapping": {
                "prediction_column": "uav_id",
                "submission_column": "id",
            },
            "row_order": "id ascending",
        },
        "locked_data_loaded": False,
        "test_data_loaded": False,
        "status": "frozen",
    }
    write_json(contract, training_contract_path(run_number))
    manifest = {
        "manifest_version": 1,
        "settings_version": settings_version,
        "phase_2_run_number": settings["phase_2_run_number"],
        "phase_3_run_number": run_number,
        "status": "complete",
        "model_family": family,
        "configuration_id": configuration["configuration_id"],
        "training_rows": expected_rows,
        "training_uavs": expected_uavs,
        "locked_data_loaded": False,
        "test_data_loaded": False,
        "artifacts": {
            "training_contract": "artifacts/final_training_contract.json"
        },
    }
    write_json(manifest, manifest_path(3, run_number))
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--settings", type=Path, default=SETTINGS_PATH)
    args = parser.parse_args()
    try:
        contract = build_contract(
            require_current_settings(args.settings),
            force=args.force,
        )
    except (FinalTrainingContractError, Phase3Error, OSError, ValueError) as error:
        print(f"Final training contract build failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("Final training contract built")
    print(f"Family: {contract['model_family']}")
    print(f"Configuration: {contract['configuration_id']}")


if __name__ == "__main__":
    main()
