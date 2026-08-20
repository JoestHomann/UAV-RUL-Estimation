"""Fit the frozen model once on all training UAVs and verify persistence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

import joblib
import numpy as np


STEP_DIR = Path(__file__).resolve().parent
PHASE_DIR = STEP_DIR.parent
REPOSITORY_ROOT = PHASE_DIR.parent
PHASE_2_DIR = REPOSITORY_ROOT / "2_model_architecture_study"
for dependency_dir in (
    PHASE_DIR,
    PHASE_2_DIR,
    PHASE_2_DIR / "2_tabular_data_adapter",
    PHASE_2_DIR / "3_sequence_data_adapter",
    PHASE_2_DIR / "4_model_adapters",
):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from base import load_model_adapter  # noqa: E402
from model_registry import ModelAdapterFactory  # noqa: E402
from tensorboard_monitoring import (  # noqa: E402
    TrainingRunContext,
    create_study_monitor,
)
from phase_3_common import (  # noqa: E402
    PHASE_2_SPECIFICATION_PATH,
    Phase3Error,
    artifacts_directory,
    atomic_replace,
    complete_manifest,
    final_model_path,
    final_preprocessor_path,
    invalidate_downstream_manifests,
    load_resolved_phase_3_settings,
    manifest_path,
    read_json,
    require_current_settings,
    training_contract_path,
    write_json,
)
from phase_3_data import first_rows, load_final_training_data  # noqa: E402
from phase_3_run_layout import SETTINGS_PATH, tensorboard_log_root  # noqa: E402


class FinalModelTrainingError(Phase3Error):
    """Explain a failed final fit, persistence check, or contract mismatch."""


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")


def _write_state(
    run_number: int,
    settings_version: int,
    contract: dict[str, Any],
    status: str,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "manifest_version": 1,
        "settings_version": settings_version,
        "phase_2_run_number": contract["phase_2_run_number"],
        "phase_3_run_number": run_number,
        "status": status,
        "model_family": contract["model_family"],
        "configuration_id": contract["configuration_id"],
        "test_data_loaded": False,
    }
    if error:
        payload["error"] = error
    write_json(payload, manifest_path(4, run_number))


def train_final_model(run_number: int, *, force: bool = False) -> dict[str, Any]:
    settings = load_resolved_phase_3_settings(run_number)
    settings_version = int(settings["settings_version"])
    if not force and complete_manifest(4, run_number, settings_version) is not None:
        return read_json(
            artifacts_directory(4, run_number) / "final_training_summary.json",
            "final training summary",
        )
    if force:
        invalidate_downstream_manifests(4, run_number)
    if complete_manifest(3, run_number, settings_version) is None:
        raise FinalModelTrainingError("Phase 3 Step 3 is not complete")
    contract = read_json(training_contract_path(run_number), "final training contract")
    if contract.get("status") != "frozen" or contract.get("test_data_loaded") is not False:
        raise FinalModelTrainingError("Final training contract did not pass its gate")

    model_path = final_model_path(run_number)
    preprocessor_path = final_preprocessor_path(run_number)
    if force:
        model_path.unlink(missing_ok=True)
        preprocessor_path.unlink(missing_ok=True)
    _write_state(run_number, settings_version, contract, "running")

    try:
        training_data, preprocessor = load_final_training_data(contract)
        if len(training_data) != int(contract["training"]["row_count"]):
            raise FinalModelTrainingError("Final training row count changed")
        factory = ModelAdapterFactory(PHASE_2_SPECIFICATION_PATH)
        context = TrainingRunContext(
            stage="step_6",
            model_family=contract["model_family"],
            representation=contract["representation"],
            outer_fold=0,
            seed=int(contract["training"]["model_seed"]),
            configuration_id=contract["configuration_id"],
            feature_set=contract["input_schema"]["feature_set"],
            lookback=contract["input_schema"]["lookback"],
        )
        with create_study_monitor(
            stage="step_6",
            model_family=contract["model_family"],
            outer_fold=0,
            log_root=tensorboard_log_root(run_number),
        ) as study_monitor:
            with study_monitor.fit(context) as monitor:
                model = factory.create(
                    contract["model_family"],
                    contract["hyperparameters"],
                    seed=int(contract["training"]["model_seed"]),
                    training_iterations=contract["training"]["iterations_or_epochs"],
                    training_monitor=monitor,
                )
                summary = model.fit(training_data, None)
                model.detach_training_monitor()

        smoke_data = first_rows(training_data, 32)
        predictions_before = model.predict(smoke_data)
        temporary_model = _temporary_path(model_path)
        try:
            model.save(temporary_model)
            atomic_replace(temporary_model, model_path)
        finally:
            temporary_model.unlink(missing_ok=True)
        reloaded = load_model_adapter(model_path)
        predictions_after = reloaded.predict(smoke_data)
        if not np.allclose(
            predictions_before,
            predictions_after,
            rtol=1e-7,
            atol=1e-8,
        ):
            raise FinalModelTrainingError(
                "Reloaded model does not reproduce the smoke predictions"
            )

        if preprocessor is not None:
            temporary_preprocessor = _temporary_path(preprocessor_path)
            try:
                joblib.dump(preprocessor, temporary_preprocessor)
                atomic_replace(temporary_preprocessor, preprocessor_path)
            finally:
                temporary_preprocessor.unlink(missing_ok=True)
        else:
            preprocessor_path.unlink(missing_ok=True)

        artifact_dir = artifacts_directory(4, run_number)
        write_json(contract["input_schema"], artifact_dir / "final_feature_order.json")
        summary_payload = {
            **summary.to_dict(),
            "settings_version": settings_version,
            "phase_2_run_number": contract["phase_2_run_number"],
            "phase_3_run_number": run_number,
            "configuration_id": contract["configuration_id"],
            "training_uavs": int(contract["training"]["uav_count"]),
            "fixed_training_duration": contract["training"][
                "iterations_or_epochs"
            ],
            "smoke_rows": len(smoke_data),
            "reload_rtol": 1e-7,
            "reload_atol": 1e-8,
            "reload_prediction_equivalence": True,
            "test_data_loaded": False,
        }
        write_json(summary_payload, artifact_dir / "final_training_summary.json")
        manifest = {
            "manifest_version": 1,
            "settings_version": settings_version,
            "phase_2_run_number": contract["phase_2_run_number"],
            "phase_3_run_number": run_number,
            "status": "complete",
            "model_family": contract["model_family"],
            "configuration_id": contract["configuration_id"],
            "training_rows": len(training_data),
            "training_uavs": int(contract["training"]["uav_count"]),
            "reload_prediction_equivalence": True,
            "test_data_loaded": False,
            "artifacts": {
                "model": "artifacts/final_model.joblib",
                "preprocessor": (
                    "artifacts/final_preprocessor.joblib"
                    if preprocessor is not None
                    else None
                ),
                "feature_order": "artifacts/final_feature_order.json",
                "training_summary": "artifacts/final_training_summary.json",
            },
        }
        write_json(manifest, manifest_path(4, run_number))
        return summary_payload
    except (KeyboardInterrupt, SystemExit) as error:
        _write_state(
            run_number,
            settings_version,
            contract,
            "interrupted",
            str(error) or type(error).__name__,
        )
        raise
    except Exception as error:
        _write_state(run_number, settings_version, contract, "failed", str(error))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--settings", type=Path, default=SETTINGS_PATH)
    args = parser.parse_args()
    try:
        summary = train_final_model(
            require_current_settings(args.settings),
            force=args.force,
        )
    except (FinalModelTrainingError, Phase3Error, OSError, ValueError) as error:
        print(f"Final model training failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("Final model training complete")
    print(f"Family: {summary['model_family']}")
    print(f"Training rows: {summary['training_rows']}")


if __name__ == "__main__":
    main()
