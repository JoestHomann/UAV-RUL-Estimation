"""Load test features for the first time and create frozen-model predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
PHASE_DIR = STEP_DIR.parent
REPOSITORY_ROOT = PHASE_DIR.parent
PHASE_2_DIR = REPOSITORY_ROOT / "2_architecture_experiments" / "2_model_architecture_study"
for dependency_dir in (
    PHASE_DIR,
    PHASE_2_DIR / "2_tabular_data_adapter",
    PHASE_2_DIR / "3_sequence_data_adapter",
    PHASE_2_DIR / "3_trajectory_data_adapter",
    PHASE_2_DIR / "4_model_adapters",
):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from base import load_model_adapter  # noqa: E402
from phase_3_common import (  # noqa: E402
    Phase3Error,
    complete_manifest,
    final_model_path,
    final_preprocessor_path,
    invalidate_downstream_manifests,
    load_resolved_phase_3_settings,
    manifest_path,
    read_json,
    require_current_settings,
    test_predictions_path,
    training_contract_path,
    write_csv,
    write_json,
)
from phase_3_data import load_final_test_data  # noqa: E402
from phase_3_run_layout import SETTINGS_PATH  # noqa: E402


class TestInferenceError(Phase3Error):
    """Explain a broken frozen-model or test-prediction contract."""


def generate_predictions(run_number: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Regenerate test predictions in memory from frozen artifacts."""

    contract = read_json(training_contract_path(run_number), "final training contract")
    if contract.get("status") != "frozen":
        raise TestInferenceError("Final training contract is not frozen")
    model = load_model_adapter(final_model_path(run_number))
    if model.family != contract["model_family"]:
        raise TestInferenceError("Saved model family differs from the contract")

    preprocessor = None
    if contract["preprocessing"]["separate_artifact"]:
        path = final_preprocessor_path(run_number)
        try:
            preprocessor = joblib.load(path)
        except Exception as error:
            raise TestInferenceError(
                f"Cannot load final preprocessor {path}: {error}"
            ) from error
    test_data = load_final_test_data(contract, preprocessor)
    if test_data.target is not None or test_data.sample_weights is not None:
        raise TestInferenceError("Test endpoints unexpectedly contain targets or weights")
    expected_metadata = contract["test_contract"]["metadata_columns"]
    missing_metadata = sorted(set(expected_metadata) - set(test_data.metadata.columns))
    if missing_metadata:
        raise TestInferenceError(f"Test metadata are missing {missing_metadata}")
    expected_rows = int(contract["test_contract"]["expected_rows"])
    if len(test_data) != expected_rows:
        raise TestInferenceError(
            f"Expected {expected_rows} test rows, found {len(test_data)}"
        )

    predictions = model.predict(test_data)
    if len(predictions) != expected_rows or not np.isfinite(predictions).all():
        raise TestInferenceError("Test predictions are missing or non-finite")
    minimum = float(contract["prediction_minimum"])
    if np.any(predictions < minimum):
        raise TestInferenceError("Test predictions violate the frozen minimum")

    table = test_data.metadata.loc[:, expected_metadata].copy()
    table["model_family"] = contract["model_family"]
    table["configuration_id"] = contract["configuration_id"]
    table["model_seed"] = int(contract["training"]["model_seed"])
    table["RUL"] = predictions.astype(float)
    expected_columns = contract["test_contract"]["prediction_columns"]
    table = table.loc[:, expected_columns]
    if table["uav_id"].astype(str).duplicated().any():
        raise TestInferenceError("Test prediction table has duplicate UAV IDs")
    training_uav_ids = set(str(value) for value in contract["training"]["uav_ids"])
    test_uav_ids = set(table["uav_id"].astype(str))
    overlap = sorted(training_uav_ids & test_uav_ids)
    if overlap:
        raise TestInferenceError(
            f"Test prediction table overlaps training UAVs: {overlap[:5]}"
        )
    table = table.sort_values("uav_id", kind="stable").reset_index(drop=True)
    return table, contract


def run_inference(run_number: int, *, force: bool = False) -> dict[str, Any]:
    settings = load_resolved_phase_3_settings(run_number)
    settings_version = int(settings["settings_version"])
    if not force and complete_manifest(5, run_number, settings_version) is not None:
        return read_json(manifest_path(5, run_number), "inference manifest")
    if force:
        invalidate_downstream_manifests(5, run_number)
    for prerequisite in (3, 4):
        if complete_manifest(prerequisite, run_number, settings_version) is None:
            raise TestInferenceError(f"Phase 3 Step {prerequisite} is not complete")

    table, contract = generate_predictions(run_number)
    write_csv(
        table,
        contract["test_contract"]["prediction_columns"],
        test_predictions_path(run_number),
    )
    manifest = {
        "manifest_version": 1,
        "settings_version": settings_version,
        "phase_2_run_number": contract["phase_2_run_number"],
        "phase_3_run_number": run_number,
        "status": "complete",
        "model_family": contract["model_family"],
        "configuration_id": contract["configuration_id"],
        "prediction_rows": len(table),
        "test_uav_count": int(table["uav_id"].nunique()),
        "test_uav_ids": table["uav_id"].astype(str).tolist(),
        "prediction_minimum": float(table["RUL"].min()),
        "prediction_maximum": float(table["RUL"].max()),
        "test_data_loaded": True,
        "test_target_loaded": False,
        "test_metrics_calculated": False,
        "artifacts": {"test_predictions": "artifacts/test_predictions.csv"},
    }
    write_json(manifest, manifest_path(5, run_number))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--settings", type=Path, default=SETTINGS_PATH)
    args = parser.parse_args()
    try:
        manifest = run_inference(
            require_current_settings(args.settings),
            force=args.force,
        )
    except (TestInferenceError, Phase3Error, OSError, ValueError) as error:
        print(f"Test inference failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("Test inference complete")
    print(f"Predictions: {manifest['prediction_rows']}")


if __name__ == "__main__":
    main()
