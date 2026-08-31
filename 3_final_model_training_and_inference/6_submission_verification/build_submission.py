"""Build and independently verify the final Kaggle submission."""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
PHASE_DIR = STEP_DIR.parent
for dependency_dir in (PHASE_DIR, PHASE_DIR / "5_test_inference"):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from run_test_inference import generate_predictions  # noqa: E402
from phase_3_common import (  # noqa: E402
    Phase3Error,
    complete_manifest,
    final_run_manifest_path,
    invalidate_downstream_manifests,
    load_resolved_phase_3_settings,
    manifest_path,
    read_json,
    require_current_settings,
    submission_path,
    test_predictions_path,
    training_contract_path,
    write_csv,
    write_json,
)
from phase_3_run_layout import SETTINGS_PATH  # noqa: E402


class SubmissionVerificationError(Phase3Error):
    """Explain a prediction mismatch or invalid Kaggle upload table."""


def _csv_round_trip(table: pd.DataFrame) -> pd.DataFrame:
    """Apply the same decimal serialization boundary as persisted CSV output."""

    return pd.read_csv(StringIO(table.to_csv(index=False)))


def _validate_submission(
    table: pd.DataFrame,
    expected_uav_ids: list[str],
) -> None:
    if list(table.columns) != ["id", "RUL"]:
        raise SubmissionVerificationError("Submission must contain exactly id and RUL")
    observed_ids = table["id"].astype(str).tolist()
    sorted_ids = (
        table.sort_values("id", kind="stable")["id"].astype(str).tolist()
    )
    if observed_ids != sorted_ids:
        raise SubmissionVerificationError("Submission IDs are not sorted")
    if observed_ids != expected_uav_ids:
        raise SubmissionVerificationError("Submission UAV identifiers changed")
    if table["id"].duplicated().any():
        raise SubmissionVerificationError("Submission contains duplicate UAV IDs")
    values = pd.to_numeric(table["RUL"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise SubmissionVerificationError(
            "Submission RUL values must be finite and nonnegative"
        )


def build_submission(run_number: int, *, force: bool = False) -> dict[str, Any]:
    settings = load_resolved_phase_3_settings(run_number)
    settings_version = int(settings["settings_version"])
    if not force and complete_manifest(6, run_number, settings_version) is not None:
        return read_json(manifest_path(6, run_number), "submission manifest")
    if force:
        invalidate_downstream_manifests(6, run_number)
    if complete_manifest(5, run_number, settings_version) is None:
        raise SubmissionVerificationError("Phase 3 Step 5 is not complete")

    contract = read_json(training_contract_path(run_number), "final training contract")
    expected_columns = contract["test_contract"]["prediction_columns"]
    if contract["test_contract"].get("submission_columns") != ["id", "RUL"]:
        raise SubmissionVerificationError(
            "Frozen contract does not declare the Kaggle id,RUL schema"
        )
    expected_mapping = {
        "prediction_column": "uav_id",
        "submission_column": "id",
    }
    if (
        contract["test_contract"].get("submission_identifier_mapping")
        != expected_mapping
    ):
        raise SubmissionVerificationError(
            "Frozen contract does not map internal uav_id values to Kaggle id"
        )
    try:
        stored = pd.read_csv(test_predictions_path(run_number))
    except (OSError, pd.errors.ParserError) as error:
        raise SubmissionVerificationError(
            f"Cannot read test predictions: {error}"
        ) from error
    if list(stored.columns) != expected_columns:
        raise SubmissionVerificationError("Stored prediction columns changed")
    regenerated, _ = generate_predictions(run_number)
    identity_columns = [column for column in expected_columns if column != "RUL"]
    if not stored[identity_columns].equals(regenerated[identity_columns]):
        raise SubmissionVerificationError(
            "Stored prediction identities differ from regenerated inference"
        )
    canonical_regenerated = _csv_round_trip(regenerated)
    stored_values = stored["RUL"].to_numpy(dtype=float)
    regenerated_values = canonical_regenerated["RUL"].to_numpy(dtype=float)
    if not np.array_equal(stored_values, regenerated_values):
        raise SubmissionVerificationError(
            "Stored RUL values differ from regenerated predictions after the "
            "canonical CSV round trip"
        )

    submission = stored.loc[:, ["uav_id", "RUL"]].rename(
        columns={"uav_id": "id"}
    )
    submission = submission.sort_values("id", kind="stable").reset_index(drop=True)
    expected_ids = regenerated["uav_id"].astype(str).tolist()
    _validate_submission(submission, expected_ids)
    write_csv(submission, ["id", "RUL"], submission_path(run_number))

    reread = pd.read_csv(submission_path(run_number))
    _validate_submission(reread, expected_ids)
    manifest = {
        "manifest_version": 1,
        "settings_version": settings_version,
        "phase_2_run_number": contract["phase_2_run_number"],
        "phase_3_run_number": run_number,
        "status": "complete",
        "rows": len(reread),
        "columns": ["id", "RUL"],
        "identifier_mapping": expected_mapping,
        "identifier_set_verified": True,
        "finite_nonnegative_values_verified": True,
        "deterministic_order_verified": True,
        "regenerated_prediction_equivalence": True,
        "prediction_comparison_representation": "canonical_csv_round_trip",
        "conditional_calibration_applied": (
            contract.get("conditional_calibrator") is not None
        ),
        "test_metrics_calculated": False,
        "artifacts": {"submission": "artifacts/submission.csv"},
    }
    write_json(manifest, manifest_path(6, run_number))

    final_manifest = {
        "manifest_version": 1,
        "settings_version": settings_version,
        "phase_2_run_number": contract["phase_2_run_number"],
        "phase_3_run_number": run_number,
        "status": "complete",
        "model_family": contract["model_family"],
        "configuration_id": contract["configuration_id"],
        "step_manifests": {
            str(step): str(manifest_path(step, run_number).relative_to(
                final_run_manifest_path(run_number).parent
            ).as_posix())
            for step in range(1, 7)
        },
        "artifacts": {
            "settings": "1_winning_architecture_selection/artifacts/resolved_phase_3_settings.json",
            "architecture": "1_winning_architecture_selection/artifacts/selected_architecture.json",
            "configuration": "2_final_configuration_search/artifacts/selected_configuration.json",
            "training_contract": "3_final_training_contract/artifacts/final_training_contract.json",
            "model": "4_final_model_training/artifacts/final_model.joblib",
            "test_predictions": "5_test_inference/artifacts/test_predictions.csv",
            "submission": "6_submission_verification/artifacts/submission.csv",
        },
        "test_metrics_calculated": False,
    }
    write_json(final_manifest, final_run_manifest_path(run_number))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--settings", type=Path, default=SETTINGS_PATH)
    args = parser.parse_args()
    try:
        manifest = build_submission(
            require_current_settings(args.settings),
            force=args.force,
        )
    except (SubmissionVerificationError, Phase3Error, OSError, ValueError) as error:
        print(f"Submission verification failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("Submission verification complete")
    print(f"Rows: {manifest['rows']}")


if __name__ == "__main__":
    main()
