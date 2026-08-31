"""Verify Phase 3 gates without opening locked scenarios or real test features."""

from __future__ import annotations

from collections import Counter
from io import StringIO
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd


PHASE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PHASE_DIR.parent
PHASE_2_DIR = REPOSITORY_ROOT / "2_architecture_experiments" / "2_model_architecture_study"
for dependency_dir in (
    PHASE_DIR,
    PHASE_DIR / "1_winning_architecture_selection",
    PHASE_DIR / "5_test_inference",
    PHASE_DIR / "6_submission_verification",
    PHASE_2_DIR / "2_tabular_data_adapter",
    PHASE_2_DIR / "3_sequence_data_adapter",
    PHASE_2_DIR / "3_trajectory_data_adapter",
    PHASE_2_DIR / "4_model_adapters",
):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

import run_test_inference as inference_module  # noqa: E402
import run_phase_3 as runner_module  # noqa: E402
import phase_3_common as common_module  # noqa: E402
from build_submission import (  # noqa: E402
    SubmissionVerificationError,
    _csv_round_trip,
    _validate_submission,
)
from phase_3_common import (  # noqa: E402
    complete_manifest,
    load_resolved_phase_3_settings,
    require_current_settings,
)
from phase_3_data import first_rows  # noqa: E402
from phase_3_run_layout import (  # noqa: E402
    SETTINGS_PATH,
    Phase3RunLayoutError,
    run_root,
    step_directory,
)
from sequence_data_adapter import SequenceDataAdapter  # noqa: E402
from tabular_data_adapter import TabularDataAdapter  # noqa: E402
from trajectory_data_adapter import TrajectoryDataAdapter  # noqa: E402
from verify_phase_3_settings import load_and_verify_settings  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class _SyntheticDataset:
    def __init__(self) -> None:
        self.metadata = pd.DataFrame(
            {
                "sample_id": ["sample_c", "sample_a", "sample_b"],
                "uav_id": ["uav_03", "uav_01", "uav_02"],
                "cutoff": [30, 10, 20],
            }
        )
        self.target = None
        self.sample_weights = None

    def __len__(self) -> int:
        return len(self.metadata)


class _SyntheticModel:
    family = "xgboost"

    def __init__(self, predictions: list[float]) -> None:
        self.predictions = np.asarray(predictions, dtype=float)

    def predict(self, _: Any) -> np.ndarray:
        return self.predictions.copy()


def _verify_layout() -> None:
    _require(run_root(3).name == "run_3", "Phase 3 run folder is misnamed")
    for step in range(1, 8):
        resolved = step_directory(step, run_number=3)
        _require(resolved.parent == run_root(3), f"Step {step} is outside its run")
    _require(
        set(runner_module.STEPS) == set(range(1, 8)),
        "Phase 3 runner does not contain all seven steps",
    )
    _require(
        runner_module.STEPS[7].script
        == PHASE_DIR / "7_post_run_reporting" / "build_phase_3_report.py",
        "Phase 3 Step 7 does not invoke post-run reporting",
    )
    for rejected in (0, -1, True):
        try:
            run_root(rejected)
        except Phase3RunLayoutError:
            continue
        raise RuntimeError(f"run_root accepted {rejected!r}")


def _verify_settings_and_selection() -> tuple[int, str]:
    settings, phase_2 = load_and_verify_settings(SETTINGS_PATH)
    _require(
        require_current_settings(SETTINGS_PATH) == settings.run_number,
        "Current TOML differs from the resolved Step 1 settings",
    )
    resolved = load_resolved_phase_3_settings(settings.run_number)
    expected_resolved = settings.model_dump(mode="json", exclude_none=True)
    if settings.selected_model_family == "calibrated_tree_blend":
        promoted_specification = resolved.get("phase_2_specification")
        _require(
            isinstance(promoted_specification, str)
            and Path(promoted_specification).name
            == "promoted_phase_2_specification.json",
            "Promoted ensemble did not resolve its generated Phase 2 specification",
        )
        expected_resolved["phase_2_specification"] = promoted_specification
    _require(
        resolved == expected_resolved,
        "Resolved settings do not reproduce the validated TOML",
    )
    _require(
        complete_manifest(1, settings.run_number, settings.settings_version)
        is not None,
        "Phase 3 Step 1 is not complete",
    )
    _require(
        phase_2.selected_family == settings.selected_model_family,
        "Phase 2 verification selected another family",
    )
    changed = settings.model_dump(mode="json", exclude_none=True)
    changed["final_search"]["model_seed"] += 1
    with patch.object(common_module.tomllib, "load", return_value=changed):
        try:
            require_current_settings(SETTINGS_PATH)
        except ValueError:
            pass
        else:
            raise RuntimeError("A downstream command accepted changed settings")
    return settings.run_number, settings.selected_model_family


def _verify_development_folds() -> None:
    adapter = TabularDataAdapter()
    folds = adapter.outer_fold_labels()
    _require(len(folds) == 5, f"Expected five folds, found {len(folds)}")
    validation_counts: Counter[str] = Counter()
    sample = None
    for fold in folds:
        split = adapter.get_final_search_split(fold, "screened_drift_pruned")
        training_ids = set(split.training.metadata["uav_id"].astype(str))
        validation_ids = set(split.validation.metadata["uav_id"].astype(str))
        _require(len(split.training) == 1600, f"Fold {fold} training rows changed")
        _require(len(split.validation) == 100, f"Fold {fold} validation rows changed")
        _require(len(training_ids) == 80, f"Fold {fold} training UAVs changed")
        _require(len(validation_ids) == 20, f"Fold {fold} validation UAVs changed")
        _require(not training_ids & validation_ids, f"Fold {fold} overlaps UAVs")
        _require(
            split.validation.metadata["scenario"].nunique() == 5,
            f"Fold {fold} does not contain five development scenarios",
        )
        weight_sums = (
            split.training.sample_weights.groupby(
                split.training.metadata["uav_id"].astype(str)
            )
            .sum()
            .to_numpy(dtype=float)
        )
        _require(
            np.allclose(weight_sums, 1.0, rtol=0, atol=1e-12),
            f"Fold {fold} does not weight UAVs equally",
        )
        validation_counts.update(validation_ids)
        sample = split.training
    _require(
        len(validation_counts) == 100 and set(validation_counts.values()) == {1},
        "Every training UAV must be held out exactly once",
    )
    _require(sample is not None and len(first_rows(sample, 7)) == 7, "Row slicing failed")

    sequence_adapter = SequenceDataAdapter()
    sequence = sequence_adapter.get_final_search_split(folds[0], 50)
    _require(sequence.training.scaled, "Sequence training telemetry is not scaled")
    _require(sequence.validation.scaled, "Sequence validation telemetry is not scaled")
    _require(
        np.all(sequence.training.sequences[sequence.training.padding_mask] == 0.0),
        "Scaled sequence padding changed from zero",
    )
    raw_sequences = sequence_adapter.load_training(50)
    scaled_sequences = sequence.channel_scaler.transform(raw_sequences)
    _require(
        np.array_equal(raw_sequences.side_features, scaled_sequences.side_features),
        "Telemetry preprocessing unexpectedly scaled sequence side features",
    )

    trajectory_adapter = TrajectoryDataAdapter()
    trajectory = trajectory_adapter.get_final_search_split(folds[0])
    _require(trajectory.training.scaled, "Trajectory training data is not scaled")
    _require(trajectory.validation.scaled, "Trajectory validation data is not scaled")
    _require(
        trajectory.training.reference_library is not None,
        "Trajectory training data has no reference library",
    )
    _require(
        trajectory.training.reference_library.scaled,
        "Trajectory references are not scaled",
    )
    _require(
        len(first_rows(trajectory.training, 7)) == 7,
        "Trajectory row slicing failed",
    )


def _verify_pre_inference_source_gate() -> None:
    forbidden = (".load_test(", ".load_locked(", "load_final_test_data")
    for step in range(1, 5):
        source_dir = PHASE_DIR / step_directory(step, run_number=1).name
        for path in source_dir.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                _require(
                    token not in source,
                    f"Pre-inference source {path.name} contains forbidden {token}",
                )


def _synthetic_contract() -> dict[str, Any]:
    return {
        "status": "frozen",
        "model_family": "xgboost",
        "configuration_id": "xgboost__candidate_001",
        "prediction_minimum": 0.0,
        "preprocessing": {"separate_artifact": False},
        "training": {"model_seed": 13, "uav_ids": ["training_uav"]},
        "test_contract": {
            "expected_rows": 3,
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
        },
    }


def _generate_synthetic_predictions(values: list[float]) -> pd.DataFrame:
    contract = _synthetic_contract()
    originals = (
        inference_module.read_json,
        inference_module.load_model_adapter,
        inference_module.load_final_test_data,
    )
    inference_module.read_json = lambda *_args, **_kwargs: contract
    inference_module.load_model_adapter = lambda *_args, **_kwargs: _SyntheticModel(
        values
    )
    inference_module.load_final_test_data = (
        lambda *_args, **_kwargs: _SyntheticDataset()
    )
    try:
        table, observed_contract = inference_module.generate_predictions(1)
    finally:
        (
            inference_module.read_json,
            inference_module.load_model_adapter,
            inference_module.load_final_test_data,
        ) = originals
    _require(observed_contract is contract, "Inference replaced the frozen contract")
    return table


def _verify_inference_and_submission() -> None:
    table = _generate_synthetic_predictions([3.5, 1.25, 2.0])
    expected_columns = _synthetic_contract()["test_contract"]["prediction_columns"]
    _require(list(table.columns) == expected_columns, "Prediction columns changed")
    _require(
        table["uav_id"].tolist() == ["uav_01", "uav_02", "uav_03"],
        "Predictions are not deterministically sorted",
    )
    submission = table.loc[:, ["uav_id", "RUL"]].rename(
        columns={"uav_id": "id"}
    )
    expected_ids = table["uav_id"].astype(str).tolist()
    _validate_submission(submission, expected_ids)

    numeric_submission = pd.DataFrame(
        {"id": [1, 2, 10], "RUL": [1.0, 2.0, 3.0]}
    )
    _validate_submission(numeric_submission, ["1", "2", "10"])

    reread = pd.read_csv(StringIO(submission.to_csv(index=False)))
    _require(
        np.array_equal(
            submission["RUL"].to_numpy(dtype=float),
            reread["RUL"].to_numpy(dtype=float),
        ),
        "CSV round-trip changed prediction values",
    )
    awkward_values = pd.DataFrame(
        {"uav_id": [1, 2], "RUL": [124.05750274658203, 185.94729614257812]}
    )
    _require(
        _csv_round_trip(awkward_values).equals(
            pd.read_csv(StringIO(awkward_values.to_csv(index=False)))
        ),
        "Canonical CSV comparison does not match persisted parsing",
    )
    duplicate = submission.copy()
    duplicate.loc[1, "id"] = duplicate.loc[0, "id"]
    try:
        _validate_submission(duplicate, expected_ids)
    except SubmissionVerificationError:
        pass
    else:
        raise RuntimeError("Submission verification accepted a duplicate UAV")

    try:
        _generate_synthetic_predictions([3.5, -1.0, 2.0])
    except inference_module.TestInferenceError:
        pass
    else:
        raise RuntimeError("Inference accepted a prediction below the frozen minimum")


def main() -> None:
    try:
        _verify_layout()
        run_number, family = _verify_settings_and_selection()
        _verify_development_folds()
        _verify_pre_inference_source_gate()
        _verify_inference_and_submission()
    except (RuntimeError, OSError, ValueError) as error:
        print(f"Phase 3 implementation verification failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("Phase 3 implementation verified")
    print(f"Phase 3 run: {run_number}")
    print(f"Selected family: {family}")
    print("Training/development folds: 5")
    print("Real locked/test data loaded: no")


if __name__ == "__main__":
    main()
