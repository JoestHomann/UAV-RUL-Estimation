"""Fit the confirmed history/TabPFN candidate on all UAVs and build submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from advanced_r2_utils import REPOSITORY_ROOT, load_workflow


PHASE1 = REPOSITORY_ROOT / "1_dataset_construction"
PHASE2 = REPOSITORY_ROOT / "2_architecture_experiments" / "2_model_architecture_study"
for directory in (
    PHASE1 / "5_prefix_feature_engineering",
    PHASE2 / "2_tabular_data_adapter",
    PHASE2 / "4_model_adapters",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from history_corrected_tree_system import HistoryCorrectedTreeSystem  # noqa: E402
from model_registry import ModelAdapterFactory  # noqa: E402
from no_op_training_monitor import NoOpTrainingMonitor  # noqa: E402
from run_tabular_prior import (  # noqa: E402
    fit_tabpfn,
    fitting_target,
    tabpfn_dependency_status,
)
from tabular_data_adapter import TabularDataAdapter  # noqa: E402


def _repository_file(value: str) -> Path:
    path = (REPOSITORY_ROOT / value).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Source path escapes the repository: {value}") from error
    if not path.is_file():
        raise ValueError(f"Required source is missing: {path}")
    return path


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_joblib(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _valid_component(path: Path, methods: set[str], expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return False
    required = {"sample_id", "uav_id", "cutoff", "method", "predicted_rul"}
    return (
        required.issubset(table)
        and set(table.method.astype(str)) == methods
        and len(table) == expected_rows * len(methods)
        and not table.duplicated(["sample_id", "method"]).any()
        and np.isfinite(table.predicted_rul.to_numpy(float)).all()
    )


def _validate_submission(table: pd.DataFrame, expected_ids: list[str]) -> None:
    if list(table.columns) != ["id", "RUL"]:
        raise ValueError("Submission must contain exactly id and RUL")
    ids = table.id.astype(str).tolist()
    if ids != sorted(ids) or ids != expected_ids or len(ids) != len(set(ids)):
        raise ValueError("Submission IDs are missing, duplicated, or out of order")
    values = table.RUL.to_numpy(float)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Submission predictions must be finite and nonnegative")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_23")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workflow, _, root = load_workflow(
        args.config, "final_candidate_workflows", args.workflow
    )
    reporting = root / "reporting"
    models = root / "models"
    reporting.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    final_manifest_path = reporting / "final_manifest.json"
    submission_path = reporting / "submission.csv"
    history_model_path = models / "history_corrected_tree_system.joblib"
    history_predictions_path = reporting / "history_component_predictions.csv"
    tabpfn_predictions_path = reporting / "tabpfn_component_predictions.csv"
    if args.force:
        for path in (
            final_manifest_path,
            submission_path,
            history_model_path,
            history_predictions_path,
            tabpfn_predictions_path,
            reporting / "component_predictions.csv",
        ):
            path.unlink(missing_ok=True)
    if final_manifest_path.is_file() and submission_path.is_file():
        manifest = _json(final_manifest_path)
        if manifest.get("status") == "complete":
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return

    combined_manifest_path = _repository_file(str(workflow["combined_winner_manifest"]))
    combined_manifest = _json(combined_manifest_path)
    required_winner = str(workflow["required_winner"])
    if combined_manifest.get("promoted") is not True or combined_manifest.get("winner") != required_winner:
        raise ValueError(
            f"Final fitting requires promoted winner {required_winner!r}; "
            f"observed {combined_manifest.get('winner')!r}"
        )
    candidate_path = _repository_file(str(workflow["candidate_contract"]))
    candidate = _json(candidate_path)
    if candidate.get("status") != "frozen" or candidate.get("method") != required_winner:
        raise ValueError("The combined final-candidate contract is not frozen")

    dependency = tabpfn_dependency_status(candidate["tabpfn"])
    if not dependency["ready"]:
        raise RuntimeError(
            "Final candidate fitting requires the pinned TabPFN dependency and checkpoint"
        )
    adapter = TabularDataAdapter(REPOSITORY_ROOT / candidate["tabular_manifest"])
    feature_set = str(candidate["feature_set"])
    training = adapter.load_training(feature_set)
    test = adapter.load_test(feature_set)
    raw_train = pd.read_csv(REPOSITORY_ROOT / candidate["train_csv"])
    raw_test = pd.read_csv(REPOSITORY_ROOT / candidate["test_csv"])
    expected_rows = int(workflow["expected_test_rows"])
    if len(test) != expected_rows:
        raise ValueError(f"Expected {expected_rows} test rows, found {len(test)}")

    if (
        not history_model_path.is_file()
        or not _valid_component(
            history_predictions_path,
            {"control", "prediction_history"},
            expected_rows,
        )
    ):
        if history_model_path.is_file():
            system = joblib.load(history_model_path)
        else:
            factory = ModelAdapterFactory(REPOSITORY_ROOT / candidate["specification"])
            base_adapter = factory.create(
                "residual_corrected_tree_ensemble",
                {"ensemble_contract_path": str(candidate["source_contract"])},
                seed=int(candidate["model_seed"]),
                allow_disabled=True,
                training_monitor=NoOpTrainingMonitor(),
            )
            system = HistoryCorrectedTreeSystem(
                base_adapter,
                history_lags=[int(value) for value in candidate["history_lags"]],
                feature_profile=str(candidate["feature_profile"]),
                near_cap_threshold=float(candidate["near_cap_threshold"]),
            )
            system.fit(training, raw_train)
            _atomic_joblib(system, history_model_path)
        predictions = system.predict_methods(test, raw_test)
        frames = []
        for method, values in predictions.items():
            frame = test.metadata[["sample_id", "uav_id", "cutoff"]].copy()
            frame["method"] = method
            frame["predicted_rul"] = np.asarray(values, dtype=float)
            frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(history_predictions_path, index=False)

    if not _valid_component(
        tabpfn_predictions_path,
        {"tabpfn_full"},
        expected_rows,
    ):
        tabpfn_prediction = fit_tabpfn(
            training.features,
            fitting_target(training, float(candidate["target_cap"])),
            test.features,
            {
                **candidate["tabpfn"],
                "checkpoint": dependency["resolved_checkpoint"],
            },
        )
        frame = test.metadata[["sample_id", "uav_id", "cutoff"]].copy()
        frame["method"] = "tabpfn_full"
        frame["predicted_rul"] = np.maximum(tabpfn_prediction, 0.0)
        frame.to_csv(tabpfn_predictions_path, index=False)

    history = pd.read_csv(history_predictions_path)
    history = history.loc[history.method.eq("prediction_history")].copy()
    tabpfn = pd.read_csv(tabpfn_predictions_path)
    keys = ["sample_id", "uav_id", "cutoff"]
    components = history[keys + ["predicted_rul"]].rename(
        columns={"predicted_rul": "prediction_history"}
    ).merge(
        tabpfn[keys + ["predicted_rul"]].rename(
            columns={"predicted_rul": "tabpfn_full"}
        ),
        on=keys,
        validate="one_to_one",
    )
    if len(components) != expected_rows:
        raise ValueError("Final component predictions do not cover every test UAV")
    weight = float(candidate["tabpfn_weight"])
    allowed = {float(value) for value in candidate["tabpfn_weight_grid"]}
    if weight not in allowed or not 0.0 <= weight <= 1.0:
        raise ValueError("Frozen TabPFN blend weight is invalid")
    components["tabpfn_weight"] = weight
    components["RUL"] = np.maximum(
        (1.0 - weight) * components.prediction_history.to_numpy(float)
        + weight * components.tabpfn_full.to_numpy(float),
        0.0,
    )
    components = components.sort_values("uav_id", kind="stable").reset_index(drop=True)
    components.to_csv(reporting / "component_predictions.csv", index=False)
    submission = components[["uav_id", "RUL"]].rename(columns={"uav_id": "id"})
    expected_ids = sorted(test.metadata.uav_id.astype(str).tolist())
    _validate_submission(submission, expected_ids)
    submission.to_csv(submission_path, index=False)
    _validate_submission(pd.read_csv(submission_path), expected_ids)

    checkpoint_path = Path(dependency["resolved_checkpoint"])
    manifest = {
        "manifest_version": 1,
        "status": "complete",
        "method": required_winner,
        "training_rows": len(training),
        "training_uavs": int(training.metadata.uav_id.nunique()),
        "test_rows": len(test),
        "test_uavs": int(test.metadata.uav_id.nunique()),
        "tabpfn_weight": weight,
        "weight_selection": candidate["weight_selection"],
        "candidate_contract": str(workflow["candidate_contract"]),
        "candidate_contract_sha256": _sha256(candidate_path),
        "combined_winner_manifest": str(workflow["combined_winner_manifest"]),
        "combined_winner_manifest_sha256": _sha256(combined_manifest_path),
        "tabpfn_version": dependency["installed_version"],
        "tabpfn_checkpoint": dependency["checkpoint"],
        "tabpfn_checkpoint_sha256": _sha256(checkpoint_path),
        "history_model_persisted": True,
        "tabpfn_predictions_checkpointed": True,
        "test_data_loaded": True,
        "test_labels_loaded": False,
        "test_metrics_calculated": False,
        "submission_verified": True,
        "artifacts": {
            "history_model": "models/history_corrected_tree_system.joblib",
            "history_predictions": "reporting/history_component_predictions.csv",
            "tabpfn_predictions": "reporting/tabpfn_component_predictions.csv",
            "component_predictions": "reporting/component_predictions.csv",
            "submission": "reporting/submission.csv",
        },
    }
    final_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
