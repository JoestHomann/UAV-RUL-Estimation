"""Refit Run 7 and an early-history specialist with nested UAV-only selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from advanced_r2_utils import REPOSITORY_ROOT, load_workflow, method_report, subset_dataset
from confirmation_utils import evaluation_jobs, generated_partitions, prediction_records, select_uavs, validate_partitions
from followup_experiment_utils import (
    aligned_methods, atomic_csv, atomic_json, cell_rows, complete_cell, input_path,
    register_run, select_routed_blend, validate_saved_nested,
)


PHASE2 = REPOSITORY_ROOT / "2_architecture_experiments" / "2_model_architecture_study"
for directory in (
    REPOSITORY_ROOT / "1_dataset_construction" / "2_UAV_grouped_validation_folds",
    PHASE2 / "2_tabular_data_adapter", PHASE2 / "4_model_adapters",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from model_registry import ModelAdapterFactory  # noqa: E402
from no_op_training_monitor import NoOpTrainingMonitor  # noqa: E402
from short_history_tree_system import ShortHistoryTreeEnsemble, early_prefixes  # noqa: E402
from tabular_data_adapter import TabularDataAdapter  # noqa: E402


METHODS = {"control", "short_history_specialist"}


def new_model(workflow: dict, *, specialist: bool) -> Any:
    factory = ModelAdapterFactory(input_path(workflow["specification"]))
    control = factory.create(
        "residual_corrected_tree_ensemble", {"ensemble_contract_path": workflow["source_contract"]},
        seed=int(workflow["model_seed"]), allow_disabled=True, training_monitor=NoOpTrainingMonitor(),
    )
    if control.target_policy.mode != "piecewise_cap" or control.target_policy.maximum_rul != float(workflow["target_cap"]):
        raise ValueError("Run 7 source target policy differs from the declared cap")
    if not specialist:
        return control
    model = ShortHistoryTreeEnsemble(
        maximum_cutoff=float(workflow["maximum_history"]), hyperparameters=control.hyperparameters,
        seed=control.seed, prediction_minimum=control.prediction_minimum, training_monitor=NoOpTrainingMonitor(),
    )
    model.configure_policies(control.target_policy, control.prediction_policy)
    return model


def routed_specialist_predictions(model: Any, validation: Any, control: np.ndarray, maximum_history: float) -> np.ndarray:
    """Predict only within the specialist's history support; use control elsewhere."""
    mask = validation.metadata.cutoff.to_numpy(float) <= maximum_history
    prediction = np.asarray(control, float).copy()
    if mask.any():
        prediction[mask] = model.predict(subset_dataset(validation, mask))
    return prediction


def run(workflow: dict, root: Path, *, check_only: bool = False, force: bool = False) -> dict:
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    if len(set(workflow["split_seeds"])) != len(workflow["split_seeds"]):
        raise ValueError("Split seeds must be distinct")
    if not 0.0 < float(workflow["maximum_history"]):
        raise ValueError("Maximum history must be positive")
    if not workflow["specialist_weights"] or 0.0 not in workflow["specialist_weights"] or any(
        not np.isfinite(value) or not 0 <= value <= 1 for value in workflow["specialist_weights"]
    ):
        raise ValueError("Specialist weights must be finite in [0,1] and include zero")
    outer, inner = generated_partitions(
        history_summary_path=workflow["history_summary"], split_seeds=workflow["split_seeds"],
        outer_fold_count=int(workflow["outer_fold_count"]), inner_fold_count=int(workflow["inner_fold_count"]),
    )
    validate_partitions(outer, inner, expected_outer_folds=int(workflow["outer_fold_count"]),
                        expected_inner_folds=int(workflow["inner_fold_count"]))
    jobs = evaluation_jobs(outer, inner, include_inner=True)
    adapter = TabularDataAdapter(input_path(workflow["tabular_manifest"]))
    training = adapter.load_training(workflow["feature_set"])
    development = adapter.load_development(workflow["feature_set"])
    if training.fitting_target is not None and not np.allclose(
        training.fitting_target, np.minimum(training.target, float(workflow["target_cap"])), rtol=0, atol=1e-10
    ):
        raise ValueError("Precomputed fitting targets differ from cap-125 control")
    if training.sample_weights is None or not np.allclose(
        training.sample_weights.groupby(training.metadata.uav_id).sum(), 1.0
    ):
        raise ValueError("Control training weights must sum to one per UAV")
    prototype = new_model(workflow, specialist=True)
    preflight = []
    for job in jobs:
        fold_training = select_uavs(training, job.training_uavs)
        select_uavs(development, job.validation_uavs)
        short = early_prefixes(fold_training, float(workflow["maximum_history"]))
        calibration = prototype._calibration_data(short)
        preflight.append({"split_seed": job.split_seed, "source_outer_fold": job.outer_fold,
            "evaluation_level": job.evaluation_level, "inner_fold": job.inner_fold,
            "control_training_rows": len(fold_training), "specialist_training_rows": len(short),
            "training_uavs": int(short.metadata.uav_id.nunique()),
            "specialist_calibration_rows": len(calibration),
            "specialist_calibration_uavs": int(calibration.metadata.uav_id.nunique())})
    readiness = {"ready": True, "evaluation_jobs": len(jobs), "expected_model_fits": 2 * len(jobs),
                 "uses_test_labels": False, "uses_locked_evaluation": False}
    if check_only:
        atomic_csv(reporting / "input_coverage.csv", pd.DataFrame(preflight))
        atomic_json(reporting / "input_verification.json", readiness)
        return readiness

    contract = json.loads(input_path(workflow["source_contract"]).read_text(encoding="utf-8"))
    paths = [input_path(workflow[key]) for key in ("tabular_manifest", "history_summary", "specification", "source_contract")]
    paths.append(input_path(contract["calibration_features_path"]))
    for name in ("training", "development", "feature_catalog"):
        paths.append(adapter._copied_path(name))
    paths.extend([Path(__file__), Path(__file__).with_name("short_history_tree_system.py"),
                  Path(__file__).with_name("followup_experiment_utils.py"),
                  Path(__file__).with_name("confirmation_utils.py"), Path(__file__).with_name("advanced_r2_utils.py")])
    paths.extend((PHASE2 / "4_model_adapters").rglob("*.py"))
    paths.extend([PHASE2 / "2_tabular_data_adapter" / "tabular_data_adapter.py",
                  REPOSITORY_ROOT / "1_dataset_construction" / "2_UAV_grouped_validation_folds" / "create_uav_grouped_folds.py"])
    register_run(reporting, workflow, paths)
    atomic_csv(reporting / "outer_folds.csv", outer)
    atomic_csv(reporting / "inner_folds.csv", inner)
    atomic_csv(reporting / "input_coverage.csv", pd.DataFrame(preflight))
    checkpoint = reporting / "fold_predictions.csv"
    if force:
        checkpoint.unlink(missing_ok=True)
    (reporting / "winner_manifest.json").unlink(missing_ok=True)
    completed = pd.read_csv(checkpoint) if checkpoint.is_file() else pd.DataFrame()
    for number, job in enumerate(jobs, start=1):
        fold_training = select_uavs(training, job.training_uavs)
        held = select_uavs(development, job.validation_uavs)
        expected = pd.DataFrame(prediction_records(job, held, {"control": np.zeros(len(held))}))
        for method in ("control", "short_history_specialist"):
            if complete_cell(completed, job, method, expected):
                print(f"PE_26 job {number}/{len(jobs)} {method}: already complete", flush=True)
                continue
            print(f"PE_26 job {number}/{len(jobs)} {method}: seed={job.split_seed} "
                  f"outer={job.outer_fold} inner={job.inner_fold}", flush=True)
            model = new_model(workflow, specialist=method != "control")
            model.fit(fold_training, None)
            if method == "control":
                predictions = model.predict(held)
            else:
                saved = cell_rows(completed, job, "control")
                identity = ["uav_id", "scenario", "cutoff", "observed_rul"]
                control = expected[identity].merge(saved[identity + ["predicted_rul"]], on=identity,
                    how="left", validate="one_to_one").predicted_rul.to_numpy(float)
                if not np.isfinite(control).all():
                    raise ValueError("Specialist has no aligned control fallback")
                predictions = routed_specialist_predictions(model, held, control, float(workflow["maximum_history"]))
            records = pd.DataFrame(prediction_records(job, held, {method: predictions}))
            completed = pd.concat([completed, records], ignore_index=True)
            atomic_csv(checkpoint, completed)
            del model
    validate_saved_nested(completed, outer, inner, methods=METHODS,
        outer_count=int(workflow["outer_fold_count"]), inner_count=int(workflow["inner_fold_count"]))
    held = aligned_methods(completed.loc[completed.evaluation_level.eq("outer")], "short_history_specialist")
    selection = aligned_methods(completed.loc[completed.evaluation_level.eq("inner")], "short_history_specialist")
    predictions, provenance = select_routed_blend(held, selection, weights=workflow["specialist_weights"],
        thresholds=[float(workflow["maximum_history"])], route_column="cutoff")
    atomic_csv(reporting / "selection_provenance.csv", provenance)
    primary = "short_history_blend"
    result = method_report(held, {"control": held.control_prediction.to_numpy(float), primary: predictions,
        "short_history_hard_route": held.challenger_prediction.to_numpy(float)}, root=root, control="control",
        promotion_eligible_methods={primary}, minimum_fold_wins=int(workflow["minimum_fold_wins"]),
        minimum_relative_rmse_improvement=float(workflow["minimum_relative_rmse_improvement"]),
        minimum_pooled_r2=float(workflow["minimum_pooled_r2"]), require_bootstrap_improvement=True)
    result.update({**readiness, "all_prediction_cells_completed": True, "nested_inner_predictions_complete": True,
        "split_seeds": workflow["split_seeds"], "maximum_history": float(workflow["maximum_history"]),
        "retained_production_model": "phase3_run_7", "automatic_production_replacement": False,
        "evaluation_caveat": "Fresh grouped splits of previously used UAVs; not an untouched test set.",
        "rule": "control + weight * (observed_history <= maximum_history) * (specialist - control)"})
    atomic_json(reporting / "winner_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_26")
    parser.add_argument("--check", action="store_true", help="Check data, policies and every fold's early-history coverage without fitting")
    parser.add_argument("--force", action="store_true", help="Restart model fits under the same registered recipe")
    args = parser.parse_args()
    workflow, _, root = load_workflow(args.config, "short_history_workflows", args.workflow)
    print(json.dumps(run(workflow, root, check_only=args.check, force=args.force), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
