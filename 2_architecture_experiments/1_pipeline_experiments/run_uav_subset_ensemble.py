"""Screen four UAV-subset Run 7 ensembles, then conditionally confirm eight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from advanced_r2_utils import load_workflow, method_report, REPOSITORY_ROOT
from confirmation_utils import evaluation_jobs, generated_partitions, prediction_records, select_uavs, validate_partitions
from followup_experiment_utils import aligned_methods, atomic_csv, atomic_json, cell_rows, complete_cell, input_path, register_run


PHASE2 = REPOSITORY_ROOT / "2_architecture_experiments" / "2_model_architecture_study"
for directory in (
    PHASE2 / "4_model_adapters", PHASE2 / "2_tabular_data_adapter",
    REPOSITORY_ROOT / "1_dataset_construction" / "2_UAV_grouped_validation_folds",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from model_registry import ModelAdapterFactory  # noqa: E402
from no_op_training_monitor import NoOpTrainingMonitor  # noqa: E402
from tabular_data_adapter import TabularDataAdapter  # noqa: E402


def member_name(index: int) -> str:
    return f"subset_{index:02d}"


def uav_hash(uavs: Any) -> str:
    return hashlib.sha256(json.dumps(sorted(map(str, uavs))).encode("utf-8")).hexdigest()


def validate_workflow(workflow: dict) -> None:
    seeds = [int(workflow["screen_split_seed"]), *map(int, workflow["confirmation_split_seeds"])]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Screen and confirmation need distinct nonnegative split seeds")
    fraction = float(workflow["subset_fraction"])
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Subset fraction must be strictly between zero and one")
    if int(workflow["subset_seed"]) < 0:
        raise ValueError("Subset seed must be nonnegative")
    if int(workflow["screen_members"]) != 4 or int(workflow["expanded_members"]) != 8:
        raise ValueError("PE_27 predefines a four-member screen and eight-member expansion")
    folds = int(workflow["outer_fold_count"])
    if folds < 2 or int(workflow["inner_fold_count"]) < 2:
        raise ValueError("Grouped folds must be at least two")
    for key, maximum in (("screen_minimum_fold_wins", folds),
                         ("confirmation_minimum_fold_wins", folds * (len(seeds) - 1))):
        if not 1 <= int(workflow[key]) <= maximum:
            raise ValueError(f"Invalid {key}")
    for key in ("screen_minimum_relative_rmse_improvement", "confirmation_minimum_relative_rmse_improvement"):
        if not np.isfinite(workflow[key]) or not 0 < float(workflow[key]) < 1:
            raise ValueError(f"Invalid {key}")
    if not np.isfinite(workflow["confirmation_minimum_pooled_r2"]) or not 0 < workflow["confirmation_minimum_pooled_r2"] < 1:
        raise ValueError("Invalid confirmation R2 threshold")


def subset_members(job: Any, *, count: int, fraction: float, seed: int, minimum_uavs: int) -> list[frozenset[str]]:
    """Sample whole UAVs without replacement; membership depends on IDs/seeds only."""
    uavs = np.asarray(sorted(job.training_uavs), dtype=object)
    size = math.floor(len(uavs) * fraction)
    if not minimum_uavs <= size < len(uavs) or math.comb(len(uavs), size) < count:
        raise ValueError("Too few training UAVs for distinct subsets and internal calibration folds")
    if set(uavs) & set(job.validation_uavs):
        raise ValueError("Outer training and validation UAVs overlap")
    members = []
    for index in range(count):
        rng = np.random.default_rng(np.random.SeedSequence([seed, job.split_seed, job.outer_fold, index]))
        for _ in range(1000):
            chosen = frozenset(rng.choice(uavs, size=size, replace=False).tolist())
            if chosen not in members:
                members.append(chosen)
                break
        else:
            raise ValueError("Could not construct distinct UAV subsets")
    return members


def new_model(workflow: dict) -> Any:
    model = ModelAdapterFactory(input_path(workflow["specification"])).create(
        "residual_corrected_tree_ensemble", {"ensemble_contract_path": workflow["source_contract"]},
        seed=int(workflow["model_seed"]), allow_disabled=True, training_monitor=NoOpTrainingMonitor(),
    )
    if model.target_policy.mode != "piecewise_cap" or model.target_policy.maximum_rul != float(workflow["target_cap"]):
        raise ValueError("Source model does not use the declared capped fitting target")
    if model.internal_folds != int(workflow["inner_fold_count"]):
        raise ValueError("Source model internal calibration folds differ from the declared recipe")
    return model


def checked_calibration(model: Any, training: Any) -> Any:
    calibration = model._calibration_data(training)
    if set(calibration.metadata.uav_id) != set(training.metadata.uav_id):
        raise ValueError("Member calibration must contain exactly its training UAVs")
    return calibration


def expected_records(job: Any, held: Any) -> pd.DataFrame:
    return pd.DataFrame(prediction_records(job, held, {"control": np.zeros(len(held))}))


def checked_cell(table: pd.DataFrame, job: Any, method: str, held: Any, training: Any) -> bool:
    if not complete_cell(table, job, method, expected_records(job, held)):
        return False
    rows = cell_rows(table, job, method)
    uavs = set(training.metadata.uav_id)
    for column, expected in (("training_uav_sha256", uav_hash(uavs)),
                             ("calibration_uav_sha256", uav_hash(uavs)),
                             ("training_uavs", len(uavs)), ("training_rows", len(training))):
        if column not in rows or not rows[column].eq(expected).all():
            raise ValueError("Checkpoint training/calibration membership does not match the frozen subset")
    if rows.predicted_rul.lt(0).any():
        raise ValueError("Checkpoint has negative predictions")
    return True


def average_members(table: pd.DataFrame, jobs: list, count: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = table.loc[table.outer_fold.isin([job.report_fold for job in jobs])]
    first = aligned_methods(rows, member_name(0))
    members = [first.challenger_prediction.to_numpy(float)]
    for index in range(1, count):
        aligned = aligned_methods(rows, member_name(index))
        members.append(aligned.challenger_prediction.to_numpy(float))
    return first, {"control": first.control_prediction.to_numpy(float),
                   f"uav_subset_mean_{count}": np.mean(np.column_stack(members), axis=1)}


def run_stages(workflow: dict, root: Path, screen_jobs: list, confirmation_jobs: list, fit_stage: Any, report_stage: Any) -> dict:
    """Gate all additional fitting; confirmation rows never decide ensemble size."""
    completed_stages = []
    stages = [("screen_4", screen_jobs, 4, False), ("screen_8", screen_jobs, 8, False),
              ("confirmation_8", confirmation_jobs, 8, True)]
    for name, jobs, count, confirmation in stages:
        fit_stage(jobs, count)
        result = report_stage(name, jobs, count, confirmation)
        completed_stages.append(name)
        passing = bool(result["promoted"]) if confirmation else result["status"] == "screening_candidate"
        if not passing:
            final = {**result, "status": "no_promotion", "winner": "control", "promoted": False,
                     "stopped_after": name, "completed_stages": completed_stages,
                     "skipped_stages": [stage[0] for stage in stages[len(completed_stages):]]}
            break
    else:
        final = {**result, "completed_stages": completed_stages, "skipped_stages": []}
    final.update({"retained_production_model": "phase3_run_7", "automatic_production_replacement": False,
                  "uses_locked_evaluation": False, "uses_test_labels": False,
                  "confirmation_completed": "confirmation_8" in completed_stages,
                  "evaluation_caveat": "Confirmation uses different splits of the same previously examined UAVs, not an untouched test set."})
    for name in final["skipped_stages"]:
        directory = root / "stages" / name / "reporting"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "winner_manifest.json", {"status": "skipped", "promoted": False,
            "reason": "Earlier screen did not pass", "stopped_after": final["stopped_after"]})
    atomic_json(root / "reporting" / "winner_manifest.json", final)
    return final


def run(workflow: dict, root: Path, *, check_only: bool = False, force: bool = False) -> dict:
    validate_workflow(workflow)
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    seeds = [int(workflow["screen_split_seed"]), *map(int, workflow["confirmation_split_seeds"])]
    outer, inner = generated_partitions(history_summary_path=workflow["history_summary"], split_seeds=seeds,
        outer_fold_count=int(workflow["outer_fold_count"]), inner_fold_count=int(workflow["inner_fold_count"]))
    validate_partitions(outer, inner, expected_outer_folds=int(workflow["outer_fold_count"]),
                        expected_inner_folds=int(workflow["inner_fold_count"]))
    # The fixed uniform average has no learned outer blend weights. Each Run 7
    # fit still performs its own four internal UAV folds for residual correction.
    jobs = evaluation_jobs(outer, inner, include_inner=False)
    screen_jobs = [job for job in jobs if job.split_seed == seeds[0]]
    confirmation_jobs = [job for job in jobs if job.split_seed != seeds[0]]
    adapter = TabularDataAdapter(input_path(workflow["tabular_manifest"]))
    training = adapter.load_training(workflow["feature_set"])
    development = adapter.load_development(workflow["feature_set"])
    if training.sample_weights is None or not np.allclose(training.sample_weights.groupby(training.metadata.uav_id).sum(), 1.):
        raise ValueError("Training weights must sum to one per UAV")
    if training.fitting_target is not None and not np.allclose(training.fitting_target,
        np.minimum(training.target, float(workflow["target_cap"])), rtol=0, atol=1e-10):
        raise ValueError("Precomputed fitting targets differ from the declared cap")
    prototype = new_model(workflow)
    checked_calibration(prototype, training)
    plan, membership = {}, []
    coverage = []
    for job in jobs:
        select_uavs(development, job.validation_uavs)
        subsets = subset_members(job, count=8, fraction=float(workflow["subset_fraction"]),
            seed=int(workflow["subset_seed"]), minimum_uavs=prototype.internal_folds)
        methods = {"control": job.training_uavs, **{member_name(i): ids for i, ids in enumerate(subsets)}}
        plan[job.report_fold] = methods
        for method, uavs in methods.items():
            selected = select_uavs(training, uavs)
            for uav in sorted(uavs):
                membership.append({"split_seed": job.split_seed, "outer_fold": job.report_fold,
                                   "source_outer_fold": job.outer_fold, "method": method, "uav_id": uav})
            coverage.append({"split_seed": job.split_seed, "outer_fold": job.report_fold, "method": method,
                "training_uavs": len(uavs), "training_rows": len(selected), "validation_uavs": len(job.validation_uavs),
                "training_validation_uav_overlap": len(set(uavs) & set(job.validation_uavs)),
                "training_uav_sha256": uav_hash(uavs)})
    readiness = {"ready": True, "screen_model_fits": len(screen_jobs) * 5,
        "expansion_additional_fits": len(screen_jobs) * 4, "confirmation_additional_fits": len(confirmation_jobs) * 9,
        "maximum_model_fits": len(jobs) * 9, "sampling": "whole_uavs_without_replacement",
        "subset_fraction": float(workflow["subset_fraction"]), "uses_test_labels": False, "uses_locked_evaluation": False}
    if check_only:
        atomic_csv(reporting / "input_coverage.csv", pd.DataFrame(coverage))
        atomic_json(reporting / "input_verification.json", readiness)
        return readiness
    contract = json.loads(input_path(workflow["source_contract"]).read_text(encoding="utf-8"))
    paths = [input_path(workflow[name]) for name in ("tabular_manifest", "history_summary", "source_contract", "specification")]
    paths += [input_path(contract["calibration_features_path"]), *(adapter._copied_path(name) for name in ("training", "development", "feature_catalog"))]
    paths += [Path(__file__), *(Path(__file__).with_name(name) for name in (
        "advanced_r2_utils.py", "confirmation_utils.py", "followup_experiment_utils.py", "oof_experiment_utils.py"))]
    paths += list((PHASE2 / "4_model_adapters").rglob("*.py"))
    paths += [PHASE2 / "2_tabular_data_adapter" / "tabular_data_adapter.py",
        REPOSITORY_ROOT / "1_dataset_construction" / "2_UAV_grouped_validation_folds" / "create_uav_grouped_folds.py"]
    register_run(reporting, workflow, paths)
    atomic_csv(reporting / "outer_folds.csv", outer)
    atomic_csv(reporting / "inner_folds.csv", inner)
    atomic_csv(reporting / "member_uavs.csv", pd.DataFrame(membership))
    atomic_csv(reporting / "input_coverage.csv", pd.DataFrame(coverage))
    checkpoint = reporting / "fold_predictions.csv"
    if force:
        checkpoint.unlink(missing_ok=True)
    (reporting / "winner_manifest.json").unlink(missing_ok=True)
    for stage in ("screen_4", "screen_8", "confirmation_8"):
        (root / "stages" / stage / "reporting" / "winner_manifest.json").unlink(missing_ok=True)
    completed = pd.read_csv(checkpoint) if checkpoint.is_file() else pd.DataFrame()
    if not completed.empty:
        expected_cells = {(job.split_seed, job.outer_fold, "outer", -1, method)
                          for job in jobs for method in plan[job.report_fold]}
        keys = ["split_seed", "source_outer_fold", "evaluation_level", "inner_fold", "method"]
        actual_cells = set(map(tuple, completed[keys].drop_duplicates().to_numpy()))
        if not actual_cells <= expected_cells:
            raise ValueError("Checkpoint contains unexpected model cells")
        for job in jobs:
            held = select_uavs(development, job.validation_uavs)
            for method, uavs in plan[job.report_fold].items():
                checked_cell(completed, job, method, held, select_uavs(training, uavs))

    def fit_stage(stage_jobs: list, count: int) -> None:
        nonlocal completed
        for number, job in enumerate(stage_jobs, start=1):
            held = select_uavs(development, job.validation_uavs)
            for method in ["control", *map(member_name, range(count))]:
                selected = select_uavs(training, plan[job.report_fold][method])
                if checked_cell(completed, job, method, held, selected):
                    print(f"PE_27 fold {job.report_fold} {method}: already complete", flush=True)
                    continue
                print(f"PE_27 job {number}/{len(stage_jobs)} seed={job.split_seed} fold={job.outer_fold} "
                      f"{method}: {selected.metadata.uav_id.nunique()} training UAVs", flush=True)
                model = new_model(workflow)
                calibration = checked_calibration(model, selected)
                model.fit(selected, None)
                records = pd.DataFrame(prediction_records(job, held, {method: model.predict(held)}))
                records["training_uav_sha256"] = uav_hash(selected.metadata.uav_id.unique())
                records["calibration_uav_sha256"] = uav_hash(calibration.metadata.uav_id.unique())
                records["training_uavs"] = selected.metadata.uav_id.nunique()
                records["training_rows"] = len(selected)
                completed = pd.concat([completed, records], ignore_index=True)
                atomic_csv(checkpoint, completed)
                del model

    def report_stage(name: str, stage_jobs: list, count: int, confirmation: bool) -> dict:
        rows, predictions = average_members(completed, stage_jobs, count)
        stage_root = root / "stages" / name
        (stage_root / "reporting").mkdir(parents=True, exist_ok=True)
        prefix = "confirmation" if confirmation else "screen"
        result = method_report(rows, predictions, root=stage_root, control="control",
            minimum_fold_wins=int(workflow[f"{prefix}_minimum_fold_wins"]),
            minimum_relative_rmse_improvement=float(workflow[f"{prefix}_minimum_relative_rmse_improvement"]),
            promotion_allowed=confirmation, promotion_eligible_methods={f"uav_subset_mean_{count}"},
            require_bootstrap_improvement=confirmation,
            minimum_pooled_r2=float(workflow["confirmation_minimum_pooled_r2"]) if confirmation else None)
        result.update({"stage": name, "ensemble_members": count,
            "split_seeds": sorted({job.split_seed for job in stage_jobs}),
            "selection_scope": "confirmation_after_separate_split_screen" if confirmation else "development_screen",
            "uniform_weights": True})
        atomic_json(stage_root / "reporting" / "winner_manifest.json", result)
        return result

    result = run_stages(workflow, root, screen_jobs, confirmation_jobs, fit_stage, report_stage)
    result.update({**readiness, "completed_model_fits": len(completed.groupby(
        ["split_seed", "source_outer_fold", "method"])), "all_required_stages_completed": True})
    atomic_json(reporting / "winner_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_27")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true", help="Restart under the same registered settings and source hashes")
    args = parser.parse_args()
    workflow, _, root = load_workflow(args.config, "uav_subset_workflows", args.workflow)
    print(json.dumps(run(workflow, root, check_only=args.check, force=args.force), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
