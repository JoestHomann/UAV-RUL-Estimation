"""PE_29: fixed XGBoost weight-scale audit, gated density test and confirmation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_feature_comparison as pe28
from advanced_r2_utils import load_workflow, method_report
from confirmation_utils import evaluation_jobs, generated_partitions, prediction_records, select_uavs, validate_partitions
from feature_comparison_features import prefix_features, validate_raw
from feature_comparison_models import fit_simple, fit_run7
from followup_experiment_utils import aligned_methods, atomic_csv, atomic_json, input_path, register_run
from tabular_data_adapter import TabularDataset


def adjusted_workflow(source, row_weight):
    """Scale Hessian/gradient-unit parameters, leaving CatBoost and other XGB choices fixed."""
    if not np.isfinite(row_weight) or row_weight <= 0:
        raise ValueError("Weight scaling factor must be finite and positive")
    result = deepcopy(source)
    parameters = result["simple"]["xgboost"]
    for name, default in (("reg_lambda", 1.), ("reg_alpha", 0.), ("gamma", 0.), ("min_child_weight", 1.)):
        parameters[name] = float(parameters.get(name, default)) * row_weight
    return result


def validate_workflow(workflow):
    seeds = [workflow["audit_split_seed"], *workflow["confirmation_split_seeds"]]
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Audit and two confirmation seeds must be distinct nonnegative integers")
    if workflow["reference_prefix_count"] != 20 or workflow["feature_variant"] != "B_other":
        raise ValueError("PE_29 fixes the reference at 20 prefixes and the 266-feature B representation")
    if any(type(workflow[key]) is not int or workflow[key] < 1 for key in ("max_workers", "cpu_threads")):
        raise ValueError("Invalid process/thread count")
    for name in ("audit", "density", "reference", "confirmation"):
        if not 0 < workflow[f"{name}_minimum_relative_rmse_improvement"] < 1:
            raise ValueError("Invalid stage RMSE threshold")
        wins = workflow[f"{name}_minimum_fold_wins"]
        if type(wins) is not int or not 1 <= wins <= (10 if name == "confirmation" else 5):
            raise ValueError("Invalid stage fold-win threshold")
    if not 0 < workflow["confirmation_minimum_pooled_r2"] < 1:
        raise ValueError("Invalid confirmation R2 threshold")


def verified_source(workflow):
    """Require the exact completed PE_28 source contract before reusing any predictions."""
    registration_path = input_path(workflow["source_registration"])
    root = registration_path.parent.parent
    registration = json.loads(registration_path.read_text())
    if registration["python_version"] != sys.version:
        raise ValueError("Python differs from the PE_28 source run")
    for name, expected in registration["package_versions"].items():
        if version(name) != expected:
            raise ValueError(f"PE_28 source package version changed: {name}")
    for name in ("catboost", "threadpoolctl"):
        if registration["workflow"][f"{name}_version"] != version(name):
            raise ValueError(f"PE_28 source package version changed: {name}")
    paths = [registration_path]
    for relative, expected in registration["input_sha256"].items():
        path = input_path(relative)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"PE_28 registered source changed: {relative}")
        paths.append(path)
    source = registration["workflow"]
    if source["screen_split_seed"] != workflow["audit_split_seed"] or source["outer_fold_count"] != 5 or source["inner_fold_count"] != 4:
        raise ValueError("Audit must reproduce PE_28's five screening folds and four calibration folds")
    if set(workflow["confirmation_split_seeds"]) & set(source["confirmation_split_seeds"]):
        raise ValueError("PE_29 confirmation must use seeds distinct from PE_28 confirmation")
    manifest_path = root / "reporting/winner_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("all_required_stages_completed") or not manifest.get("confirmation_completed"):
        raise ValueError("PE_28 source run is incomplete")
    paths.append(manifest_path)
    return root, source, paths


def sparse_view(data, histories):
    rows = []
    for record, target in zip(data.metadata.itertuples(), data.target):
        history = histories[str(record.uav_id)]
        cutoff = int(record.cutoff)
        if cutoff != record.cutoff or not 1 <= cutoff <= len(history):
            raise ValueError("Invalid sparse cutoff")
        if not np.isclose(history.RUL.iloc[cutoff-1], target, atol=1e-10, rtol=0):
            raise ValueError("Sparse endpoint labels differ from raw training data")
        rows.append(prefix_features(history, cutoff, "B_other"))
    return replace(data, features=pd.DataFrame(rows))


def dense_view(raw):
    """All observed training cycles, including RUL=0, with total weight one per UAV."""
    histories = validate_raw(raw)
    values, metadata, targets, weights = [], [], [], []
    for uav, history in histories.items():
        for row in history.itertuples():
            cutoff = int(row.flight_cycle)
            values.append(prefix_features(history, cutoff, "B_other"))
            metadata.append({"sample_id": f"dense_{uav}_{cutoff}", "uav_id": uav, "cutoff": cutoff})
            targets.append(float(row.RUL))
            weights.append(1. / len(history))
    return TabularDataset(pd.DataFrame(values), pd.DataFrame(metadata), pd.Series(targets), pd.Series(weights))


def prepare(workflow):
    source_root, source, paths = verified_source(workflow)
    adapter = pe28.TabularDataAdapter(input_path(source["tabular_manifest"]))
    current_train = adapter.load_training(source["feature_set"])
    current_dev = adapter.load_development(source["feature_set"])
    raw = pd.read_csv(input_path(source["raw_training"]))
    histories = validate_raw(raw)
    if set(histories) != set(current_train.metadata.uav_id) or set(histories) != set(current_dev.metadata.uav_id):
        raise ValueError("Training/development UAV coverage differs from raw trajectories")
    counts = current_train.metadata.groupby("uav_id").size()
    weight = 1. / workflow["reference_prefix_count"]
    if not counts.eq(workflow["reference_prefix_count"]).all() or current_train.sample_weights is None or not np.allclose(current_train.sample_weights, weight):
        raise ValueError("Source training data does not have 20 prefixes with weight 0.05 each")
    sparse, dev = sparse_view(current_train, histories), sparse_view(current_dev, histories)
    outer, inner = generated_partitions(history_summary_path=source["history_summary"],
        split_seeds=[workflow["audit_split_seed"], *workflow["confirmation_split_seeds"]],
        outer_fold_count=source["outer_fold_count"], inner_fold_count=source["inner_fold_count"])
    validate_partitions(outer, inner, expected_outer_folds=5, expected_inner_folds=4)
    jobs = evaluation_jobs(outer, inner, include_inner=False)
    audit_jobs = [job for job in jobs if job.split_seed == workflow["audit_split_seed"]]
    baselines = {"legacy_sparse": [], "run7": []}
    for job in audit_jobs:
        for key, method, train, validation in (("legacy_sparse", "xgb_cat__B_other", sparse, dev),
                                              ("run7", "control", current_train, current_dev)):
            path = pe28.cell_path(source_root / "cells", job, method)
            payload = pe28.load_cell(path, job, method, select_uavs(train, job.training_uavs), select_uavs(validation, job.validation_uavs))
            if payload is None:
                raise ValueError(f"Missing completed PE_28 reference cell: {path}")
            baselines[key].append(payload)
            paths.append(path)
    return source, raw, (sparse, dev), (current_train, current_dev), jobs, baselines, paths, outer, inner


def training_digest(data):
    digest = hashlib.sha256()
    for frame in (data.features, data.metadata[["uav_id", "cutoff"]], data.target, data.sample_weights):
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def load_cell(path, job, method, training, held):
    payload = pe28.load_cell(path, job, method, training, held)
    if payload is not None and payload["training_sha256"] != training_digest(training):
        raise ValueError("Training values, targets or sample weights changed in a PE_29 checkpoint")
    return payload


def execute_cell(job, method, training, calibration, held, fit_workflow, path, threads):
    print(f"PE_29 seed={job.split_seed} fold={job.outer_fold} {method}: fitting {len(training)} rows", flush=True)
    fitter = fit_run7 if method == "run7" else fit_simple
    with threadpool_limits(limits=threads):
        prediction, audit = fitter(training, calibration, held, fit_workflow)
    audit.update({"worker_pid": os.getpid(), "weight_sum": float(training.sample_weights.sum()),
        "weight_min": float(training.sample_weights.min()), "weight_max": float(training.sample_weights.max()),
        "sampling": "all_cycles" if method == "scaled_dense" else "current20",
        "xgboost_parameters": fit_workflow["simple"]["xgboost"] if method != "run7" else None})
    payload = {"method": method, "training_uavs": sorted(map(str, job.training_uavs)), "training_rows": len(training),
        "feature_names": list(training.features), "feature_sha256": pe28.digest(list(training.features)),
        "training_sha256": training_digest(training), "audit": audit,
        "predictions": prediction_records(job, held, {method: np.asarray(prediction, float)})}
    atomic_json(path, {"payload": payload, "sha256": pe28.digest(payload)})
    return load_cell(path, job, method, training, held)


def execute_pending(tasks, workers):
    if not tasks:
        return []
    if workers == 1:
        return [execute_cell(*task) for task in tasks]
    results = [None] * len(tasks)
    remaining = iter(enumerate(tasks))
    active = {}
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        for _ in range(min(workers, len(tasks))):
            index, task = next(remaining)
            active[pool.submit(execute_cell, *task)] = index
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                results[active.pop(future)] = future.result()
            for _ in done:
                index, task = next(remaining, (None, None))
                if index is not None:
                    active[pool.submit(execute_cell, *task)] = index
    finally:
        for future in active:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    return results


def paired_report(root, name, reference, candidate, workflow, anchor=None):
    directory = root / "stages" / name / "reporting"
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for label, payloads in [("control", reference), (candidate[0]["method"], candidate), ("run7_anchor", anchor)]:
        if payloads is not None:
            frame = pd.concat([pd.DataFrame(p["predictions"]) for p in payloads], ignore_index=True)
            frame["method"] = label
            frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    candidate_name = candidate[0]["method"]
    aligned = aligned_methods(rows, candidate_name)
    predictions = {"control": aligned.control_prediction.to_numpy(float), candidate_name: aligned.challenger_prediction.to_numpy(float)}
    if anchor is not None:
        predictions["run7_anchor"] = aligned_methods(rows, "run7_anchor").challenger_prediction.to_numpy(float)
    confirmation = name == "confirmation"
    result = method_report(aligned, predictions, root=directory.parent, control="control",
        minimum_fold_wins=workflow[f"{name}_minimum_fold_wins"],
        minimum_relative_rmse_improvement=workflow[f"{name}_minimum_relative_rmse_improvement"],
        promotion_allowed=confirmation, promotion_eligible_methods={candidate_name}, require_bootstrap_improvement=confirmation,
        minimum_pooled_r2=workflow["confirmation_minimum_pooled_r2"] if confirmation else None)
    result.update({"reference_configuration": reference[0]["method"], "candidate_configuration": candidate_name,
        "stage": name, "exploratory": not confirmation})
    atomic_json(directory / "winner_manifest.json", result)
    return result


def gated_stages(root, workflow, audit_jobs, confirmation_jobs, baselines, fit, report):
    completed = []
    stages = ["audit", "density", "reference", "confirmation"]
    scaled = fit(audit_jobs, "scaled_sparse")
    result = report("audit", baselines["legacy_sparse"], scaled, baselines["run7"])
    completed.append("audit")
    selected = None
    if result["status"] == "screening_candidate":
        dense = fit(audit_jobs, "scaled_dense")
        density = report("density", scaled, dense, baselines["run7"])
        completed.append("density")
        chosen = dense if density["status"] == "screening_candidate" else scaled
        selected = chosen[0]["method"]
        selection = {"selected_configuration": selected, "density_passed": density["status"] == "screening_candidate"}
        path = root / "reporting/selected_configuration.json"
        if path.is_file() and json.loads(path.read_text()) != selection:
            raise ValueError("Registered screen selection changed")
        atomic_json(path, selection)
        result = report("reference", baselines["run7"], chosen)
        completed.append("reference")
        if result["status"] == "screening_candidate":
            result = report("confirmation", fit(confirmation_jobs, "run7"), fit(confirmation_jobs, selected))
            completed.append("confirmation")
    skipped = [name for name in stages if name not in completed]
    for name in skipped:
        directory = root / "stages" / name / "reporting"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "winner_manifest.json", {"status": "skipped", "promoted": False,
            "reason": f"Required gate did not pass after {completed[-1]}"})
    if "confirmation" not in completed:
        result = {**result, "status": "no_promotion", "promoted": False, "winner": "control"}
    result.update({"completed_stages": completed, "skipped_stages": skipped, "selected_configuration": selected,
        "retained_production_model": "phase3_run_7", "automatic_production_replacement": False,
        "confirmation_completed": "confirmation" in completed, "uses_locked_evaluation": False, "uses_test_labels": False,
        "evaluation_caveat": "Audit reuses PE_28 screening endpoints; confirmation uses new splits of previously examined UAVs."})
    return result


def run(workflow, root, *, check_only=False, force=False):
    validate_workflow(workflow)
    root = Path(root)
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    source, raw, sparse_pair, current_pair, jobs, baselines, paths, outer, inner = prepare(workflow)
    adjusted = adjusted_workflow(source, 1. / workflow["reference_prefix_count"])
    audit_jobs = [job for job in jobs if job.split_seed == workflow["audit_split_seed"]]
    confirmation_jobs = [job for job in jobs if job.split_seed != workflow["audit_split_seed"]]
    readiness = {"ready": True, "audit_new_recipe_evaluations": len(audit_jobs), "reused_reference_cells": 2*len(audit_jobs),
        "density_additional_recipe_evaluations": len(audit_jobs), "confirmation_additional_recipe_evaluations": 2*len(confirmation_jobs),
        "maximum_new_recipe_evaluations": 2*len(jobs), "sparse_training_rows": len(sparse_pair[0]), "dense_training_rows": len(raw),
        "features": sparse_pair[0].features.shape[1], "total_training_weight_per_uav": 1.,
        "adjusted_xgboost_parameters": adjusted["simple"]["xgboost"], "parallel_model_fits": workflow["max_workers"],
        "uses_test_labels": False, "uses_locked_evaluation": False}
    if check_only:
        atomic_json(reporting / "input_verification.json", readiness)
        return readiness
    paths.append(Path(__file__))
    if not (reporting / "pre_registration.json").is_file() and any((root / "cells").glob("*.json")):
        raise ValueError("Cannot trust PE_29 cells without their pre-registration")
    register_run(reporting, {**workflow, "source_workflow": source}, paths)
    cells = root / "cells"
    cells.mkdir(exist_ok=True)
    if force:
        for path in cells.glob("*.json"):
            path.resolve().relative_to(root.resolve())
            path.unlink()
        (reporting / "selected_configuration.json").unlink(missing_ok=True)
    (reporting / "winner_manifest.json").unlink(missing_ok=True)
    for stage in ("audit", "density", "reference", "confirmation"):
        (root / "stages" / stage / "reporting/winner_manifest.json").unlink(missing_ok=True)
    atomic_csv(reporting / "outer_folds.csv", outer)
    atomic_csv(reporting / "inner_folds.csv", inner)
    dense = None

    def data_for(method):
        nonlocal dense
        if method == "run7":
            return current_pair
        if method == "scaled_dense":
            if dense is None:
                print("PE_29: constructing causal all-cycle features", flush=True)
                dense = dense_view(raw)
            return dense, sparse_pair[1]
        return sparse_pair

    allowed = {pe28.cell_path(cells, job, method).name: (job, method)
        for job in jobs for method in ("scaled_sparse", "scaled_dense", "run7")
        if not (method == "run7" and job in audit_jobs)}
    for path in cells.glob("*.json"):
        if path.name not in allowed:
            raise ValueError("Unexpected PE_29 cell checkpoint")
        job, method = allowed[path.name]
        train, dev = data_for(method)
        load_cell(path, job, method, select_uavs(train, job.training_uavs), select_uavs(dev, job.validation_uavs))

    def fit(stage_jobs, method):
        train, dev = data_for(method)
        payloads, tasks, positions = [], [], []
        for job in stage_jobs:
            training, held = select_uavs(train, job.training_uavs), select_uavs(dev, job.validation_uavs)
            path = pe28.cell_path(cells, job, method)
            payload = load_cell(path, job, method, training, held)
            if payload is None:
                positions.append(len(payloads))
                tasks.append((job, method, training, select_uavs(dev, job.training_uavs), held,
                    source if method == "run7" else adjusted, path, workflow["cpu_threads"]))
            else:
                print(f"PE_29 seed={job.split_seed} fold={job.outer_fold} {method}: already complete", flush=True)
            payloads.append(payload)
        for index, payload in zip(positions, execute_pending(tasks, workflow["max_workers"])):
            payloads[index] = payload
        return payloads

    result = gated_stages(root, workflow, audit_jobs, confirmation_jobs, baselines, fit,
        lambda name, left, right, anchor=None: paired_report(root, name, left, right, workflow, anchor))
    result.update({**readiness, "all_required_stages_completed": True, "completed_new_recipe_evaluations": len(list(cells.glob("*.json")))})
    atomic_json(reporting / "winner_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_29")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workflow, _, root = load_workflow(args.config, "weight_scale_workflows", args.workflow)
    print(json.dumps(run(workflow, root, check_only=args.check, force=args.force), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
