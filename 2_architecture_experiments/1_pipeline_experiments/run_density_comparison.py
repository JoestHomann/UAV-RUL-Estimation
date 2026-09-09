"""PE_30: all-cycle training with the original simple recipe and equal UAV weights."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run_weight_scale_audit as source_tools
from advanced_r2_utils import load_workflow
from confirmation_utils import prediction_records, select_uavs
from feature_comparison_models import fit_simple, fit_run7
from followup_experiment_utils import atomic_csv, atomic_json, register_run

pe28 = source_tools.pe28
CANDIDATE = "original_dense"


def validate_workflow(workflow):
    seeds = [workflow["screen_split_seed"], *workflow["confirmation_split_seeds"]]
    if len(seeds) != 3 or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != 3:
        raise ValueError("Screen and two confirmation seeds must be distinct nonnegative integers")
    if workflow["reference_prefix_count"] != 20 or workflow["feature_variant"] != "B_other":
        raise ValueError("PE_30 fixes the 20-prefix reference and 266-feature B representation")
    for key in ("max_workers", "cpu_threads"):
        if type(workflow[key]) is not int or workflow[key] < 1:
            raise ValueError("Invalid worker/thread count")
    for stage in ("density", "reference", "confirmation"):
        wins = workflow[f"{stage}_minimum_fold_wins"]
        if type(wins) is not int or not 1 <= wins <= (10 if stage == "confirmation" else 5):
            raise ValueError("Invalid fold-win threshold")
        if not 0 < workflow[f"{stage}_minimum_relative_rmse_improvement"] < 1:
            raise ValueError("Invalid RMSE threshold")
    if not 0 < workflow["confirmation_minimum_pooled_r2"] < 1:
        raise ValueError("Invalid R2 threshold")


def prepare(workflow):
    # Reuse only PE_29's verified-source and causal-feature helpers. Its adjustment
    # and gating functions are deliberately not part of this experiment.
    source, raw, sparse_pair, current_pair, jobs, baselines, paths, outer, inner = source_tools.prepare(
        {**workflow, "audit_split_seed": workflow["screen_split_seed"]})
    print("PE_30: constructing and validating causal all-cycle features", flush=True)
    dense = source_tools.dense_view(raw)
    if len(dense) != len(raw) or list(dense.features) != list(sparse_pair[0].features):
        raise ValueError("Dense row coverage or feature schema differs")
    totals = dense.sample_weights.groupby(dense.metadata.uav_id).sum()
    if not np.allclose(totals, 1., rtol=0, atol=1e-12):
        raise ValueError("Dense data must retain total weight one per UAV")
    return source, (dense, sparse_pair[1]), current_pair, jobs, baselines, paths, outer, inner, len(sparse_pair[0])


def load_cell(path, job, method, training, held):
    payload = pe28.load_cell(path, job, method, training, held)
    if payload is not None and payload.get("training_sha256") != source_tools.training_digest(training):
        raise ValueError("PE_30 checkpoint training values, targets or weights changed")
    return payload


def execute_cell(job, method, training, calibration, held, source, path, threads):
    print(f"PE_30 seed={job.split_seed} fold={job.outer_fold} {method}: fitting {len(training)} rows", flush=True)
    if method not in {CANDIDATE, "run7"}:
        raise ValueError("Unknown PE_30 recipe")
    with threadpool_limits(limits=threads):
        prediction, audit = (fit_run7 if method == "run7" else fit_simple)(training, calibration, held, source)
    audit.update({"worker_pid": os.getpid(), "weight_sum": float(training.sample_weights.sum()),
        "weight_min": float(training.sample_weights.min()), "weight_max": float(training.sample_weights.max()),
        "sampling": "all_cycles" if method == CANDIDATE else "current20",
        "xgboost_parameters": source["simple"]["xgboost"] if method == CANDIDATE else None,
        "regularization_adjusted": False})
    payload = {"method": method, "training_uavs": sorted(map(str, job.training_uavs)), "training_rows": len(training),
        "feature_names": list(training.features), "feature_sha256": pe28.digest(list(training.features)),
        "training_sha256": source_tools.training_digest(training), "audit": audit,
        "predictions": prediction_records(job, held, {method: np.asarray(prediction, float)})}
    atomic_json(path, {"payload": payload, "sha256": pe28.digest(payload)})
    return load_cell(path, job, method, training, held)


def execute_pending(tasks, workers):
    if not tasks:
        return []
    if workers == 1:
        return [execute_cell(*task) for task in tasks]
    results = [None] * len(tasks)
    remaining, active = iter(enumerate(tasks)), {}
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


def gated_stages(root, screen_jobs, confirmation_jobs, baselines, fit, report):
    dense = fit(screen_jobs, CANDIDATE)
    density = report("density", baselines["legacy_sparse"], dense, baselines["run7"])
    reference = report("reference", baselines["run7"], dense)
    selection = {"selected_configuration": CANDIDATE,
        "density_passed": density["status"] == "screening_candidate",
        "reference_passed": reference["status"] == "screening_candidate"}
    selection["confirmation_eligible"] = selection["density_passed"] and selection["reference_passed"]
    path = root / "reporting/selected_configuration.json"
    if path.is_file() and json.loads(path.read_text()) != selection:
        raise ValueError("Frozen PE_30 screening decision changed")
    atomic_json(path, selection)
    if selection["confirmation_eligible"]:
        result = report("confirmation", fit(confirmation_jobs, "run7"), fit(confirmation_jobs, CANDIDATE))
    else:
        directory = root / "stages/confirmation/reporting"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "winner_manifest.json", {"status": "skipped", "promoted": False,
            "reason": "The dense candidate must pass both the sparse-baseline and Run 7 screening gates"})
        result = {**reference, "status": "no_promotion", "promoted": False, "winner": "control"}
    return {**result, **selection,
        "completed_stages": ["density", "reference"] + (["confirmation"] if selection["confirmation_eligible"] else []),
        "skipped_stages": [] if selection["confirmation_eligible"] else ["confirmation"],
        "confirmation_completed": selection["confirmation_eligible"],
        "retained_production_model": "phase3_run_7", "automatic_production_replacement": False,
        "uses_locked_evaluation": False, "uses_test_labels": False,
        "evaluation_caveat": "Screening reuses PE_28 endpoints that informed this hypothesis; confirmation uses new splits of previously examined UAVs."}


def run(workflow, root, *, check_only=False, force=False):
    validate_workflow(workflow)
    root = Path(root)
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    source, dense_pair, current_pair, jobs, baselines, paths, outer, inner, sparse_rows = prepare(workflow)
    screen_jobs = [job for job in jobs if job.split_seed == workflow["screen_split_seed"]]
    confirmation_jobs = [job for job in jobs if job.split_seed != workflow["screen_split_seed"]]
    readiness = {"ready": True, "screen_new_recipe_evaluations": len(screen_jobs), "reused_reference_cells": 2*len(screen_jobs),
        "confirmation_additional_recipe_evaluations": 2*len(confirmation_jobs),
        "maximum_new_recipe_evaluations": len(screen_jobs) + 2*len(confirmation_jobs),
        "sparse_training_rows": sparse_rows, "dense_training_rows": len(dense_pair[0]),
        "dense_terminal_rows": int(dense_pair[0].target.eq(0).sum()), "features": dense_pair[0].features.shape[1],
        "total_training_weight_per_uav": 1., "original_xgboost_parameters": source["simple"]["xgboost"],
        "regularization_adjusted": False, "parallel_workers": workflow["max_workers"],
        "uses_test_labels": False, "uses_locked_evaluation": False}
    if check_only:
        atomic_json(reporting / "input_verification.json", readiness)
        return readiness
    cells = root / "cells"
    if not (reporting / "pre_registration.json").is_file() and any(cells.glob("*.json")):
        raise ValueError("Cannot trust PE_30 cells without their pre-registration")
    register_run(reporting, {**workflow, "source_workflow": source}, [*paths, Path(__file__), Path(source_tools.__file__)])
    cells.mkdir(exist_ok=True)
    if force:
        for path in cells.glob("*.json"):
            path.resolve().relative_to(root.resolve())
            path.unlink()
        (reporting / "selected_configuration.json").unlink(missing_ok=True)
    (reporting / "winner_manifest.json").unlink(missing_ok=True)
    for stage in ("density", "reference", "confirmation"):
        (root / "stages" / stage / "reporting/winner_manifest.json").unlink(missing_ok=True)
    atomic_csv(reporting / "outer_folds.csv", outer)
    atomic_csv(reporting / "inner_folds.csv", inner)
    pairs = {CANDIDATE: dense_pair, "run7": current_pair}
    allowed = {pe28.cell_path(cells, job, method).name: (job, method)
        for job in jobs for method in pairs if method != "run7" or job in confirmation_jobs}
    for path in cells.glob("*.json"):
        if path.name not in allowed:
            raise ValueError("Unexpected PE_30 checkpoint")
        job, method = allowed[path.name]
        train, dev = pairs[method]
        load_cell(path, job, method, select_uavs(train, job.training_uavs), select_uavs(dev, job.validation_uavs))

    def fit(stage_jobs, method):
        train, dev = pairs[method]
        payloads, tasks, positions = [], [], []
        for job in stage_jobs:
            training, held = select_uavs(train, job.training_uavs), select_uavs(dev, job.validation_uavs)
            path = pe28.cell_path(cells, job, method)
            payload = load_cell(path, job, method, training, held)
            if payload is None:
                positions.append(len(payloads))
                tasks.append((job, method, training, select_uavs(dev, job.training_uavs), held, source, path, workflow["cpu_threads"]))
            else:
                print(f"PE_30 seed={job.split_seed} fold={job.outer_fold} {method}: already complete", flush=True)
            payloads.append(payload)
        for index, payload in zip(positions, execute_pending(tasks, workflow["max_workers"])):
            payloads[index] = payload
        return payloads

    result = gated_stages(root, screen_jobs, confirmation_jobs, baselines, fit,
        lambda name, left, right, anchor=None: source_tools.paired_report(root, name, left, right, workflow, anchor))
    result.update({**readiness, "all_required_stages_completed": True,
        "completed_new_recipe_evaluations": len(list(cells.glob("*.json")))})
    atomic_json(reporting / "winner_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_30")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workflow, _, root = load_workflow(args.config, "density_workflows", args.workflow)
    print(json.dumps(run(workflow, root, check_only=args.check, force=args.force), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
