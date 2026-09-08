"""PE_28: ten fixed representations crossed with two matched model recipes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from advanced_r2_utils import REPOSITORY_ROOT, load_workflow, method_report
from confirmation_utils import evaluation_jobs, generated_partitions, prediction_records, select_uavs, validate_partitions
from followup_experiment_utils import aligned_methods, atomic_csv, atomic_json, complete_cell, input_path, register_run
from feature_comparison_features import FEATURE_SETS, build_views

PHASE2 = REPOSITORY_ROOT / "2_architecture_experiments/2_model_architecture_study"
for directory in (PHASE2 / "4_model_adapters", PHASE2 / "2_tabular_data_adapter",
    REPOSITORY_ROOT / "1_dataset_construction/2_UAV_grouped_validation_folds"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from tabular_data_adapter import TabularDataAdapter  # noqa: E402
from feature_comparison_models import fit_run7, fit_simple  # noqa: E402

RECIPES = ("run7", "xgb_cat")
CONFIGURATIONS = {("control" if variant == "A_current" and recipe == "run7" else f"{recipe}__{variant}"):
                  (recipe, variant) for variant in FEATURE_SETS for recipe in RECIPES}
CONTRASTS = (("B_other", "A_current"), ("C_other_14", "B_other"),
    ("D_long_windows", "B_other"), ("E_compact_22", "B_other"),
    ("F_baseline_10", "B_other"), ("G_all_windows", "B_other"),
    ("G_all_windows", "D_long_windows"), ("H_compact_14", "C_other_14"),
    ("H_compact_14", "E_compact_22"), ("I_four_sensors", "C_other_14"), ("J_recent_slopes", "B_other"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_workflow(workflow):
    seeds = [workflow["screen_split_seed"], *workflow["confirmation_split_seeds"]]
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("One screen and two distinct confirmation seeds are required")
    if workflow["outer_fold_count"] < 2 or workflow["inner_fold_count"] < 2:
        raise ValueError("Grouped fold counts must be at least two")
    if list(workflow["feature_sets"]) != list(FEATURE_SETS) or list(workflow["recipes"]) != list(RECIPES):
        raise ValueError("PE_28 requires the frozen ten-by-two comparison")
    if workflow["target_cap"] != 125.0:
        raise ValueError("PE_28 fixes the fitting cap at 125")
    for stage, multiplier in (("screen", 1), ("confirmation", 2)):
        if not 1 <= workflow[f"{stage}_minimum_fold_wins"] <= multiplier * workflow["outer_fold_count"]:
            raise ValueError("Invalid stage fold-win threshold")
        threshold = workflow[f"{stage}_minimum_relative_rmse_improvement"]
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Invalid RMSE threshold")
    if not 0 < workflow["confirmation_minimum_pooled_r2"] < 1 or workflow["cpu_threads"] < 1:
        raise ValueError("Invalid R2 threshold or thread count")
    if type(workflow["max_workers"]) is not int or workflow["max_workers"] < 1:
        raise ValueError("max_workers must be a positive integer")
    simple = workflow["simple"]
    if not 0 < simple["stopping_fraction"] < .5 or simple["early_stopping_rounds"] < 1:
        raise ValueError("Invalid training-side stopping policy")
    if not simple["blend_weights"] or any(not np.isfinite(w) or not 0 <= w <= 1 for w in simple["blend_weights"]):
        raise ValueError("Invalid blend grid")


def prepare(workflow):
    adapter = TabularDataAdapter(input_path(workflow["tabular_manifest"]))
    training = adapter.load_training(workflow["feature_set"])
    development = adapter.load_development(workflow["feature_set"])
    if training.sample_weights is None or not np.isfinite(training.sample_weights).all() or (training.sample_weights <= 0).any():
        raise ValueError("Training weights must be positive and finite")
    if not np.allclose(training.sample_weights.groupby(training.metadata.uav_id).sum(), 1):
        raise ValueError("Training must retain equal total weight per UAV")
    if training.fitting_target is not None and not np.allclose(training.fitting_target, np.minimum(training.target, 125), rtol=0, atol=1e-10):
        raise ValueError("Precomputed training targets differ from the fixed cap")
    # Verify that replacing the calibration loader does not alter control endpoints.
    contract = json.loads(input_path(workflow["source_contract"]).read_text())
    if contract["internal_folds"] != workflow["inner_fold_count"]:
        raise ValueError("Run 7 internal calibration folds differ from the declared recipe")
    columns = ["sample_id", "uav_id", "scenario", "cutoff", "RUL", *training.features]
    original = pd.read_csv(input_path(contract["calibration_features_path"]), usecols=columns).sort_values("sample_id").reset_index(drop=True)
    current = development.metadata[["sample_id", "uav_id", "scenario", "cutoff"]].copy()
    current["RUL"] = development.target.to_numpy(float)
    current = pd.concat([current.reset_index(drop=True), development.features.reset_index(drop=True)], axis=1).sort_values("sample_id").reset_index(drop=True)
    if len(original) != len(current) or not original[["sample_id", "uav_id", "scenario"]].equals(current[["sample_id", "uav_id", "scenario"]]):
        raise ValueError("Run 7 control calibration endpoints changed")
    if not np.allclose(original[["cutoff", "RUL", *training.features]], current[["cutoff", "RUL", *training.features]], rtol=0, atol=1e-10):
        raise ValueError("Run 7 control calibration labels/features changed")
    started = time.perf_counter()
    views = build_views(training, development, pd.read_csv(input_path(workflow["raw_training"])))
    for variant, (train, dev) in views.items():
        if not np.isfinite(train.features.to_numpy(float)).all() or not np.isfinite(dev.features.to_numpy(float)).all():
            raise ValueError(f"Non-finite inputs in {variant}")
        if not set(contract["residual_features"]) <= set(train.features):
            raise ValueError(f"Run 7 residual inputs missing in {variant}")
    return adapter, views, time.perf_counter() - started


def cell_path(directory, job, method):
    return directory / f"{job.split_seed}_{job.outer_fold}_{method}.json"


def load_cell(path, job, method, training, held):
    if not path.is_file():
        return None
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    if digest(payload) != envelope["sha256"]:
        raise ValueError("Cell checkpoint checksum does not match")
    if payload["method"] != method or payload["feature_names"] != list(training.features):
        raise ValueError("Cell configuration or feature names changed")
    expected_uavs = sorted(map(str, training.metadata.uav_id.unique()))
    if payload["training_uavs"] != expected_uavs or payload["audit"]["calibration_uavs"] != expected_uavs:
        raise ValueError("Cell training/calibration UAV membership changed")
    if payload["feature_sha256"] != digest(list(training.features)) or payload["training_rows"] != len(training):
        raise ValueError("Cell feature representation or training rows changed")
    table = pd.DataFrame(payload["predictions"])
    expected = pd.DataFrame(prediction_records(job, held, {method: np.zeros(len(held))}))
    if not complete_cell(table, job, method, expected) or len(table) != len(expected) or table.predicted_rul.lt(0).any():
        raise ValueError("Cell prediction checkpoint is invalid")
    return payload


def execute_cell(job, method, training, calibration, held, workflow, path):
    """One process owns one atomic checkpoint; no shared reports are written here."""
    recipe, _ = CONFIGURATIONS[method]
    print(f"PE_28 seed={job.split_seed} fold={job.outer_fold} {method}: fitting", flush=True)
    fitter = fit_run7 if recipe == "run7" else fit_simple
    with threadpool_limits(limits=int(workflow["cpu_threads"])):
        prediction, audit = fitter(training, calibration, held, workflow)
    audit["worker_pid"] = os.getpid()
    records = prediction_records(job, held, {method: np.asarray(prediction, float)})
    payload = {"method": method, "training_uavs": sorted(map(str, job.training_uavs)),
        "training_rows": len(training), "feature_names": list(training.features),
        "feature_sha256": digest(list(training.features)), "audit": audit, "predictions": records}
    atomic_json(path, {"payload": payload, "sha256": digest(payload)})
    return load_cell(path, job, method, training, held)


def execute_pending(tasks, max_workers):
    """Bound in-flight work and return results in declared, not completion, order."""
    if not tasks:
        return []
    if max_workers == 1:
        return [execute_cell(*task) for task in tasks]
    results = [None] * len(tasks)
    remaining = iter(enumerate(tasks))
    # Explicit spawn is portable to Windows and avoids forking CUDA/thread pools.
    pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn"))
    active = {}
    try:
        for _ in range(min(max_workers, len(tasks))):
            index, task = next(remaining)
            active[pool.submit(execute_cell, *task)] = index
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            # Check completed outcomes before scheduling replacement work.
            for future in done:
                results[active.pop(future)] = future.result()
            for _ in done:
                index, task = next(remaining, (None, None))
                if index is not None:
                    active[pool.submit(execute_cell, *task)] = index
    finally:
        # Running cells finish/checkpoint; work not started is cancelled on errors.
        for future in active:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    return results


def fit_stage_cells(stage_jobs, methods, views, cells, workflow):
    payloads, pending, positions = [], [], []
    identities = [(job.split_seed, job.outer_fold, method) for job in stage_jobs for method in methods]
    if len(identities) != len(set(identities)):
        raise ValueError("Stage would dispatch duplicate cell checkpoints")
    for job in stage_jobs:
        for method in methods:
            _, variant = CONFIGURATIONS[method]
            train, dev = views[variant]
            training = select_uavs(train, job.training_uavs)
            held = select_uavs(dev, job.validation_uavs)
            path = cell_path(cells, job, method)
            payload = load_cell(path, job, method, training, held)
            if payload is None:
                positions.append(len(payloads))
                pending.append((job, method, training, select_uavs(dev, job.training_uavs), held, workflow, path))
            else:
                print(f"PE_28 seed={job.split_seed} fold={job.outer_fold} {method}: already complete", flush=True)
            payloads.append(payload)
    for index, payload in zip(positions, execute_pending(pending, workflow["max_workers"])):
        payloads[index] = payload
    return payloads


def report_contrasts(folds, destination):
    lookup = {value: key for key, value in CONFIGURATIONS.items()}
    pairs = [(lookup[(recipe, a)], lookup[(recipe, b)]) for recipe in RECIPES for a, b in CONTRASTS]
    pairs += [(lookup[("xgb_cat", variant)], lookup[("run7", variant)]) for variant in FEATURE_SETS]
    records = []
    for candidate, reference in pairs:
        left = folds.loc[folds.method.eq(candidate), ["outer_fold", "rmse"]]
        right = folds.loc[folds.method.eq(reference), ["outer_fold", "rmse"]]
        if left.empty or right.empty:
            continue
        joined = left.merge(right, on="outer_fold", suffixes=("_candidate", "_reference"), validate="one_to_one")
        if len(joined) != len(left) or len(joined) != len(right):
            raise ValueError("Contrast folds are incomplete")
        records.append({"candidate": candidate, "reference": reference, "folds": len(joined),
            "fold_wins": int((joined.rmse_candidate < joined.rmse_reference).sum()),
            "mean_rmse_change": float((joined.rmse_candidate - joined.rmse_reference).mean()),
            "relative_mean_rmse_improvement": float(1 - joined.rmse_candidate.mean() / joined.rmse_reference.mean()),
            "exploratory": True})
    atomic_csv(destination, pd.DataFrame(records))


def stage_report(root, name, payloads, workflow, challenger=None):
    reporting = root / "stages" / name / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    table = pd.concat([pd.DataFrame(p["predictions"]) for p in payloads], ignore_index=True)
    methods = sorted(set(table.method) - {"control"})
    predictions = {}
    for method in methods:
        rows = aligned_methods(table, method)
        predictions[method] = rows.challenger_prediction.to_numpy(float)
    predictions["control"] = rows.control_prediction.to_numpy(float)
    confirmation = name == "confirmation"
    result = method_report(rows, predictions, root=reporting.parent, control="control",
        minimum_fold_wins=workflow[f"{name}_minimum_fold_wins"],
        minimum_relative_rmse_improvement=workflow[f"{name}_minimum_relative_rmse_improvement"],
        promotion_allowed=confirmation, promotion_eligible_methods={challenger} if confirmation else set(methods),
        require_bootstrap_improvement=confirmation,
        minimum_pooled_r2=workflow["confirmation_minimum_pooled_r2"] if confirmation else None)
    costs = pd.DataFrame([{"method": p["method"], "feature_count": len(p["feature_names"]),
        "fit_seconds": p["audit"]["fit_seconds"], "predict_seconds": p["audit"]["predict_seconds"],
        "base_estimator_fits": p["audit"]["base_estimator_fits"]} for p in payloads])
    costs = costs.groupby("method", as_index=False).agg(feature_count=("feature_count", "first"),
        mean_fit_seconds=("fit_seconds", "mean"), mean_predict_seconds=("predict_seconds", "mean"),
        total_base_estimator_fits=("base_estimator_fits", "sum"))
    summary = pd.read_csv(reporting / "summary.csv").merge(costs, on="method", validate="one_to_one")
    atomic_csv(reporting / "summary_with_costs.csv", summary)
    report_contrasts(pd.read_csv(reporting / "fold_metrics.csv"), reporting / "matched_contrasts.csv")
    # Deterministic screen ranking among candidates meeting the frozen practical gate.
    if not confirmation:
        decisions = pd.read_csv(reporting / "promotion_decisions.csv")
        passing = set(decisions.loc[decisions.passes_gate, "method"])
        if passing:
            result["winner"] = str(summary.loc[summary.method.isin(passing)].sort_values(
                ["mean_rmse", "feature_count", "method"]).iloc[0].method)
    result["stage"] = name
    atomic_json(reporting / "winner_manifest.json", result)
    # Keep all twenty labels legible; overwrite the shared report's compact chart.
    import matplotlib.pyplot as plt
    ordered = summary.sort_values(["mean_rmse", "method"])
    figure, axis = plt.subplots(figsize=(11, max(4, .38 * len(ordered) + 1.5)))
    axis.barh(ordered.method, ordered.mean_rmse,
        color=["#287271" if name == result["winner"] else "#68768a" for name in ordered.method])
    axis.invert_yaxis()
    axis.set_xlabel("Mean grouped-fold RMSE (cycles; lower is better)")
    axis.set_title(f"PE_28 {name}: feature and model comparison")
    figure.tight_layout()
    figure.savefig(reporting / "comparison.png", dpi=150)
    plt.close(figure)
    return result


def gated_stages(root, workflow, screen_jobs, confirmation_jobs, fit_stage, report):
    """Freeze exactly one screen winner before any confirmation fit."""
    screen = report("screen", fit_stage(screen_jobs, list(CONFIGURATIONS)))
    selection = {"selected_configuration": screen["winner"], "screen_status": screen["status"],
                 "rule": "Pass screen gate, then lowest mean RMSE; ties use feature count and method name"}
    path = root / "reporting/selected_configuration.json"
    if path.is_file() and json.loads(path.read_text()) != selection:
        raise ValueError("Screen selection changed within a registered run")
    atomic_json(path, selection)
    if screen["status"] != "screening_candidate":
        directory = root / "stages/confirmation/reporting"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "winner_manifest.json", {"status": "skipped", "promoted": False,
            "reason": "No configuration passed the screening gate"})
        result = {**screen, "status": "no_promotion", "winner": "control", "promoted": False,
                  "confirmation_completed": False}
    else:
        candidate = selection["selected_configuration"]
        if candidate == "control" or candidate not in CONFIGURATIONS:
            raise ValueError("Screen selected an invalid challenger")
        result = report("confirmation", fit_stage(confirmation_jobs, ["control", candidate]), candidate)
        result["confirmation_completed"] = True
    result.update({"selected_configuration": selection["selected_configuration"],
        "retained_production_model": "phase3_run_7", "automatic_production_replacement": False,
        "uses_test_labels": False, "uses_locked_evaluation": False,
        "evaluation_caveat": "Screen and confirmation reuse previously examined UAVs on different splits; not an untouched test set."})
    return result


def run(workflow, root, *, check_only=False, force=False):
    validate_workflow(workflow)
    root = Path(root)
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    # Import both dependencies before costly data preparation or any scientific fitting.
    import catboost  # noqa: F401
    import xgboost  # noqa: F401
    adapter, views, feature_seconds = prepare(workflow)
    outer, inner = generated_partitions(history_summary_path=workflow["history_summary"],
        split_seeds=[workflow["screen_split_seed"], *workflow["confirmation_split_seeds"]],
        outer_fold_count=workflow["outer_fold_count"], inner_fold_count=workflow["inner_fold_count"])
    validate_partitions(outer, inner, expected_outer_folds=workflow["outer_fold_count"], expected_inner_folds=workflow["inner_fold_count"])
    jobs = evaluation_jobs(outer, inner, include_inner=False)
    for job in jobs:
        select_uavs(views["A_current"][0], job.training_uavs)
        select_uavs(views["A_current"][1], job.validation_uavs)
    screen_jobs = [job for job in jobs if job.split_seed == workflow["screen_split_seed"]]
    confirm_jobs = [job for job in jobs if job.split_seed != workflow["screen_split_seed"]]
    readiness = {"ready": True, "feature_sets": {key: value[0].features.shape[1] for key, value in views.items()},
        "configurations": len(CONFIGURATIONS), "screen_recipe_evaluations": len(screen_jobs) * len(CONFIGURATIONS),
        "maximum_confirmation_recipe_evaluations": len(confirm_jobs) * 2,
        "training_rows": len(views["A_current"][0]), "development_rows": len(views["A_current"][1]),
        "feature_build_seconds": feature_seconds, "uses_locked_evaluation": False, "uses_test_labels": False,
        "parallel_model_fits": workflow["max_workers"], "cpu_threads_per_worker": workflow["cpu_threads"]}
    if check_only:
        atomic_json(reporting / "input_verification.json", readiness)
        return readiness
    paths = [input_path(workflow[name]) for name in ("tabular_manifest", "history_summary", "source_contract",
        "specification", "raw_training", "reference_implementation")]
    paths += [adapter._copied_path(name) for name in ("training", "development", "feature_catalog")]
    contract = json.loads(input_path(workflow["source_contract"]).read_text())
    paths.append(input_path(contract["calibration_features_path"]))
    paths += [Path(__file__), *(Path(__file__).with_name(name) for name in ("feature_comparison_features.py",
        "feature_comparison_models.py", "advanced_r2_utils.py", "confirmation_utils.py", "followup_experiment_utils.py", "oof_experiment_utils.py"))]
    paths += list((PHASE2 / "4_model_adapters").rglob("*.py"))
    paths += [PHASE2 / "2_tabular_data_adapter/tabular_data_adapter.py",
              REPOSITORY_ROOT / "1_dataset_construction/2_UAV_grouped_validation_folds/create_uav_grouped_folds.py"]
    registered_workflow = {**workflow, "catboost_version": version("catboost"), "threadpoolctl_version": version("threadpoolctl")}
    register_run(reporting, registered_workflow, paths)
    cells = root / "cells"
    cells.mkdir(exist_ok=True)
    if force:
        for path in cells.glob("*.json"):
            path.resolve().relative_to(root.resolve())
            path.unlink()
        (reporting / "selected_configuration.json").unlink(missing_ok=True)
    (reporting / "winner_manifest.json").unlink(missing_ok=True)
    for stage in ("screen", "confirmation"):
        (root / "stages" / stage / "reporting/winner_manifest.json").unlink(missing_ok=True)
    allowed = {cell_path(cells, job, method).name for job in jobs for method in CONFIGURATIONS}
    if any(path.name not in allowed for path in cells.glob("*.json")):
        raise ValueError("Unexpected cell checkpoints in this run")
    # Audit every existing cell before allowing any new fitting.
    for job in jobs:
        for method, (_, variant) in CONFIGURATIONS.items():
            path = cell_path(cells, job, method)
            if path.is_file():
                train, dev = views[variant]
                load_cell(path, job, method, select_uavs(train, job.training_uavs), select_uavs(dev, job.validation_uavs))
    atomic_csv(reporting / "outer_folds.csv", outer)
    atomic_csv(reporting / "inner_folds.csv", inner)
    atomic_json(reporting / "feature_catalog.json", {variant: list(pair[0].features) for variant, pair in views.items()})

    def fit_stage(stage_jobs, methods):
        return fit_stage_cells(stage_jobs, methods, views, cells, workflow)

    result = gated_stages(root, workflow, screen_jobs, confirm_jobs, fit_stage,
        lambda name, payloads, challenger=None: stage_report(root, name, payloads, workflow, challenger))
    result.update({**readiness, "all_required_stages_completed": True, "completed_recipe_evaluations": len(list(cells.glob("*.json")))})
    atomic_json(reporting / "winner_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_28")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workflow, _, root = load_workflow(args.config, "feature_comparison_workflows", args.workflow)
    print(json.dumps(run(workflow, root, check_only=args.check, force=args.force), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
