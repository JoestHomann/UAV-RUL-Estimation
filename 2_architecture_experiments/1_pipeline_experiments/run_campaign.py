"""PE_31: reference audit, interaction benchmark and complete nested joint selection."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

import campaign_data as data_tools
import campaign_models as models
import campaign_reporting as reporting_tools
from advanced_r2_utils import load_workflow
from confirmation_utils import select_uavs
from followup_experiment_utils import atomic_csv, atomic_json, register_run


def digest(data):
    h = hashlib.sha256()
    for value in (data.features, data.metadata, data.target, data.sample_weights):
        h.update(b"None" if value is None else pd.util.hash_pandas_object(value, index=False).to_numpy().tobytes())
    h.update(json.dumps(list(data.features)).encode())
    return h.hexdigest()


def validate(workflow):
    if len(workflow["split_seeds"]) < 2 or len(workflow["model_seeds"]) < 2 or len(workflow["endpoint_seeds"]) < 3:
        raise ValueError("Campaign requires at least two partitions/model seeds and three endpoint draws")
    for key in ("split_seeds", "model_seeds", "endpoint_seeds"):
        values = workflow[key]
        if len(set(values)) != len(values) or any(type(v) is not int or v < 0 for v in values):
            raise ValueError("Invalid or duplicate seeds")
    if workflow["selection_folds"] != 3:
        raise ValueError("Frozen campaign uses three inner selection folds")
    for key in ("max_workers", "cpu_threads", "maximum_recipe_evaluations", "bootstrap_repetitions"):
        if type(workflow[key]) is not int or workflow[key] < 1:
            raise ValueError("Invalid resource budget")
    for key in ("minimum_relative_improvement", "nominal_maximum_regression", "stress_maximum_regression", "minimum_pooled_r2"):
        if not 0 < workflow[key] < 1:
            raise ValueError("Invalid decision threshold")
    if workflow["blend_weights"] != [0., .25, .5, .75, 1.]:
        raise ValueError("Frozen campaign blend grid changed")


def budget(workflow, jobs):
    tasks = len(jobs)*len(workflow["model_seeds"])
    # Benchmark 8; search upper bound 6*K + 12*K + K Run7 + three outer fits.
    # Cached duplicate base recipes and outer Run7 reduce actual fits.
    benchmark = tasks*8
    search = tasks*((6+12+1)*workflow["selection_folds"]+3)
    return {"outer_tasks_per_stage": tasks, "interaction_configurations": 6, "joint_search_configurations_per_outer_task": 12,
        "benchmark_recipe_evaluations": benchmark, "search_recipe_evaluations_upper_bound": search,
        "maximum_recipe_evaluations": benchmark+search,
        "maximum_base_estimator_fits": (benchmark+search)*30,
        "note": "Conservative upper bound; recipe evaluations include nested fits. No endpoint-draw refits. Exact reference replay is a separate 12-fit run."}


class Engine:
    def __init__(self, root, job, model_seed, source, data):
        self.root = Path(root); self.job = job; self.seed = model_seed; self.source = source; self.data = data
        self.directory = self.root / "cells" / f"{job.split_seed}_{job.outer_fold}_{model_seed}"
        self.directory.mkdir(parents=True, exist_ok=True)

    def fit(self, training_ids, held_ids, method="candidate", policy="sparse_uav_mass", recipe=None):
        training_ids, held_ids = set(training_ids), set(held_ids)
        if training_ids & held_ids or not training_ids or not held_ids:
            raise ValueError("Overlapping or empty UAV membership")
        recipe = recipe or data_tools.base_recipe()
        view = "run7" if method == "run7" else recipe["view"]
        key = "run7" if method == "run7" else data_tools.POLICIES[policy][0]+"_"+view
        training = select_uavs(self.data["train"][key], training_ids)
        calibration = select_uavs(self.data["dev"][view], training_ids)
        held = select_uavs(self.data["dev"][view], held_ids)
        identity = {"method": method, "policy": policy, "recipe": recipe, "training_uavs": sorted(training_ids), "held_uavs": sorted(held_ids)}
        name = data_tools.prior.pe28.digest(identity)
        path = self.directory / f"{name}.json"
        contract = {**identity, "training_digest": digest(training), "calibration_digest": digest(calibration), "held_digest": digest(held)}
        expected = held.metadata[["uav_id", "scenario", "suite", "endpoint_seed", "cutoff"]].copy()
        expected["observed_rul"] = held.target.to_numpy(float)
        expected["split_seed"] = self.job.split_seed; expected["outer_fold"] = self.job.outer_fold; expected["model_seed"] = self.seed
        if path.exists():
            obj = json.loads(path.read_text()); payload = obj["payload"]
            if obj["sha256"] != data_tools.prior.pe28.digest(payload) or payload["contract"] != contract:
                raise ValueError("Campaign checkpoint checksum or data contract changed")
            rows = pd.DataFrame(payload["predictions"])
            if len(rows) != len(held) or not np.isfinite(rows.predicted_rul).all():
                raise ValueError("Campaign checkpoint prediction coverage changed")
            if not rows[reporting_tools.KEYS].equals(expected[reporting_tools.KEYS]):
                raise ValueError("Campaign checkpoint endpoint identity changed")
            return rows[[*reporting_tools.KEYS, "predicted_rul"]]
        print(f"PE_31 split={self.job.split_seed} fold={self.job.outer_fold} seed={self.seed} {method}/{policy}/{recipe}: {len(training_ids)} train, {len(held_ids)} held UAVs", flush=True)
        if method == "run7":
            prediction, audit = models.fit_run7(training, models.training_endpoints(calibration, training_ids), held, self.source)
        elif method == "reference_protocol":
            prediction, audit = models.reference_protocol(self.data["raw"], training_ids, held, self.source, self.seed)
        else:
            prediction, audit = models.fit_candidate(training, calibration, held, self.source, policy, recipe, self.seed)
        if np.asarray(prediction).shape != (len(held),) or not np.isfinite(prediction).all():
            raise ValueError("Invalid fitted predictions")
        rows = held.metadata[["uav_id", "scenario", "suite", "endpoint_seed", "cutoff"]].copy()
        rows["observed_rul"] = held.target.to_numpy(float); rows["predicted_rul"] = prediction
        rows["split_seed"] = self.job.split_seed; rows["outer_fold"] = self.job.outer_fold; rows["model_seed"] = self.seed
        payload = {"contract": contract, "audit": audit, "predictions": rows.to_dict("records")}
        atomic_json(path, {"payload": payload, "sha256": data_tools.prior.pe28.digest(payload)})
        return rows[[*reporting_tools.KEYS, "predicted_rul"]]


def inner_groups(ids, seed, folds):
    values = np.asarray(sorted(ids))
    return [(set(values[train]), set(values[held])) for train, held in KFold(folds, shuffle=True, random_state=seed).split(values)]


def nested_search(engine, workflow):
    job = engine.job
    partitions = inner_groups(job.training_uavs, job.split_seed+job.outer_fold, workflow["selection_folds"])
    inner_control = pd.concat([engine.fit(train, held, "run7") for train, held in partitions], ignore_index=True)
    policy_scores = []
    inner_cache = {}
    for policy in data_tools.POLICIES:
        rows = pd.concat([engine.fit(train, held, policy=policy) for train, held in partitions], ignore_index=True)
        inner_cache[policy] = rows
        policy_scores.append({"policy": policy, "historical_rmse": reporting_tools.score(rows), "stress_rmse": reporting_tools.score(rows, "unrestricted")})
    policy = min(policy_scores, key=lambda r: (r["historical_rmse"], r["policy"]))["policy"]
    candidates = []; predictions = {}
    control_stress = reporting_tools.score(inner_control, "unrestricted")
    control_nominal = reporting_tools.score(inner_control, "nominal")
    for name, recipe in data_tools.recipes().items():
        rows = inner_cache[policy] if recipe == data_tools.base_recipe() else pd.concat(
            [engine.fit(train, held, policy=policy, recipe=recipe) for train, held in partitions], ignore_index=True)
        predictions[name] = rows
        for weight in workflow["blend_weights"]:
            blended = reporting_tools.blended(rows, inner_control, weight)
            candidates.append({"recipe": name, "weight": weight, "historical_rmse": reporting_tools.score(blended),
                "nominal_rmse": reporting_tools.score(blended, "nominal"), "stress_rmse": reporting_tools.score(blended, "unrestricted")})
    eligible = [row for row in candidates if row["stress_rmse"] <= control_stress*(1+workflow["stress_maximum_regression"])
                and row["nominal_rmse"] <= control_nominal*(1+workflow["nominal_maximum_regression"])]
    # Zero-weight Run 7 is always eligible. Tie-break toward less added complexity.
    chosen = min(eligible, key=lambda r: (r["historical_rmse"], r["weight"], r["recipe"]))
    standalone = min(predictions, key=lambda name: (reporting_tools.score(predictions[name]), name))
    selection = {"policy": policy, "blend_selection": chosen, "standalone_selection": standalone,
        "policy_scores": policy_scores, "candidate_scores": candidates, "training_uavs": sorted(job.training_uavs),
        "inner_partitions": [{"train": sorted(a), "held": sorted(b)} for a,b in partitions],
        "selection_uses_outer_labels": False}
    path = engine.directory / "selection.json"
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError("Frozen training-side selection changed")
    atomic_json(path, selection)
    control = engine.fit(job.training_uavs, job.validation_uavs, "run7")
    output = [control.assign(method="run7")]
    raw_best = engine.fit(job.training_uavs, job.validation_uavs, policy=policy, recipe=data_tools.recipes()[standalone])
    output.append(raw_best.assign(method="selected_standalone"))
    if chosen["weight"] == 0:
        blend = control.copy()
    else:
        fitted = raw_best if chosen["recipe"] == standalone else engine.fit(job.training_uavs, job.validation_uavs,
            policy=policy, recipe=data_tools.recipes()[chosen["recipe"]])
        blend = reporting_tools.blended(fitted, control, chosen["weight"])
    output.append(blend.assign(method="selected_blend"))
    return pd.concat(output, ignore_index=True)


def execute_task(root, stage, job, model_seed, source, data, workflow):
    engine = Engine(root, job, model_seed, source, data)
    with threadpool_limits(limits=workflow["cpu_threads"]):
        if stage == "search":
            return nested_search(engine, workflow)
        results = []
        for policy in data_tools.POLICIES:
            results.append(engine.fit(job.training_uavs, job.validation_uavs, policy=policy).assign(method=policy))
        results.append(engine.fit(job.training_uavs, job.validation_uavs, "run7").assign(method="run7"))
        results.append(engine.fit(job.training_uavs, job.validation_uavs, "reference_protocol", "dense_unit_rows").assign(method="reference_protocol"))
        return pd.concat(results, ignore_index=True)


def execute_tasks(tasks, workers):
    if workers == 1:
        return [execute_task(*task) for task in tasks]
    remaining = iter(enumerate(tasks)); active = {}; results = [None]*len(tasks)
    pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        for _ in range(min(workers, len(tasks))):
            i, task = next(remaining); active[pool.submit(execute_task, *task)] = i
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                results[active.pop(future)] = future.result()
            for _ in done:
                i, task = next(remaining, (None, None))
                if i is not None: active[pool.submit(execute_task, *task)] = i
    finally:
        for future in active: future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    return results


def run(workflow, root, stage):
    validate(workflow)
    root = Path(root); reporting = root / "reporting"; reporting.mkdir(parents=True, exist_ok=True)
    provenance = data_tools.provenance(workflow, root)
    if stage == "provenance": return provenance
    source, data, jobs, paths, outer, inner = data_tools.prepare(workflow)
    costs = budget(workflow, jobs)
    if costs["maximum_recipe_evaluations"] > workflow["maximum_recipe_evaluations"]:
        raise ValueError(f"Campaign exceeds declared recipe-evaluation budget: {costs}")
    import campaign_models, campaign_reporting, reproduce_simple_submission
    paths.extend(Path(module.__file__) for module in (data_tools, campaign_models, campaign_reporting, reproduce_simple_submission))
    paths.append(Path(__file__))
    if not (reporting / "pre_registration.json").exists() and any((root / "cells").rglob("*.json")):
        raise ValueError("Cannot reuse campaign cells without their registration")
    if stage != "check" or (reporting / "pre_registration.json").exists():
        register_run(reporting, {**workflow, "source_workflow": source}, paths)
    readiness = {"ready": True, **costs, "policies": data_tools.POLICIES, "recipes": data_tools.recipes(),
        "endpoint_rows": len(data["dev"]["B"]), "parallel_workers": workflow["max_workers"],
        "suite_counts": data["dev"]["B"].metadata.groupby("suite").size().to_dict(), "uses_test_labels": False}
    atomic_json(reporting / "input_verification.json", readiness)
    atomic_csv(reporting / "endpoints.csv", data["dev"]["B"].metadata.assign(observed_rul=data["dev"]["B"].target))
    atomic_csv(reporting / "outer_folds.csv", outer)
    atomic_csv(reporting / "partition_builder_inner_folds.csv", inner)
    if stage == "check": return readiness
    (reporting / "winner_manifest.json").unlink(missing_ok=True)
    results = {}
    for current_stage in (["benchmark", "search"] if stage == "all" else [stage]):
        destination = root / "stages" / current_stage / "reporting/winner_manifest.json"
        destination.unlink(missing_ok=True)
        tasks = [(root, current_stage, job, seed, source, data, workflow) for job in jobs for seed in workflow["model_seeds"]]
        rows = pd.concat(execute_tasks(tasks, workflow["max_workers"]), ignore_index=True)
        results[current_stage] = reporting_tools.report(root, current_stage, rows, workflow)
    audits, selections = [], []
    for path in (root / "cells").rglob("*.json"):
        saved = json.loads(path.read_text())
        if path.name == "selection.json":
            selections.append({"task": path.parent.name, "policy": saved["policy"],
                "standalone_recipe": saved["standalone_selection"], **saved["blend_selection"]})
        else:
            payload = saved["payload"]
            audits.append({"task": path.parent.name, "method": payload["contract"]["method"],
                "policy": payload["contract"]["policy"], "recipe": json.dumps(payload["contract"]["recipe"], sort_keys=True),
                "training_uavs": len(payload["contract"]["training_uavs"]), "held_uavs": len(payload["contract"]["held_uavs"]),
                "base_estimator_fits": payload["audit"].get("base_estimator_fits"), "fit_seconds": payload["audit"].get("fit_seconds")})
    atomic_csv(reporting / "fit_costs.csv", pd.DataFrame(audits))
    atomic_csv(reporting / "selections.csv", pd.DataFrame(selections))
    manifest = {"completed_requested_stages": list(results), "status": "complete", "results": results,
        "full_campaign_completed": all((root / "stages" / s / "reporting/winner_manifest.json").exists() for s in ("benchmark", "search")),
        "automatic_production_replacement": False, "reproduction_score_status": provenance["reproduction"].get("score_status", "pending")}
    atomic_json(reporting / "winner_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_31")
    parser.add_argument("--stage", choices=["provenance", "check", "benchmark", "search", "all"], default="all")
    args = parser.parse_args()
    workflow, _, root = load_workflow(args.config, "campaign_workflows", args.workflow)
    print(json.dumps(run(workflow, root, args.stage), indent=2, sort_keys=True))
