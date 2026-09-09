"""Frozen PE_31 policy matrix, endpoint suites and traceable inputs."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_weight_scale_audit as prior
import run_complete_pipeline_validation as audit_tools
from confirmation_utils import generated_partitions, evaluation_jobs, validate_partitions
from followup_experiment_utils import input_path, atomic_json
from tabular_data_adapter import TabularDataset

POLICIES = {
    "sparse_uav_mass": ("sparse", "uav", "uav_count"),
    "sparse_unit_mean": ("sparse", "uav", "row_count"),
    "dense_uav_mass": ("dense", "uav", "uav_count"),
    "dense_uav_unit_mean": ("dense", "uav", "row_count"),
    "dense_cycle_mass": ("dense", "cycle", "uav_count"),
    "dense_unit_rows": ("dense", "cycle", "row_count"),
}


def weight_data(data, policy):
    """Normalize inside EACH actual fit, never using outer-held UAV statistics."""
    _, influence, mass = POLICIES[policy]
    counts = data.metadata.groupby("uav_id").uav_id.transform("size").to_numpy(float)
    relative = 1. / counts if influence == "uav" else np.ones(len(data))
    total = data.metadata.uav_id.nunique() if mass == "uav_count" else len(data)
    return replace(data, sample_weights=pd.Series(relative * (total / relative.sum())))


def recipes():
    return {f"{view}_{family}_{capacity}": {"view": view, "family": family, "capacity": capacity}
        for view in ("B", "E") for family in ("blend", "catboost", "extra_trees") for capacity in ("original", "regularized")}


def base_recipe():
    return {"view": "B", "family": "blend", "capacity": "original"}


def seeded_source(source, model_seed, recipe):
    result = deepcopy(source)
    result["simple"]["xgboost"]["random_state"] = int(model_seed)
    result["simple"]["catboost"]["random_seed"] = int(model_seed)
    if recipe["capacity"] == "regularized":
        result["simple"]["xgboost"].update(max_depth=3, reg_lambda=5., min_child_weight=2.)
        result["simple"]["catboost"].update(depth=4, l2_leaf_reg=10.)
    return result


def endpoint_suites(raw, historical, cutoffs, seeds):
    """Independent per-UAV assignments: another UAV's labels cannot change these draws."""
    hist = historical.metadata[["uav_id", "cutoff", "scenario"]].copy()
    hist["suite"] = "historical"; hist["endpoint_seed"] = -1
    rows = hist.to_dict("records")
    for uid, group in raw.groupby("uav_id", sort=True):
        for suite in ("nominal", "unrestricted"):
            valid = group.loc[group.RUL.ge(1) & (group.RUL.le(125) if suite == "nominal" else True), "flight_cycle"].to_numpy(int)
            possible = cutoffs[np.isin(cutoffs, valid)]
            if len(possible) == 0:
                raise ValueError(f"No feasible empirical cutoff for {uid}, {suite}")
            salt = int(hashlib.sha256(str(uid).encode()).hexdigest()[:8], 16)
            for seed in seeds:
                rng = np.random.default_rng(np.random.SeedSequence([int(seed), salt, int(suite == "nominal")]))
                cutoff = int(rng.choice(possible))
                rows.append({"uav_id": str(uid), "cutoff": cutoff, "scenario": f"{suite}_{seed}",
                             "suite": suite, "endpoint_seed": int(seed)})
    result = pd.DataFrame(rows)
    result["sample_id"] = result.scenario + "::" + result.uav_id + "::" + result.cutoff.astype(str)
    labels = raw.rename(columns={"flight_cycle": "cutoff"})[["uav_id", "cutoff", "RUL"]]
    result = result.merge(labels, on=["uav_id", "cutoff"], validate="many_to_one")
    return result


def prepare(workflow):
    source_root, source, paths = prior.verified_source({"source_registration": workflow["source_registration"],
        "audit_split_seed": 20270107, "confirmation_split_seeds": workflow["split_seeds"]})
    adapter = prior.pe28.TabularDataAdapter(input_path(source["tabular_manifest"]))
    train = adapter.load_training(source["feature_set"])
    historical = adapter.load_development(source["feature_set"])
    raw = pd.read_csv(input_path(source["raw_training"]))
    cutoffs_path = input_path(workflow["test_cutoffs"])
    cutoffs = pd.read_csv(cutoffs_path).final_cycle.to_numpy(int)
    endpoints = endpoint_suites(raw, historical, cutoffs, workflow["endpoint_seeds"])
    print("PE_31: building all-cycle and multi-profile causal features", flush=True)
    dense = prior.dense_view(raw)
    histories = prior.validate_raw(raw)
    sparse = prior.sparse_view(train, histories)
    b = prior.sparse_view(TabularDataset(pd.DataFrame(index=range(len(endpoints))),
        endpoints.drop(columns="RUL"), endpoints.RUL.astype(float).reset_index(drop=True), None), histories)
    e_names = list(prior.prefix_features(next(iter(histories.values())), 1, "E_compact_22"))
    # Build Run 7 features only at evaluation endpoints; its training matrix is unchanged.
    metadata = endpoints.copy()
    metadata["outer_fold"] = 0; metadata["lifetime_quantile"] = 0
    metadata["terminal_lifetime"] = metadata.uav_id.map(raw.groupby("uav_id").flight_cycle.max())
    current_features = audit_tools.build_feature_table(raw, metadata, feature_profile="extended")
    current_dev = replace(b, features=current_features[list(train.features)].reset_index(drop=True))
    outer, inner = generated_partitions(history_summary_path=source["history_summary"], split_seeds=workflow["split_seeds"],
        outer_fold_count=5, inner_fold_count=workflow["selection_folds"])
    validate_partitions(outer, inner, expected_outer_folds=5, expected_inner_folds=workflow["selection_folds"])
    data = {"train": {"sparse_B": sparse, "dense_B": dense, "run7": train,
        "sparse_E": replace(sparse, features=sparse.features[e_names]), "dense_E": replace(dense, features=dense.features[e_names])},
        "dev": {"B": b, "E": replace(b, features=b.features[e_names]), "run7": current_dev},
        "raw": raw, "cutoffs": cutoffs}
    paths.extend([cutoffs_path, Path(prior.__file__), Path(audit_tools.__file__), Path(audit_tools.build_feature_table.__code__.co_filename)])
    return source, data, evaluation_jobs(outer, inner, include_inner=False), paths, outer, inner


def provenance(workflow, root):
    from reproduce_simple_submission import sha, ROOT
    directory = Path(root) / "reporting"; directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name, path_value, score in (("run7", workflow["run7_submission"], .87652),
                                    ("alternative_historical", workflow.get("alternative_submission", ""), .87877)):
        path = ROOT / path_value if path_value else None
        exists = path is not None and path.is_file()
        records.append({"reference": name, "path": str(path) if exists else None,
            "sha256": sha(path) if exists else None, "user_reported_score": score,
            "score_to_file_link": "user_reported_not_independently_verified" if exists else "missing_original_file"})
    replay = Path(root) / "reproduction/reproduction_manifest.json"
    result = {"references": records, "reproduction": json.loads(replay.read_text()) if replay.exists() else {"status": "pending"},
        "outer_reference_is_protocol_reproduction_not_historical_score_verification": True,
        "historical_cells_reused_for_new_scores": False,
        "reuse_reason": "New endpoint suites and fit-local weight normalization require matched refits; old results remain historical anchors."}
    atomic_json(directory / "provenance.json", result)
    return result
