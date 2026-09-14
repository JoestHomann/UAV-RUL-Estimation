"""Matched, resumable add-one-in study of eight unclear channels using the v13 main XGBoost."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import shutil
import time
import tomllib

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def load_source(path):
    spec = importlib.util.spec_from_file_location("pe37_v13", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ablation_columns(columns, removed):
    removed = set(removed)
    return [c for c in columns if c.split("__")[0] not in removed]


def partitions(data, config):
    groups = data.uav_id.to_numpy()
    result = []
    for fold, (outer_train, held) in enumerate(GroupKFold(config["outer_folds"]).split(data, groups=groups)):
        ids = data.iloc[outer_train].uav_id.unique()
        rng = np.random.RandomState(config["stopping_seed"])
        stop_ids = rng.choice(ids, max(1, int(len(ids) * config["stopping_uav_fraction"])), replace=False)
        stopping = outer_train[np.isin(groups[outer_train], stop_ids)]
        fitting = outer_train[~np.isin(groups[outer_train], stop_ids)]
        sets = [set(groups[idx]) for idx in (fitting, stopping, held)]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert len(fitting) + len(stopping) + len(held) == len(data)
        result.append((fold, fitting, stopping, held))
    return result


def prepare(config):
    source = ROOT / config["source"]
    module = load_source(source)
    contract = json.loads((ROOT / config["feature_contract"]).read_text())
    tiers = {tier: contract["sensor_tiers"][tier] for tier in ("strong", "medium", "weak")}
    data = pd.read_csv(ROOT / config["train"]).sort_values(["uav_id", "flight_cycle"]).reset_index(drop=True)
    assert not data.duplicated(["uav_id", "flight_cycle"]).any()
    assert np.isfinite(data.select_dtypes(include="number").to_numpy()).all()
    raw_y = data.RUL.to_numpy().copy()
    data = module.apply_rul_cap(data)
    # The test file contributes only UAV IDs/cutoffs, never telemetry or labels.
    cutoffs = module.get_test_cutoff_lengths(pd.read_csv(ROOT / config["test_cutoffs"], usecols=["uav_id", "flight_cycle"]))
    features = module.build_features_tiered(data, tiers)
    columns = [c for c in features if c != "uav_id"]
    assert columns == contract["feature_columns"] and len(columns) == 153
    assert np.isfinite(features[columns].to_numpy()).all()
    assert np.array_equal(features.uav_id, data.uav_id)
    assert np.array_equal(features.flight_cycle, data.flight_cycle)
    arms = {"baseline": list(config["channels"])} | {
        f"add_{channel.removeprefix('telemetry_')}": [c for c in config["channels"] if c != channel]
        for channel in config["channels"]
    }
    catalogs = {name: ablation_columns(columns, removed) for name, removed in arms.items()}
    assert len(catalogs["baseline"]) == 129
    for name, cols in catalogs.items():
        assert len(cols) == (129 if name == "baseline" else 132)
    # Ensure the full-history computation is identical to a genuinely truncated
    # trajectory, including rolling/expanding summaries (no future observations).
    probe_id = data.uav_id.iloc[0]
    probe = data[(data.uav_id == probe_id) & (data.flight_cycle <= 37)].copy()
    prefix_features = module.build_features_tiered(probe, tiers)
    expected = features.loc[probe.index, columns].to_numpy()
    np.testing.assert_allclose(prefix_features[columns].to_numpy(), expected, rtol=1e-12, atol=1e-12)
    folds = partitions(data, config)
    fold_by_row = np.full(len(data), -1, dtype=int)
    membership = []
    for fold, fitting, stopping, held in folds:
        fold_by_row[held] = fold
        for role, indices in (("fit", fitting), ("early_stop", stopping), ("score", held)):
            membership.extend({"fold": fold, "role": role, "uav_id": uid} for uid in data.iloc[indices].uav_id.unique().tolist())
    endpoints = []
    lifetimes = data.groupby("uav_id").flight_cycle.max().to_dict()
    lookup = {(row.uav_id, row.flight_cycle): idx for idx, row in data[["uav_id", "flight_cycle"]].iterrows()}
    for seed in config["evaluation_seeds"]:
        assigned = module._eligible_random_assignment(lifetimes, cutoffs, np.random.default_rng(seed))
        for uid, cutoff in assigned:
            idx = lookup[(uid, cutoff)]
            endpoints.append(dict(evaluation_seed=seed, uav_id=uid, flight_cycle=cutoff, row_index=idx,
                                  fold=int(fold_by_row[idx]), target_capped=float(data.RUL.iloc[idx]), target_raw=float(raw_y[idx])))
    endpoints = pd.DataFrame(endpoints)
    assert (endpoints.groupby(["evaluation_seed", "uav_id"]).size() == 1).all()
    assert endpoints.uav_id.nunique() == data.uav_id.nunique()
    assert (endpoints.groupby("uav_id").size() == len(config["evaluation_seeds"])).all()
    return module, data, features, tiers, arms, catalogs, folds, pd.DataFrame(membership), endpoints, cutoffs


def metrics(frame, target="target_capped"):
    actual, predicted = frame[target], frame.prediction
    return {"rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "mae": float(mean_absolute_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted)),
            "bias": float(np.mean(predicted - actual))}


def require_equal(actual, expected, label):
    if actual != expected:
        raise RuntimeError(f"Baseline reuse mismatch: {label}")


def verify_baseline(config, registration, membership, endpoints, module, data, folds, cutoffs, catalogs):
    """Read-only verification; --check never imports cells or trains models."""
    source = ROOT / config["baseline_run"]
    old = json.loads((source / "registration.json").read_text())
    require_equal(sha(ROOT / config["baseline_runner"]), old["hashes"]["runner"], "PE_35 runner")
    for key in ("source", "feature_contract", "train", "test_cutoffs"):
        require_equal(registration["hashes"][key], old["hashes"][key], key)
    for key in ("xgboost_params", "early_stopping_rounds", "target_cap", "rolling_windows", "sensor_tiers", "versions", "protocol"):
        require_equal(registration[key], old[key], key)
    for key in ("channels", "outer_folds", "stopping_uav_fraction", "stopping_seed", "evaluation_seeds"):
        require_equal(config[key], old["config"][key], key)
    require_equal(catalogs["baseline"], old["feature_columns"]["drop_all_eight"], "baseline feature order")
    require_equal(sha(source / "v13_source_snapshot.py"), registration["hashes"]["source"], "source snapshot")
    for name, expected in (("fold_membership.csv", membership), ("evaluation_endpoints.csv", endpoints)):
        pd.testing.assert_frame_equal(pd.read_csv(source / name), expected, check_dtype=False, atol=1e-12, rtol=1e-12)
    sensors = [sensor for tier in registration["sensor_tiers"].values() for sensor in tier]
    cell_hashes = {}
    for fold, fitting, stopping, held in folds:
        stem = source / "cells" / f"drop_all_eight__fold_{fold}"
        csv_path, json_path = stem.with_suffix(".csv"), stem.with_suffix(".json")
        metadata = json.loads(json_path.read_text())
        require_equal(metadata["arm"], "drop_all_eight", "source arm")
        require_equal(metadata["fold"], fold, "source fold")
        require_equal(metadata["features"], 129, "source feature count")
        require_equal(sha(csv_path), metadata["prediction_sha256"], "prediction checksum")
        saved = pd.read_csv(csv_path)
        expected = endpoints[endpoints.fold == fold].reset_index(drop=True)
        pd.testing.assert_frame_equal(saved[expected.columns].reset_index(drop=True), expected, check_dtype=False, atol=1e-12, rtol=1e-12)
        if not np.isfinite(saved.prediction).all():
            raise RuntimeError("Nonfinite baseline predictions")
        with redirect_stdout(io.StringIO()):
            weights = module.compute_feature_weights(catalogs["baseline"], data.iloc[fitting], sensors)
        require_equal(set(metadata["feature_weights"]), set(catalogs["baseline"]), "weight columns")
        np.testing.assert_allclose([metadata["feature_weights"][c] for c in catalogs["baseline"]], weights, atol=1e-12, rtol=1e-12)
        samples = module.compute_test_similarity_weights(data.iloc[fitting].flight_cycle, cutoffs)
        np.testing.assert_allclose([metadata["sample_weight_min"], metadata["sample_weight_max"]], [samples.min(), samples.max()], atol=1e-12, rtol=1e-12)
        cell_hashes[str(fold)] = {"csv": sha(csv_path), "metadata": sha(json_path)}
    return {"source_run": config["baseline_run"], "source_arm": "drop_all_eight",
            "registration_sha256": sha(source / "registration.json"), "cells": cell_hashes,
            "reused_fits": len(folds), "new_fits": len(config["channels"]) * len(folds)}


def import_baseline(output, config, reuse):
    """Import only on an execution command, after all reuse checks pass."""
    source = ROOT / config["baseline_run"]
    for fold, fingerprints in reuse["cells"].items():
        src = source / "cells" / f"drop_all_eight__fold_{fold}.csv"
        dst = output / "cells" / f"baseline__fold_{fold}.csv"
        if dst.with_suffix(".json").exists():
            require_equal(sha(dst), fingerprints["csv"], "imported baseline predictions")
            continue
        require_equal(sha(src), fingerprints["csv"], "source predictions changed during import")
        require_equal(sha(src.with_suffix(".json")), fingerprints["metadata"], "source metadata changed during import")
        dst.parent.mkdir(parents=True, exist_ok=True)
        temporary = dst.with_suffix(".csv.tmp")
        temporary.write_bytes(src.read_bytes())
        temporary.replace(dst)
        metadata = json.loads(src.with_suffix(".json").read_text())
        metadata.update(arm="baseline", reused_from=str(src.relative_to(ROOT)),
                        source_metadata_sha256=fingerprints["metadata"])
        write_json(dst.with_suffix(".json"), metadata)


def report(output, config, catalogs):
    predictions = []
    for arm in catalogs:
        for fold in range(config["outer_folds"]):
            cell = output / "cells" / f"{arm}__fold_{fold}.csv"
            frame = pd.read_csv(cell)
            frame["arm"] = arm
            predictions.append(frame)
    predictions = pd.concat(predictions, ignore_index=True)
    write_csv(output / "reporting/predictions.csv", predictions)
    rows = []
    for arm, frame in predictions.groupby("arm", sort=False):
        row = {"arm": arm, "features": len(catalogs[arm]), **metrics(frame)}
        row.update({f"raw_{k}": v for k, v in metrics(frame, "target_raw").items()})
        per_seed = [metrics(part)["r2"] for _, part in frame.groupby("evaluation_seed")]
        row["mean_test_like_seed_r2"] = float(np.mean(per_seed))
        rows.append(row)
    summary = pd.DataFrame(rows)
    write_csv(output / "reporting/summary.csv", summary)
    fold_metrics = pd.DataFrame([{"arm": arm, "fold": fold, **metrics(part)}
                                for (arm, fold), part in predictions.groupby(["arm", "fold"])])
    write_csv(output / "reporting/fold_metrics.csv", fold_metrics)
    seed_metrics = pd.DataFrame([{"arm": arm, "evaluation_seed": seed, **metrics(part)}
                                for (arm, seed), part in predictions.groupby(["arm", "evaluation_seed"])])
    write_csv(output / "reporting/seed_metrics.csv", seed_metrics)
    baseline = predictions[predictions.arm == "baseline"].set_index(["evaluation_seed", "uav_id", "flight_cycle"]).sort_index()
    uavs = sorted(baseline.index.get_level_values("uav_id").unique())
    rng = np.random.default_rng(config["bootstrap_seed"])
    draws = rng.integers(0, len(uavs), size=(config["bootstrap_repetitions"], len(uavs)))
    base_sq = (baseline.prediction - baseline.target_capped) ** 2
    base_uav_mse = base_sq.groupby(level="uav_id").mean().reindex(uavs).to_numpy()
    base_boot = np.sqrt(base_uav_mse[draws].mean(axis=1))
    paired = []
    base_rmse = float(summary.set_index("arm").loc["baseline", "rmse"])
    base_fold = fold_metrics[fold_metrics.arm == "baseline"].set_index("fold").rmse
    for arm in catalogs:
        if arm == "baseline":
            continue
        treatment = predictions[predictions.arm == arm].set_index(baseline.index.names).sort_index()
        assert treatment.index.equals(baseline.index)
        np.testing.assert_array_equal(treatment.target_capped, baseline.target_capped)
        squared = (treatment.prediction - treatment.target_capped) ** 2
        uav_mse = squared.groupby(level="uav_id").mean().reindex(uavs).to_numpy()
        changes = np.sqrt(uav_mse[draws].mean(axis=1)) - base_boot
        lo, hi = np.quantile(changes, [0.025, 0.975])
        # Bonferroni intervals for the eight predeclared baseline comparisons.
        adjusted_lo, adjusted_hi = np.quantile(changes, [0.025 / (len(catalogs) - 1), 1 - 0.025 / (len(catalogs) - 1)])
        delta = float(summary.set_index("arm").loc[arm, "rmse"] - base_rmse)
        candidate_fold = fold_metrics[fold_metrics.arm == arm].set_index("fold").rmse
        paired.append(dict(arm=arm, rmse_change=delta, rmse_change_percent=100 * delta / base_rmse,
                           improved_folds=int((candidate_fold < base_fold).sum()), total_folds=len(base_fold),
                           uav_bootstrap_ci95_low=float(lo), uav_bootstrap_ci95_high=float(hi),
                           familywise_ci95_low=float(adjusted_lo), familywise_ci95_high=float(adjusted_hi),
                           interpretation="addition_candidate" if adjusted_hi < 0 else "addition_harmful" if adjusted_lo > 0 else "inconclusive"))
    paired = pd.DataFrame(paired).sort_values("rmse_change")
    write_csv(output / "reporting/paired_additions.csv", paired)
    regions = []
    for arm, frame in predictions.groupby("arm"):
        for region, mask in (("short_history_le105", frame.flight_cycle <= 105),
                             ("longer_history_gt105", frame.flight_cycle > 105),
                             ("late_life_raw_rul_le30", frame.target_raw <= 30)):
            subset = frame[mask]
            if len(subset) > 1:
                regions.append({"arm": arm, "region": region, "rows": len(subset), "uavs": subset.uav_id.nunique(), **metrics(subset)})
    write_csv(output / "reporting/region_metrics.csv", pd.DataFrame(regions))
    text = ["# PE_37: v13 XGBoost unclear-channel add-one-in study", "", "Completed 40 new fits and reused five verified PE_35 baseline fits: degradation/context baseline plus eight individual channel additions.", "",
            "Primary target: RUL capped at 125, matching v13. Negative RMSE change favors adding the channel. Each UAV has ten test-like cutoffs and equal total evaluation weight.", "",
            "| Arm | Features | RMSE | R2 | Mean cutoff-seed R2 |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples():
        text.append(f"| {row.arm} | {row.features} | {row.rmse:.4f} | {row.r2:.5f} | {row.mean_test_like_seed_r2:.5f} |")
    text += ["", "| Addition | RMSE change | Fold wins | Paired UAV 95% interval | Eight-comparison adjusted interval | Assessment |",
             "| --- | ---: | ---: | --- | --- | --- |"]
    for row in paired.itertuples():
        text.append(f"| {row.arm} | {row.rmse_change:+.4f} | {row.improved_folds}/5 | [{row.uav_bootstrap_ci95_low:+.4f}, {row.uav_bootstrap_ci95_high:+.4f}] | [{row.familywise_ci95_low:+.4f}, {row.familywise_ci95_high:+.4f}] | {row.interpretation} |")
    text += ["", "These are development ablations on previously examined UAVs with historically selected, frozen feature tiers. Bootstrap intervals resample UAVs and condition on the fitted models; they do not measure variation from retraining on different splits. Ten cutoff seeds are repeated endpoints, not ten independent training runs.", "",
             "One main XGBoost per arm/fold, no early specialist or gating. Outer scoring UAVs are excluded from feature-weight calculation, fitting, and early stopping. A fixed 10% of outer-training UAVs supplies unweighted early stopping; the model is not refit, matching v13's final-fit holdout approach. Feature weights use only actual fitting UAVs. Tree settings and all-cycle sample-weight mechanics come directly from v13.", "",
             "An addition candidate still needs confirmation. Do not combine individual additions without evaluating that combination. No production model or submission was changed.", ""]
    (output / "reporting/report.md").write_text("\n".join(text), encoding="utf-8")
    write_json(output / "reporting/completion.json", {"status": "complete", "fits": len(predictions.arm.unique()) * config["outer_folds"], "new_fits": 40, "reused_fits": 5, "configurations": len(catalogs), "primary_target": "cap125", "bootstrap_unit": "uav_id", "production_changed": False})
    print(summary[["arm", "features", "rmse", "r2"]].to_string(index=False), flush=True)
    print(paired[["arm", "rmse_change", "improved_folds", "interpretation"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate features, partitions and endpoints without fitting")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config = tomllib.loads((HERE / "settings.toml").read_text())["study"]
    output = HERE / "runs" / config["run"]
    module, data, features, tiers, arms, catalogs, folds, membership, endpoints, cutoffs = prepare(config)
    hashes = {name: sha(ROOT / config[name]) for name in ("source", "feature_contract", "train", "test_cutoffs")}
    hashes.update(runner=sha(__file__), settings=sha(HERE / "settings.toml"))
    registration = {"config": config, "hashes": hashes, "xgboost_params": module.XGB_PARAMS,
                    "early_stopping_rounds": module.XGB_EARLY_STOPPING_ROUNDS, "target_cap": module.RUL_CAP,
                    "rolling_windows": list(module.ROLLING_WINDOWS), "sensor_tiers": tiers,
                    "feature_columns": catalogs, "versions": {"python": platform.python_version(), "xgboost": xgb.__version__, "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
                    "protocol": "main_xgboost_only; frozen_historical_tiers; inner_10pct_uav_early_stop; no_refit; fit_only_feature_weights; v13_GroupKFold; ten_test_cutoff_seeds"}
    reuse = verify_baseline(config, registration, membership, endpoints, module, data, folds, cutoffs, catalogs)
    registration["baseline_reuse"] = reuse
    manifest = output / "registration.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != registration:
            raise RuntimeError("Inputs, code or settings changed. Choose a new study.run before running.")
    else:
        write_json(manifest, registration)
        shutil.copyfile(ROOT / config["source"], output / "v13_source_snapshot.py")
    write_csv(output / "fold_membership.csv", membership)
    write_csv(output / "evaluation_endpoints.csv", endpoints)
    write_json(output / "feature_catalog.json", catalogs)
    write_json(output / "input_verification.json", {"status": "passed", "feature_counts": {k: len(v) for k, v in catalogs.items()}, "uavs": int(data.uav_id.nunique()), "training_rows": len(data), "evaluation_rows": len(endpoints), "causal_prefix_check": True, "disjoint_fit_stopping_score_uavs": True})
    print(f"Validated {len(catalogs)} arms, {len(folds)} folds, {len(endpoints)} matched endpoints", flush=True)
    if args.check:
        return
    import_baseline(output, config, reuse)
    if args.report_only:
        report(output, config, catalogs)
        return
    all_columns = [c for c in features if c != "uav_id"]
    matrix = features[all_columns].to_numpy()
    y = data.RUL.to_numpy()
    sensors = [sensor for tier in tiers.values() for sensor in tier]
    full_columns = all_columns
    completed = 0
    for fold, fitting, stopping, held in folds:
        with redirect_stdout(io.StringIO()):
            feature_weights = module.compute_feature_weights(full_columns, data.iloc[fitting], sensors)
        sample_weights = module.compute_test_similarity_weights(data.iloc[fitting].flight_cycle, cutoffs)
        selected_endpoints = endpoints[endpoints.fold == fold].copy()
        assert set(selected_endpoints.uav_id) <= set(data.iloc[held].uav_id)
        for arm, columns in catalogs.items():
            cell_csv = output / "cells" / f"{arm}__fold_{fold}.csv"
            cell_json = cell_csv.with_suffix(".json")
            if cell_json.exists():
                previous = json.loads(cell_json.read_text())
                if previous["prediction_sha256"] != sha(cell_csv):
                    raise RuntimeError(f"Checkpoint corrupted: {cell_csv}")
                completed += 1
                print(f"Reused {completed}/45 {arm} fold {fold}", flush=True)
                continue
            indices = [full_columns.index(column) for column in columns]
            print(f"Fitting {completed + 1}/45 {arm} fold {fold}: {len(columns)} features", flush=True)
            started = time.perf_counter()
            model = xgb.XGBRegressor(**module.XGB_PARAMS, early_stopping_rounds=module.XGB_EARLY_STOPPING_ROUNDS, feature_weights=feature_weights[indices])
            model.fit(matrix[np.ix_(fitting, indices)], y[fitting], sample_weight=sample_weights,
                      eval_set=[(matrix[np.ix_(stopping, indices)], y[stopping])], verbose=False)
            selected_endpoints["prediction"] = model.predict(matrix[np.ix_(selected_endpoints.row_index.to_numpy(), indices)])
            assert np.isfinite(selected_endpoints.prediction).all()
            write_csv(cell_csv, selected_endpoints)
            elapsed = time.perf_counter() - started
            write_json(cell_json, {"arm": arm, "fold": fold, "features": len(columns), "best_iteration": int(model.best_iteration),
                                  "fit_seconds": elapsed, "prediction_sha256": sha(cell_csv), "metrics": metrics(selected_endpoints),
                                  "feature_weights": dict(zip(columns, feature_weights[indices].tolist())),
                                  "sample_weight_min": float(sample_weights.min()), "sample_weight_max": float(sample_weights.max())})
            completed += 1
            print(f"Completed {completed}/45 in {elapsed:.1f}s; iteration={model.best_iteration}; RMSE={metrics(selected_endpoints)['rmse']:.4f}", flush=True)
    report(output, config, catalogs)


if __name__ == "__main__":
    main()
