"""Matched, resumable test of removing telemetry 05 and 24 together from the v13 main XGBoost."""
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
    spec = importlib.util.spec_from_file_location("pe39_v13", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ablation_columns(columns, removed):
    removed = set(removed)
    return [c for c in columns if c.split("__")[0] not in removed]


def arm_name(channels):
    return "drop_" + "_".join(channel.removeprefix("telemetry_") for channel in channels)


def build_arms(channels):
    """Baseline, each channel alone, then the pair. Order is the reporting order."""
    arms = {"baseline": []}
    for channel in channels:
        arms[arm_name([channel])] = [channel]
    arms[arm_name(channels)] = list(channels)
    return arms


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
    for channel in config["channels"]:
        assert any(c == channel or c.startswith(channel + "__") for c in columns), channel
    arms = build_arms(config["channels"])
    catalogs = {name: ablation_columns(columns, removed) for name, removed in arms.items()}
    assert catalogs["baseline"] == columns
    pair = arm_name(config["channels"])
    singles = [arm_name([channel]) for channel in config["channels"]]
    # Each channel contributes the same columns alone as it does in the pair, so the
    # pair removes exactly the union and nothing else.
    removed_by_arm = {arm: [c for c in columns if c not in set(cols)] for arm, cols in catalogs.items()}
    assert set(removed_by_arm[pair]) == set().union(*(set(removed_by_arm[s]) for s in singles))
    assert not set(removed_by_arm[singles[0]]) & set(removed_by_arm[singles[1]])
    assert len(catalogs[pair]) == 153 - sum(len(removed_by_arm[s]) for s in singles)
    # Ensure the full-history computation is identical to a genuinely truncated
    # trajectory, including rolling/expanding summaries (no future observations).
    probe_id = data.uav_id.iloc[0]
    probe = data[(data.uav_id == probe_id) & (data.flight_cycle <= 37)].copy()
    prefix_features = module.build_features_tiered(probe, tiers)
    np.testing.assert_allclose(prefix_features[columns].to_numpy(), features.loc[probe.index, columns].to_numpy(),
                               rtol=1e-12, atol=1e-12)
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


def indexed_arm(predictions, arm):
    frame = predictions[predictions.arm == arm].set_index(["evaluation_seed", "uav_id", "flight_cycle"]).sort_index()
    assert not frame.empty, arm
    return frame


def bootstrap_rmse(predictions, arms, uavs, draws):
    """Paired UAV bootstrap RMSE per arm, drawn on one shared set of resamples."""
    reference = None
    result = {}
    for arm in arms:
        frame = indexed_arm(predictions, arm)
        if reference is None:
            reference = frame
        else:
            assert frame.index.equals(reference.index)
            np.testing.assert_array_equal(frame.target_capped, reference.target_capped)
        squared = (frame.prediction - frame.target_capped) ** 2
        uav_mse = squared.groupby(level="uav_id").mean().reindex(uavs).to_numpy()
        result[arm] = np.sqrt(uav_mse[draws].mean(axis=1))
    return result


def paired_change(boot, summary, fold_metrics, treatment, reference, family):
    changes = boot[treatment] - boot[reference]
    lo, hi = np.quantile(changes, [0.025, 0.975])
    adjusted_lo, adjusted_hi = np.quantile(changes, [0.025 / family, 1 - 0.025 / family])
    rmse = summary.set_index("arm").rmse
    delta = float(rmse[treatment] - rmse[reference])
    folds = fold_metrics.set_index(["arm", "fold"]).rmse
    reference_folds = folds[reference]
    return dict(arm=treatment, reference=reference, rmse_change=delta,
                rmse_change_percent=100 * delta / float(rmse[reference]),
                improved_folds=int((folds[treatment] < reference_folds).sum()), total_folds=len(reference_folds),
                uav_bootstrap_ci95_low=float(lo), uav_bootstrap_ci95_high=float(hi),
                familywise_ci95_low=float(adjusted_lo), familywise_ci95_high=float(adjusted_hi),
                comparisons_adjusted=family,
                interpretation="removal_candidate" if adjusted_hi < 0 else "removal_harmful" if adjusted_lo > 0 else "inconclusive")


def additivity(boot, summary, pair, singles):
    """Interaction term: what the pair costs beyond the sum of the single removals."""
    rmse = summary.set_index("arm").rmse
    combined = float(rmse[pair] - rmse["baseline"])
    parts = [float(rmse[single] - rmse["baseline"]) for single in singles]
    samples = boot[pair] - boot["baseline"] - sum(boot[single] - boot["baseline"] for single in singles)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"pair": pair, "pair_change": combined, "sum_of_single_changes": float(sum(parts)),
            "interaction": combined - float(sum(parts)),
            "interaction_ci95_low": float(lo), "interaction_ci95_high": float(hi),
            "interpretation": "inconclusive" if lo <= 0 <= hi else
                              "worse_than_additive" if lo > 0 else "better_than_additive"}


def reproduction_note(output, config, catalogs, registration):
    """Bitwise comparison against PE_35's matched fits, when that run is present."""
    reference = config.get("reproduction_check") or ""
    if not reference:
        return {"status": "skipped", "reason": "no reference run configured"}
    source = ROOT / reference
    if not (source / "registration.json").exists():
        return {"status": "skipped", "reason": f"{reference} not found"}
    old = json.loads((source / "registration.json").read_text())
    shared = {arm: arm for arm in catalogs if (source / "cells" / f"{arm}__fold_0.csv").exists()}
    inputs_match = all(registration["hashes"][key] == old["hashes"][key]
                       for key in ("source", "feature_contract", "train", "test_cutoffs"))
    versions_match = registration["versions"] == old["versions"]
    settings_match = all(registration[key] == old[key] for key in ("xgboost_params", "early_stopping_rounds", "target_cap"))
    compared, mismatched, worst = [], [], 0.0
    for arm in shared:
        for fold in range(config["outer_folds"]):
            new_path = output / "cells" / f"{arm}__fold_{fold}.csv"
            old_path = source / "cells" / f"{arm}__fold_{fold}.csv"
            if not new_path.exists():
                continue
            new, previous = pd.read_csv(new_path), pd.read_csv(old_path)
            keys = ["evaluation_seed", "uav_id", "flight_cycle"]
            merged = new.merge(previous[keys + ["prediction"]], on=keys, suffixes=("", "_reference"))
            assert len(merged) == len(new)
            difference = float(np.abs(merged.prediction - merged.prediction_reference).max())
            worst = max(worst, difference)
            compared.append(f"{arm}__fold_{fold}")
            if difference > 0:
                mismatched.append({"cell": f"{arm}__fold_{fold}", "max_abs_difference": difference})
    note = {"status": "checked" if compared else "skipped", "reference_run": reference,
            "arms": sorted(shared), "cells_compared": len(compared),
            "inputs_match": inputs_match, "settings_match": settings_match, "versions_match": versions_match,
            "max_abs_prediction_difference": worst, "mismatched_cells": mismatched,
            "identical": not mismatched}
    if mismatched and inputs_match and settings_match and versions_match:
        raise RuntimeError(
            "PE_35's matched fits were not reproduced bitwise although inputs, settings and library "
            f"versions are identical (worst difference {worst:.6g}). Investigate before reporting: "
            "the protocol is supposed to be deterministic.")
    return note


def report(output, config, catalogs, registration=None):
    predictions = []
    for arm in catalogs:
        for fold in range(config["outer_folds"]):
            frame = pd.read_csv(output / "cells" / f"{arm}__fold_{fold}.csv")
            frame["arm"] = arm
            predictions.append(frame)
    predictions = pd.concat(predictions, ignore_index=True)
    write_csv(output / "reporting/predictions.csv", predictions)
    rows = []
    for arm, frame in predictions.groupby("arm", sort=False):
        row = {"arm": arm, "features": len(catalogs[arm]), **metrics(frame)}
        row.update({f"raw_{k}": v for k, v in metrics(frame, "target_raw").items()})
        row["mean_test_like_seed_r2"] = float(np.mean([metrics(part)["r2"] for _, part in frame.groupby("evaluation_seed")]))
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("arm").loc[list(catalogs)].reset_index()
    write_csv(output / "reporting/summary.csv", summary)
    fold_metrics = pd.DataFrame([{"arm": arm, "fold": fold, **metrics(part)}
                                 for (arm, fold), part in predictions.groupby(["arm", "fold"])])
    write_csv(output / "reporting/fold_metrics.csv", fold_metrics)
    seed_metrics = pd.DataFrame([{"arm": arm, "evaluation_seed": seed, **metrics(part)}
                                 for (arm, seed), part in predictions.groupby(["arm", "evaluation_seed"])])
    write_csv(output / "reporting/seed_metrics.csv", seed_metrics)

    uavs = sorted(predictions.uav_id.unique())
    rng = np.random.default_rng(config["bootstrap_seed"])
    draws = rng.integers(0, len(uavs), size=(config["bootstrap_repetitions"], len(uavs)))
    boot = bootstrap_rmse(predictions, list(catalogs), uavs, draws)
    removals = [arm for arm in catalogs if arm != "baseline"]
    paired = pd.DataFrame([paired_change(boot, summary, fold_metrics, arm, "baseline", len(removals)) for arm in removals])
    write_csv(output / "reporting/paired_ablation.csv", paired)
    pair = arm_name(config["channels"])
    singles = [arm_name([channel]) for channel in config["channels"]]
    interaction = additivity(boot, summary, pair, singles)
    write_json(output / "reporting/additivity.json", interaction)
    contrasts = [tuple(item) for item in config["secondary_contrasts"]]
    secondary = pd.DataFrame([paired_change(boot, summary, fold_metrics, treatment, reference, len(contrasts))
                              for treatment, reference in contrasts])
    write_csv(output / "reporting/secondary_contrasts.csv", secondary)

    regions = []
    for arm, frame in predictions.groupby("arm"):
        for region, mask in (("short_history_le105", frame.flight_cycle <= 105),
                             ("longer_history_gt105", frame.flight_cycle > 105),
                             ("late_life_raw_rul_le30", frame.target_raw <= 30)):
            subset = frame[mask]
            if len(subset) > 1:
                regions.append({"arm": arm, "region": region, "rows": len(subset), "uavs": subset.uav_id.nunique(), **metrics(subset)})
    write_csv(output / "reporting/region_metrics.csv", pd.DataFrame(regions))
    note = reproduction_note(output, config, catalogs, registration) if registration else {"status": "skipped", "reason": "report-only"}
    write_json(output / "reporting/reproduction_check.json", note)

    labels = ", ".join(config["channels"])
    text = [f"# PE_39: v13 XGBoost with {labels} removed together", "",
            f"Completed {len(catalogs) * config['outer_folds']} matched fits: the unchanged v13 representation, "
            "each channel removed alone, and both removed together.", "",
            "Primary target: RUL capped at 125, matching v13. Negative RMSE change favors removal. "
            "Each UAV has ten test-like cutoffs and equal total evaluation weight.", "",
            "| Arm | Features | RMSE | R2 | Mean cutoff-seed R2 |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples():
        text.append(f"| {row.arm} | {row.features} | {row.rmse:.4f} | {row.r2:.5f} | {row.mean_test_like_seed_r2:.5f} |")
    text += ["", f"| Removal | RMSE change | Fold wins | Paired UAV 95% interval | "
             f"{len(removals)}-comparison adjusted interval | Assessment |",
             "| --- | ---: | ---: | --- | --- | --- |"]
    for row in paired.sort_values("rmse_change").itertuples():
        text.append(f"| {row.arm} | {row.rmse_change:+.4f} | {row.improved_folds}/{row.total_folds} | "
                    f"[{row.uav_bootstrap_ci95_low:+.4f}, {row.uav_bootstrap_ci95_high:+.4f}] | "
                    f"[{row.familywise_ci95_low:+.4f}, {row.familywise_ci95_high:+.4f}] | {row.interpretation} |")
    text += ["", "## Is the pair more than the sum of its parts?", "",
             "PE_35 screened these channels one at a time and warned that individual removals must not be "
             "combined without evaluating the combination. This is that evaluation. The interaction term is "
             "the pair's change minus the two single changes; a positive value means removing both costs more "
             "than the singles predict.", "",
             f"- Pair change: {interaction['pair_change']:+.4f} cycles",
             f"- Sum of single changes: {interaction['sum_of_single_changes']:+.4f} cycles",
             f"- Interaction: {interaction['interaction']:+.4f} cycles, 95% interval "
             f"[{interaction['interaction_ci95_low']:+.4f}, {interaction['interaction_ci95_high']:+.4f}] "
             f"({interaction['interpretation']})", "",
             "Pre-declared secondary contrasts, adjusted over their own family: the marginal cost of removing "
             "the second channel once the first is already gone. They describe where any effect sits; only the "
             "comparisons against the baseline are decisive.", "",
             "| Contrast | RMSE change | Fold wins | Paired UAV 95% interval | Adjusted interval | Assessment |",
             "| --- | ---: | ---: | --- | --- | --- |"]
    for row in secondary.itertuples():
        text.append(f"| {row.arm} vs. {row.reference} | {row.rmse_change:+.4f} | {row.improved_folds}/{row.total_folds} | "
                    f"[{row.uav_bootstrap_ci95_low:+.4f}, {row.uav_bootstrap_ci95_high:+.4f}] | "
                    f"[{row.familywise_ci95_low:+.4f}, {row.familywise_ci95_high:+.4f}] | {row.interpretation} |")
    if note.get("status") == "checked":
        outcome = ("reproduced every shared PE_35 fit bitwise" if note["identical"] else
                   f"differed from PE_35 by at most {note['max_abs_prediction_difference']:.6g} cycles "
                   f"(inputs_match={note['inputs_match']}, versions_match={note['versions_match']})")
        text += ["", f"Reproduction check: this run refits rather than imports the shared arms "
                 f"({', '.join(note['arms'])}) and {outcome}."]
    text += ["", "These are development ablations on previously examined UAVs with historically selected, frozen "
             "feature tiers. Bootstrap intervals resample UAVs and condition on the fitted models; they do not "
             "measure variation from retraining on different splits. Ten cutoff seeds are repeated endpoints, "
             "not ten independent training runs.", "",
             "One main XGBoost per arm/fold, no early specialist or gating. Outer scoring UAVs are excluded from "
             "feature-weight calculation, fitting, and early stopping. A fixed 10% of outer-training UAVs supplies "
             "unweighted early stopping; the model is not refit, matching v13's final-fit holdout approach. "
             "Feature weights use only actual fitting UAVs. Tree settings and all-cycle sample-weight mechanics "
             "come directly from v13.", "",
             "A removal candidate still needs confirmation. This result covers exactly this pair; it does not "
             "license removing a third channel, and it does not prove that either retained channel is necessary. "
             "No production model or submission was changed.", ""]
    (output / "reporting/report.md").write_text("\n".join(text), encoding="utf-8")
    write_json(output / "reporting/completion.json",
               {"status": "complete", "fits": len(catalogs) * config["outer_folds"], "arms": len(catalogs),
                "primary_target": "cap125", "bootstrap_unit": "uav_id", "production_changed": False})
    print(summary[["arm", "features", "rmse", "r2"]].to_string(index=False), flush=True)
    print(paired[["arm", "rmse_change", "improved_folds", "interpretation"]].to_string(index=False), flush=True)
    print(f"interaction={interaction['interaction']:+.4f} "
          f"[{interaction['interaction_ci95_low']:+.4f}, {interaction['interaction_ci95_high']:+.4f}]", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate features, partitions and endpoints without fitting")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config = tomllib.loads((HERE / "settings.toml").read_text())["study"]
    output = HERE / "runs" / config["run"]
    module, data, features, tiers, arms, catalogs, folds, membership, endpoints, cutoffs = prepare(config)
    declared = {arm for pair in config["secondary_contrasts"] for arm in pair}
    assert declared <= set(catalogs), sorted(declared - set(catalogs))
    hashes = {name: sha(ROOT / config[name]) for name in ("source", "feature_contract", "train", "test_cutoffs")}
    hashes.update(runner=sha(__file__), settings=sha(HERE / "settings.toml"))
    registration = {"config": config, "hashes": hashes, "xgboost_params": module.XGB_PARAMS,
                    "early_stopping_rounds": module.XGB_EARLY_STOPPING_ROUNDS, "target_cap": module.RUL_CAP,
                    "rolling_windows": list(module.ROLLING_WINDOWS), "sensor_tiers": tiers,
                    "arms": arms, "feature_columns": catalogs,
                    "versions": {"python": platform.python_version(), "xgboost": xgb.__version__, "numpy": np.__version__,
                                 "pandas": pd.__version__, "sklearn": sklearn.__version__},
                    "protocol": "main_xgboost_only; frozen_historical_tiers; inner_10pct_uav_early_stop; no_refit; "
                                "fit_only_feature_weights; v13_GroupKFold; ten_test_cutoff_seeds"}
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
    write_json(output / "input_verification.json",
               {"status": "passed", "feature_counts": {k: len(v) for k, v in catalogs.items()},
                "uavs": int(data.uav_id.nunique()), "training_rows": len(data), "evaluation_rows": len(endpoints),
                "causal_prefix_check": True, "disjoint_fit_stopping_score_uavs": True,
                "pair_removes_union_of_singles": True})
    print(f"Validated {len(catalogs)} arms, {len(folds)} folds, {len(endpoints)} matched endpoints", flush=True)
    if args.check:
        return
    if args.report_only:
        report(output, config, catalogs, registration)
        return
    all_columns = [c for c in features if c != "uav_id"]
    matrix = features[all_columns].to_numpy()
    y = data.RUL.to_numpy()
    sensors = [sensor for tier in tiers.values() for sensor in tier]
    total = len(catalogs) * config["outer_folds"]
    completed = 0
    for fold, fitting, stopping, held in folds:
        with redirect_stdout(io.StringIO()):
            feature_weights = module.compute_feature_weights(all_columns, data.iloc[fitting], sensors)
        sample_weights = module.compute_test_similarity_weights(data.iloc[fitting].flight_cycle, cutoffs)
        selected_endpoints = endpoints[endpoints.fold == fold].copy()
        assert set(selected_endpoints.uav_id) <= set(data.iloc[held].uav_id)
        for arm, columns in catalogs.items():
            cell_csv = output / "cells" / f"{arm}__fold_{fold}.csv"
            cell_json = cell_csv.with_suffix(".json")
            if cell_json.exists():
                if json.loads(cell_json.read_text())["prediction_sha256"] != sha(cell_csv):
                    raise RuntimeError(f"Checkpoint corrupted: {cell_csv}")
                completed += 1
                print(f"Reused {completed}/{total} {arm} fold {fold}", flush=True)
                continue
            indices = [all_columns.index(column) for column in columns]
            print(f"Fitting {completed + 1}/{total} {arm} fold {fold}: {len(columns)} features", flush=True)
            started = time.perf_counter()
            model = xgb.XGBRegressor(**module.XGB_PARAMS, early_stopping_rounds=module.XGB_EARLY_STOPPING_ROUNDS,
                                     feature_weights=feature_weights[indices])
            model.fit(matrix[np.ix_(fitting, indices)], y[fitting], sample_weight=sample_weights,
                      eval_set=[(matrix[np.ix_(stopping, indices)], y[stopping])], verbose=False)
            selected_endpoints["prediction"] = model.predict(matrix[np.ix_(selected_endpoints.row_index.to_numpy(), indices)])
            assert np.isfinite(selected_endpoints.prediction).all()
            write_csv(cell_csv, selected_endpoints)
            elapsed = time.perf_counter() - started
            write_json(cell_json, {"arm": arm, "fold": fold, "features": len(columns),
                                   "best_iteration": int(model.best_iteration), "fit_seconds": elapsed,
                                   "prediction_sha256": sha(cell_csv), "metrics": metrics(selected_endpoints),
                                   "removed_channels": arms[arm],
                                   "feature_weights": dict(zip(columns, feature_weights[indices].tolist())),
                                   "sample_weight_min": float(sample_weights.min()), "sample_weight_max": float(sample_weights.max())})
            completed += 1
            print(f"Completed {completed}/{total} in {elapsed:.1f}s; iteration={model.best_iteration}; "
                  f"RMSE={metrics(selected_endpoints)['rmse']:.4f}", flush=True)
    report(output, config, catalogs, registration)


if __name__ == "__main__":
    main()
