"""Matched 2x2 of the training-row weighting in the v13 main XGBoost: test-cutoff similarity x per-UAV normalisation."""
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

# v13 weights training rows by how close their flight_cycle is to a real test
# cutoff, but never normalises per UAV, so a 525-cycle UAV contributes 525 rows
# and a 145-cycle one contributes 145. These two factors are crossed.
ARMS = {
    "baseline": {"similarity": True, "uav_equal": False},
    "similarity_uav_equal": {"similarity": True, "uav_equal": True},
    "uav_equal_only": {"similarity": False, "uav_equal": True},
    "unweighted": {"similarity": False, "uav_equal": False},
}


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
    spec = importlib.util.spec_from_file_location("pe40_v13", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def equalise_by_uav(weights, uav_ids):
    """Rescale so every UAV carries the same total weight, keeping its internal profile.

    Dividing each row by its own UAV's total weight makes every UAV sum to one,
    whatever its lifetime and whatever its rows' similarity weights. The result is
    renormalised to mean one so it stays on the same scale as v13's weights and the
    tree's regularisation means the same thing across arms.
    """
    weights = np.asarray(weights, dtype=float)
    uav_ids = np.asarray(uav_ids)
    assert len(weights) == len(uav_ids) and (weights > 0).all()
    totals = pd.Series(weights).groupby(pd.Series(uav_ids)).transform("sum").to_numpy()
    equalised = weights / totals
    return equalised / equalised.mean()


def arm_weights(recipe, similarity, uav_ids):
    """The only quantity that differs between arms."""
    weights = np.asarray(similarity, dtype=float) if recipe["similarity"] else np.ones(len(uav_ids))
    weights = weights / weights.mean()
    if recipe["uav_equal"]:
        weights = equalise_by_uav(weights, uav_ids)
    return weights


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
    # Ensure the full-history computation is identical to a genuinely truncated
    # trajectory, including rolling/expanding summaries (no future observations).
    probe_id = data.uav_id.iloc[0]
    probe = data[(data.uav_id == probe_id) & (data.flight_cycle <= 37)].copy()
    np.testing.assert_allclose(module.build_features_tiered(probe, tiers)[columns].to_numpy(),
                               features.loc[probe.index, columns].to_numpy(), rtol=1e-12, atol=1e-12)
    # Lifetimes differ by a factor of more than three, which is the whole reason
    # per-UAV normalisation might matter. Record the spread that the arms react to.
    rows_per_uav = data.groupby("uav_id").size()
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
    return (module, data, features, columns, tiers, folds, pd.DataFrame(membership), endpoints, cutoffs,
            {"min_rows": int(rows_per_uav.min()), "max_rows": int(rows_per_uav.max()),
             "ratio": float(rows_per_uav.max() / rows_per_uav.min())})


def metrics(frame, target="target_capped"):
    actual, predicted = frame[target], frame.prediction
    return {"rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "mae": float(mean_absolute_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted)),
            "bias": float(np.mean(predicted - actual))}


def bootstrap_rmse(predictions, arms, uavs, draws):
    """Paired UAV bootstrap RMSE per arm, drawn on one shared set of resamples."""
    reference, result = None, {}
    for arm in arms:
        frame = predictions[predictions.arm == arm].set_index(["evaluation_seed", "uav_id", "flight_cycle"]).sort_index()
        assert not frame.empty, arm
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
                interpretation="weighting_candidate" if adjusted_hi < 0 else "weighting_harmful" if adjusted_lo > 0 else "inconclusive")


def interaction(boot, summary):
    """Does per-UAV normalisation do the same thing with and without similarity weights?"""
    rmse = summary.set_index("arm").rmse
    with_similarity = float(rmse["similarity_uav_equal"] - rmse["baseline"])
    without_similarity = float(rmse["uav_equal_only"] - rmse["unweighted"])
    samples = ((boot["similarity_uav_equal"] - boot["baseline"])
               - (boot["uav_equal_only"] - boot["unweighted"]))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"uav_equal_effect_with_similarity": with_similarity,
            "uav_equal_effect_without_similarity": without_similarity,
            "interaction": with_similarity - without_similarity,
            "interaction_ci95_low": float(lo), "interaction_ci95_high": float(hi),
            "interpretation": "inconclusive" if lo <= 0 <= hi else "factors_interact"}


def report(output, config, arms):
    predictions = []
    for arm in arms:
        for fold in range(config["outer_folds"]):
            frame = pd.read_csv(output / "cells" / f"{arm}__fold_{fold}.csv")
            frame["arm"] = arm
            predictions.append(frame)
    predictions = pd.concat(predictions, ignore_index=True)
    write_csv(output / "reporting/predictions.csv", predictions)
    rows = []
    for arm, frame in predictions.groupby("arm", sort=False):
        row = {"arm": arm, "similarity": arms[arm]["similarity"], "uav_equal": arms[arm]["uav_equal"], **metrics(frame)}
        row.update({f"raw_{k}": v for k, v in metrics(frame, "target_raw").items()})
        row["mean_test_like_seed_r2"] = float(np.mean([metrics(part)["r2"] for _, part in frame.groupby("evaluation_seed")]))
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("arm").loc[list(arms)].reset_index()
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
    boot = bootstrap_rmse(predictions, list(arms), uavs, draws)
    treatments = [arm for arm in arms if arm != "baseline"]
    paired = pd.DataFrame([paired_change(boot, summary, fold_metrics, arm, "baseline", len(treatments))
                           for arm in treatments])
    write_csv(output / "reporting/paired_weighting.csv", paired)
    cross = interaction(boot, summary)
    write_json(output / "reporting/interaction.json", cross)
    contrasts = [tuple(pair) for pair in config["secondary_contrasts"]]
    secondary = pd.DataFrame([paired_change(boot, summary, fold_metrics, t, r, len(contrasts)) for t, r in contrasts])
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

    text = ["# PE_40: v13 XGBoost training-row weighting", "",
            "Completed 20 matched fits crossing two factors: v13's test-cutoff similarity weights, "
            "and per-UAV normalisation of the training rows. Features, folds, endpoints, tree settings "
            "and the unweighted early-stopping rule are identical in every arm.", "",
            "Primary target: RUL capped at 125, matching v13. Negative RMSE change favors the arm over "
            "v13 as shipped. Each UAV has ten test-like cutoffs and equal total evaluation weight.", "",
            "| Arm | Similarity | UAV-equal | RMSE | R2 | Mean cutoff-seed R2 |",
            "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in summary.itertuples():
        text.append(f"| {row.arm} | {'yes' if row.similarity else 'no'} | {'yes' if row.uav_equal else 'no'} | "
                    f"{row.rmse:.4f} | {row.r2:.5f} | {row.mean_test_like_seed_r2:.5f} |")
    text += ["", f"| Arm | RMSE change vs. v13 | Fold wins | Paired UAV 95% interval | "
             f"{len(treatments)}-comparison adjusted interval | Assessment |",
             "| --- | ---: | ---: | --- | --- | --- |"]
    for row in paired.sort_values("rmse_change").itertuples():
        text.append(f"| {row.arm} | {row.rmse_change:+.4f} | {row.improved_folds}/{row.total_folds} | "
                    f"[{row.uav_bootstrap_ci95_low:+.4f}, {row.uav_bootstrap_ci95_high:+.4f}] | "
                    f"[{row.familywise_ci95_low:+.4f}, {row.familywise_ci95_high:+.4f}] | {row.interpretation} |")
    text += ["", "## Do the two factors interact?", "",
             "PE_2 showed that two plausible changes to this pipeline could each hurt alone and help "
             "together, which is why they are crossed here rather than tested one at a time. The "
             "interaction is the effect of per-UAV normalisation with similarity weights present, minus "
             "its effect without them.", "",
             f"- With similarity weights: {cross['uav_equal_effect_with_similarity']:+.4f} cycles",
             f"- Without similarity weights: {cross['uav_equal_effect_without_similarity']:+.4f} cycles",
             f"- Interaction: {cross['interaction']:+.4f} cycles, 95% interval "
             f"[{cross['interaction_ci95_low']:+.4f}, {cross['interaction_ci95_high']:+.4f}] "
             f"({cross['interpretation']})", "",
             "Pre-declared secondary contrasts, adjusted over their own family. They isolate what each "
             "factor buys against no weighting at all; only the comparisons against v13 are decisive.", "",
             "| Contrast | RMSE change | Fold wins | Paired UAV 95% interval | Adjusted interval | Assessment |",
             "| --- | ---: | ---: | --- | --- | --- |"]
    for row in secondary.itertuples():
        text.append(f"| {row.arm} vs. {row.reference} | {row.rmse_change:+.4f} | {row.improved_folds}/{row.total_folds} | "
                    f"[{row.uav_bootstrap_ci95_low:+.4f}, {row.uav_bootstrap_ci95_high:+.4f}] | "
                    f"[{row.familywise_ci95_low:+.4f}, {row.familywise_ci95_high:+.4f}] | {row.interpretation} |")
    text += ["", "Weights are computed from the fitting UAVs of each fold only, exactly as v13 computes them, "
             "and every arm is renormalised to mean one so the tree's regularisation means the same thing "
             "throughout. Per-UAV normalisation divides each row by its own UAV's total weight, so every UAV "
             "carries the same total influence while keeping its internal cycle profile. Early stopping stays "
             "unweighted in all four arms, as in v13.", "",
             "These are development ablations on previously examined UAVs with historically selected, frozen "
             "feature tiers. Bootstrap intervals resample UAVs and condition on the fitted models; they do not "
             "measure variation from retraining on different splits. Ten cutoff seeds are repeated endpoints, "
             "not ten independent training runs.", "",
             "This experiment changes the weight on training rows and nothing else. It is not a test of "
             "`current20`-style prefix resampling, which also changes how many rows exist. No production model "
             "or submission was changed.", ""]
    (output / "reporting/report.md").write_text("\n".join(text), encoding="utf-8")
    write_json(output / "reporting/completion.json",
               {"status": "complete", "fits": len(arms) * config["outer_folds"], "arms": len(arms),
                "primary_target": "cap125", "bootstrap_unit": "uav_id", "production_changed": False})
    print(summary[["arm", "similarity", "uav_equal", "rmse", "r2"]].to_string(index=False), flush=True)
    print(paired[["arm", "rmse_change", "improved_folds", "interpretation"]].to_string(index=False), flush=True)
    print(f"interaction={cross['interaction']:+.4f} "
          f"[{cross['interaction_ci95_low']:+.4f}, {cross['interaction_ci95_high']:+.4f}]", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate features, weights, partitions and endpoints without fitting")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config = tomllib.loads((HERE / "settings.toml").read_text())["study"]
    output = HERE / "runs" / config["run"]
    module, data, features, columns, tiers, folds, membership, endpoints, cutoffs, spread = prepare(config)
    declared = {arm for pair in config["secondary_contrasts"] for arm in pair}
    assert declared <= set(ARMS), sorted(declared - set(ARMS))
    hashes = {name: sha(ROOT / config[name]) for name in ("source", "feature_contract", "train", "test_cutoffs")}
    hashes.update(runner=sha(__file__), settings=sha(HERE / "settings.toml"))
    registration = {"config": config, "hashes": hashes, "xgboost_params": module.XGB_PARAMS,
                    "early_stopping_rounds": module.XGB_EARLY_STOPPING_ROUNDS, "target_cap": module.RUL_CAP,
                    "rolling_windows": list(module.ROLLING_WINDOWS), "sensor_tiers": tiers,
                    "arms": ARMS, "feature_columns": columns, "rows_per_uav": spread,
                    "weight_bandwidth": getattr(module, "WEIGHT_BANDWIDTH", None),
                    "weight_floor_frac": getattr(module, "WEIGHT_FLOOR_FRAC", None),
                    "versions": {"python": platform.python_version(), "xgboost": xgb.__version__, "numpy": np.__version__,
                                 "pandas": pd.__version__, "sklearn": sklearn.__version__},
                    "protocol": "main_xgboost_only; frozen_historical_tiers; identical_features_all_arms; "
                                "inner_10pct_uav_early_stop; unweighted_early_stopping; no_refit; "
                                "fit_only_weights; v13_GroupKFold; ten_test_cutoff_seeds"}
    manifest = output / "registration.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != registration:
            raise RuntimeError("Inputs, code or settings changed. Choose a new study.run before running.")
    else:
        write_json(manifest, registration)
        shutil.copyfile(ROOT / config["source"], output / "v13_source_snapshot.py")
    write_csv(output / "fold_membership.csv", membership)
    write_csv(output / "evaluation_endpoints.csv", endpoints)
    write_json(output / "feature_catalog.json", {arm: columns for arm in ARMS})

    # Weight diagnostics per fold and arm: what the arms actually do to the data.
    diagnostics = []
    for fold, fitting, stopping, held in folds:
        similarity = module.compute_test_similarity_weights(data.iloc[fitting].flight_cycle, cutoffs)
        uav_ids = data.iloc[fitting].uav_id.to_numpy()
        for arm, recipe in ARMS.items():
            w = arm_weights(recipe, similarity, uav_ids)
            per_uav = pd.Series(w).groupby(pd.Series(uav_ids)).sum()
            diagnostics.append({"arm": arm, "fold": fold, "rows": len(w),
                                "weight_min": float(w.min()), "weight_max": float(w.max()),
                                "weight_mean": float(w.mean()),
                                "uav_total_min": float(per_uav.min()), "uav_total_max": float(per_uav.max()),
                                "uav_total_ratio": float(per_uav.max() / per_uav.min())})
    diagnostics = pd.DataFrame(diagnostics)
    write_csv(output / "weight_diagnostics.csv", diagnostics)
    equalised = diagnostics[diagnostics.arm.isin([a for a, r in ARMS.items() if r["uav_equal"]])]
    assert np.allclose(equalised.uav_total_ratio, 1.0), "per-UAV normalisation did not equalise UAV totals"
    write_json(output / "input_verification.json",
               {"status": "passed", "arms": len(ARMS), "features": len(columns),
                "uavs": int(data.uav_id.nunique()), "training_rows": len(data), "evaluation_rows": len(endpoints),
                "causal_prefix_check": True, "disjoint_fit_stopping_score_uavs": True,
                "identical_features_all_arms": True, "uav_totals_equalised_where_declared": True,
                "rows_per_uav": spread})
    print(f"Validated {len(ARMS)} arms, {len(folds)} folds, {len(endpoints)} matched endpoints; "
          f"rows per UAV {spread['min_rows']}-{spread['max_rows']} (ratio {spread['ratio']:.2f})", flush=True)
    if args.check:
        return
    if args.report_only:
        report(output, config, ARMS)
        return

    matrix = features[columns].to_numpy()
    y = data.RUL.to_numpy()
    sensors = [sensor for tier in tiers.values() for sensor in tier]
    total = len(ARMS) * config["outer_folds"]
    completed = 0
    for fold, fitting, stopping, held in folds:
        with redirect_stdout(io.StringIO()):
            feature_weights = module.compute_feature_weights(columns, data.iloc[fitting], sensors)
        similarity = module.compute_test_similarity_weights(data.iloc[fitting].flight_cycle, cutoffs)
        uav_ids = data.iloc[fitting].uav_id.to_numpy()
        selected_endpoints = endpoints[endpoints.fold == fold].copy()
        assert set(selected_endpoints.uav_id) <= set(data.iloc[held].uav_id)
        for arm, recipe in ARMS.items():
            cell_csv = output / "cells" / f"{arm}__fold_{fold}.csv"
            cell_json = cell_csv.with_suffix(".json")
            if cell_json.exists():
                if json.loads(cell_json.read_text())["prediction_sha256"] != sha(cell_csv):
                    raise RuntimeError(f"Checkpoint corrupted: {cell_csv}")
                completed += 1
                print(f"Reused {completed}/{total} {arm} fold {fold}", flush=True)
                continue
            weights = arm_weights(recipe, similarity, uav_ids)
            print(f"Fitting {completed + 1}/{total} {arm} fold {fold}", flush=True)
            started = time.perf_counter()
            model = xgb.XGBRegressor(**module.XGB_PARAMS, early_stopping_rounds=module.XGB_EARLY_STOPPING_ROUNDS,
                                     feature_weights=feature_weights)
            model.fit(matrix[fitting], y[fitting], sample_weight=weights,
                      eval_set=[(matrix[stopping], y[stopping])], verbose=False)
            selected_endpoints["prediction"] = model.predict(matrix[selected_endpoints.row_index.to_numpy()])
            assert np.isfinite(selected_endpoints.prediction).all()
            write_csv(cell_csv, selected_endpoints)
            elapsed = time.perf_counter() - started
            per_uav = pd.Series(weights).groupby(pd.Series(uav_ids)).sum()
            write_json(cell_json, {"arm": arm, "fold": fold, "recipe": recipe, "features": len(columns),
                                   "best_iteration": int(model.best_iteration), "fit_seconds": elapsed,
                                   "prediction_sha256": sha(cell_csv), "metrics": metrics(selected_endpoints),
                                   "weight_min": float(weights.min()), "weight_max": float(weights.max()),
                                   "weight_mean": float(weights.mean()),
                                   "uav_total_ratio": float(per_uav.max() / per_uav.min())})
            completed += 1
            print(f"Completed {completed}/{total} in {elapsed:.1f}s; iteration={model.best_iteration}; "
                  f"RMSE={metrics(selected_endpoints)['rmse']:.4f}", flush=True)
    report(output, config, ARMS)


if __name__ == "__main__":
    main()
