"""Separate profile metrics, conditional UAV uncertainty and PE_31 promotion rules."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from followup_experiment_utils import atomic_csv, atomic_json

KEYS = ["split_seed", "outer_fold", "model_seed", "uav_id", "scenario", "suite", "endpoint_seed", "cutoff", "observed_rul"]


def metrics(rows):
    w = 1. / rows.groupby("uav_id").uav_id.transform("size").to_numpy(float)
    y, p = rows.observed_rul.to_numpy(float), rows.predicted_rul.to_numpy(float)
    return {"rmse": float(np.sqrt(np.average((p-y)**2, weights=w))),
        "r2": float(r2_score(y, p, sample_weight=w)), "mae": float(np.average(abs(p-y), weights=w)),
        "bias": float(np.average(p-y, weights=w)), "uavs": int(rows.uav_id.nunique()), "rows": len(rows)}


def score(rows, suite="historical"):
    chosen = rows.loc[rows.suite.eq(suite)]
    if chosen.empty:
        raise ValueError(f"Missing selection suite {suite}")
    return metrics(chosen)["rmse"]


def align(left, right):
    a = left.sort_values(KEYS).reset_index(drop=True)
    b = right.sort_values(KEYS).reset_index(drop=True)
    if a.duplicated(KEYS).any() or b.duplicated(KEYS).any() or not a[KEYS].equals(b[KEYS]):
        raise ValueError("Candidate/control endpoint identities do not align")
    return a, b


def blended(candidate, control, weight):
    candidate, control = align(candidate, control)
    result = candidate.copy()
    result["predicted_rul"] = weight*candidate.predicted_rul+(1-weight)*control.predicted_rul
    return result


def report(root, stage, rows, workflow):
    directory = root / "stages" / stage / "reporting"; directory.mkdir(parents=True, exist_ok=True)
    if rows.duplicated([*KEYS, "method"]).any() or not np.isfinite(rows[["observed_rul", "predicted_rul"]]).all().all():
        raise ValueError("Duplicate/nonfinite campaign predictions")
    rows.to_csv(directory / "predictions.csv.gz", index=False, compression="gzip")
    dimensions = ["method", "suite", "split_seed", "outer_fold", "model_seed", "endpoint_seed"]
    variation = [{**dict(zip(dimensions, key)), **metrics(group)} for key, group in rows.groupby(dimensions, sort=True)]
    atomic_csv(directory / "variation.csv", pd.DataFrame(variation))
    # Model-seed averaging is a defined prediction ensemble, not extra independent UAVs.
    ensemble_keys = [key for key in KEYS if key != "model_seed"]
    ensemble = rows.groupby([*ensemble_keys, "method"], as_index=False).predicted_rul.mean()
    ensemble["model_seed"] = -1
    fold_records = [{"method": method, "suite": suite, "split_seed": seed, "outer_fold": fold, **metrics(group)}
        for (method, suite, seed, fold), group in ensemble.groupby(["method", "suite", "split_seed", "outer_fold"])]
    fold_metrics = pd.DataFrame(fold_records)
    atomic_csv(directory / "fold_metrics.csv", fold_metrics)
    if stage == "benchmark":
        pairs = [("sparse_uav_mass", "sparse_unit_mean", "weight_scale_sparse"),
            ("dense_uav_mass", "dense_uav_unit_mean", "weight_scale_dense_equal_uav"),
            ("dense_cycle_mass", "dense_unit_rows", "weight_scale_dense_equal_cycle"),
            ("dense_uav_mass", "dense_cycle_mass", "relative_weighting_fixed_uav_mass"),
            ("dense_uav_unit_mean", "dense_unit_rows", "relative_weighting_fixed_row_mass"),
            ("sparse_uav_mass", "dense_uav_mass", "sampling_equal_uav_fixed_uav_mass"),
            ("sparse_unit_mean", "dense_uav_unit_mean", "sampling_equal_uav_unit_mean"),
            ("sparse_uav_mass", "dense_cycle_mass", "sampling_equal_cycle_fixed_uav_mass"),
            ("sparse_unit_mean", "dense_unit_rows", "sampling_unit_rows")]
        contrasts = []
        for left, right, name in pairs:
            a = fold_metrics.loc[fold_metrics.method.eq(left)]
            b = fold_metrics.loc[fold_metrics.method.eq(right)]
            paired = a.merge(b, on=["suite", "split_seed", "outer_fold"], suffixes=("_reference", "_candidate"), validate="one_to_one")
            for row in paired.itertuples():
                contrasts.append({"contrast": name, "reference": left, "candidate": right, "suite": row.suite,
                    "split_seed": row.split_seed, "outer_fold": row.outer_fold,
                    "rmse_change": row.rmse_candidate-row.rmse_reference})
        atomic_csv(directory / "interaction_contrasts.csv", pd.DataFrame(contrasts))
    summary = []
    for (method, suite), group in ensemble.groupby(["method", "suite"]):
        fold = fold_metrics.loc[fold_metrics.method.eq(method) & fold_metrics.suite.eq(suite)]
        summary.append({"method": method, "suite": suite, "mean_fold_rmse": float(fold.rmse.mean()), **metrics(group)})
    summary = pd.DataFrame(summary); atomic_csv(directory / "summary.csv", summary)
    comparisons = []
    for suite in sorted(ensemble.suite.unique()):
        control = ensemble.loc[ensemble.method.eq("run7") & ensemble.suite.eq(suite)]
        for method in sorted(set(ensemble.method)-{"run7"}):
            candidate, aligned_control = align(ensemble.loc[ensemble.method.eq(method) & ensemble.suite.eq(suite)], control)
            candidate["squared"] = (candidate.predicted_rul-candidate.observed_rul)**2
            aligned_control["squared"] = (aligned_control.predicted_rul-aligned_control.observed_rul)**2
            a = candidate.groupby("uav_id").squared.mean().to_numpy()
            b = aligned_control.groupby("uav_id").squared.mean().to_numpy()
            rng = np.random.default_rng(workflow["bootstrap_seed"])
            idx = rng.integers(len(a), size=(workflow["bootstrap_repetitions"], len(a)))
            deltas = np.sqrt(a[idx].mean(axis=1))-np.sqrt(b[idx].mean(axis=1))
            folds_a = fold_metrics.loc[fold_metrics.method.eq(method) & fold_metrics.suite.eq(suite)].sort_values(["split_seed", "outer_fold"])
            folds_b = fold_metrics.loc[fold_metrics.method.eq("run7") & fold_metrics.suite.eq(suite)].sort_values(["split_seed", "outer_fold"])
            residual_a = candidate.predicted_rul-candidate.observed_rul
            residual_b = aligned_control.predicted_rul-aligned_control.observed_rul
            correlation = (float(np.corrcoef(residual_a, residual_b)[0,1])
                           if residual_a.std() > 0 and residual_b.std() > 0 else None)
            comparisons.append({"method": method, "suite": suite,
                "relative_rmse_improvement": float(1-folds_a.rmse.mean()/folds_b.rmse.mean()),
                "fold_wins": int((folds_a.rmse.to_numpy()<folds_b.rmse.to_numpy()).sum()),
                "bootstrap_low": float(np.quantile(deltas, .025)), "bootstrap_high": float(np.quantile(deltas, .975)),
                "residual_correlation": correlation})
    comparisons = pd.DataFrame(comparisons); atomic_csv(directory / "paired_comparisons.csv", comparisons)
    regions = []
    for (method, suite), group in ensemble.groupby(["method", "suite"]):
        for region, mask in {"short_history": group.cutoff.le(100), "rul_1_25": group.observed_rul.between(1,25),
            "rul_26_75": group.observed_rul.between(26,75), "rul_76_125": group.observed_rul.between(76,125),
            "rul_over_125": group.observed_rul.gt(125)}.items():
            selected = group.loc[mask]
            if len(selected)>1:
                regions.append({"method": method, "suite": suite, "region": region, **metrics(selected)})
    atomic_csv(directory / "regions.csv", pd.DataFrame(regions))
    promoted = False
    if stage == "search":
        selected = comparisons.loc[comparisons.method.eq("selected_blend")].set_index("suite")
        pooled = summary.loc[summary.method.eq("selected_blend")].set_index("suite")
        promoted = bool(selected.loc["historical", "relative_rmse_improvement"] >= workflow["minimum_relative_improvement"]
            and selected.loc["historical", "bootstrap_high"] < 0
            and pooled.loc["historical", "r2"] >= workflow["minimum_pooled_r2"]
            and selected.loc["nominal", "relative_rmse_improvement"] >= -workflow["nominal_maximum_regression"]
            and selected.loc["unrestricted", "relative_rmse_improvement"] >= -workflow["stress_maximum_regression"])
    manifest = {"stage": stage, "status": "promoted_for_final_review" if promoted else "no_promotion" if stage == "search" else "diagnostic_complete",
        "promoted": promoted, "automatic_production_replacement": False, "retained_production_model": "phase3_run_7",
        "bootstrap_unit": "uav_id", "bootstrap_scope": "conditional on fitted predictions; excludes adaptive-history uncertainty",
        "model_seed_ensemble": True, "new_splits_are_not_untouched_data": True, "uses_test_labels": False}
    atomic_json(directory / "winner_manifest.json", manifest)
    return manifest
