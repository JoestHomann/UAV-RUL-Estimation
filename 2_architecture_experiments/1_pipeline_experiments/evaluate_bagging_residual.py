"""Evaluate aggregation and nested residual correction for PE_11."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from experiment_config import read_experiment_config
from experiment_paths import gallery_directory, repository_path, run_directory
from oof_experiment_utils import ALIGNMENT_KEYS, regression_metrics, write_json


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]


def _aggregate(members: pd.DataFrame) -> pd.DataFrame:
    wide = members.pivot(index=ALIGNMENT_KEYS, columns="member", values="predicted_rul")
    if wide.isna().any().any():
        raise ValueError("PE_11 members do not align on every development endpoint")
    values = wide.to_numpy(dtype=np.float64)
    result = wide.reset_index()
    result["member_mean"] = values.mean(axis=1)
    result["member_median"] = np.median(values, axis=1)
    ordered = np.sort(values, axis=1)
    result["trimmed_mean"] = ordered[:, 1:-1].mean(axis=1)
    result["uncertainty_std"] = values.std(axis=1, ddof=1)
    result["uncertainty_range"] = values.max(axis=1) - values.min(axis=1)
    xgb = wide.loc[:, [name for name in wide if name.startswith("xgboost__")]].mean(axis=1)
    extra = wide.loc[:, [name for name in wide if name.startswith("extra_trees__")]].mean(axis=1)
    result["xgboost_mean"] = xgb.to_numpy(float)
    result["extra_trees_mean"] = extra.to_numpy(float)
    result["family_disagreement"] = np.abs(
        result["xgboost_mean"] - result["extra_trees_mean"]
    )
    return result


def _cross_fitted_blend(rows: pd.DataFrame, weights: list[float]) -> tuple[np.ndarray, list[dict]]:
    prediction = np.empty(len(rows), dtype=np.float64)
    provenance: list[dict] = []
    for outer_fold, outer_rows in rows.groupby("outer_fold", sort=True):
        for inner_fold, held in outer_rows.groupby("inner_fold", sort=True):
            training = outer_rows.loc[outer_rows["inner_fold"].ne(inner_fold)]
            if set(training["uav_id"]) & set(held["uav_id"]):
                raise ValueError("PE_11 blend tuning has UAV overlap")
            scores = []
            for weight in weights:
                estimate = (
                    weight * training["xgboost_mean"]
                    + (1.0 - weight) * training["extra_trees_mean"]
                )
                scores.append((regression_metrics(training["observed_rul"], estimate)["rmse"], weight))
            _, chosen = min(scores, key=lambda item: (item[0], item[1]))
            prediction[held.index] = (
                chosen * held["xgboost_mean"]
                + (1.0 - chosen) * held["extra_trees_mean"]
            )
            provenance.append(
                {
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "selected_xgboost_weight": float(chosen),
                    "training_rows": len(training),
                    "validation_rows": len(held),
                    "uav_overlap": 0,
                }
            )
    return prediction, provenance


def _cross_fitted_residual(
    rows: pd.DataFrame,
    *,
    base_column: str,
    maximum_iterations: int,
    maximum_leaf_nodes: int,
    l2_regularization: float,
    additional_features: list[str],
) -> tuple[np.ndarray, list[dict]]:
    feature_columns = [
        base_column,
        "cutoff",
        "uncertainty_std",
        "uncertainty_range",
        "family_disagreement",
        *additional_features,
    ]
    corrected = np.empty(len(rows), dtype=np.float64)
    provenance: list[dict] = []
    for outer_fold, outer_rows in rows.groupby("outer_fold", sort=True):
        for inner_fold, held in outer_rows.groupby("inner_fold", sort=True):
            training = outer_rows.loc[outer_rows["inner_fold"].ne(inner_fold)]
            overlap = set(training["uav_id"]) & set(held["uav_id"])
            if overlap:
                raise ValueError("PE_11 residual cross-fitting has UAV overlap")
            residual = training[base_column] - training["observed_rul"]
            model = HistGradientBoostingRegressor(
                max_iter=maximum_iterations,
                max_leaf_nodes=maximum_leaf_nodes,
                min_samples_leaf=20,
                l2_regularization=l2_regularization,
                learning_rate=0.05,
                random_state=13,
            )
            model.fit(training[feature_columns], residual)
            correction = model.predict(held[feature_columns])
            corrected[held.index] = np.maximum(held[base_column] - correction, 0.0)
            provenance.append(
                {
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "training_rows": len(training),
                    "validation_rows": len(held),
                    "training_uavs": training["uav_id"].nunique(),
                    "validation_uavs": held["uav_id"].nunique(),
                    "uav_overlap": 0,
                    "features": json.dumps(feature_columns),
                }
            )
    return corrected, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_11")
    args = parser.parse_args()
    config = read_experiment_config(args.config.resolve())
    workflow = config["bagging_residual_workflows"][args.workflow]
    definition = config["run_definitions"][args.workflow]
    root = run_directory(EXPERIMENTS_DIR, args.workflow, definition)
    members = pd.read_csv(root / "members" / "member_predictions.csv.gz")
    expected_members = len(workflow["families"]) * len(workflow["seeds"])
    counts = members.groupby(ALIGNMENT_KEYS, dropna=False)["member"].nunique()
    if counts.empty or not counts.eq(expected_members).all():
        raise ValueError("PE_11 member table is incomplete")
    rows = _aggregate(members).reset_index(drop=True)
    residual_features = [str(value) for value in workflow["residual_features"]]
    static = members[[*ALIGNMENT_KEYS, *residual_features]].drop_duplicates()
    if len(static) != len(rows):
        raise ValueError("PE_11 residual feature values disagree across members")
    rows = rows.merge(static, on=ALIGNMENT_KEYS, validate="one_to_one")
    blend, blend_provenance = _cross_fitted_blend(
        rows, [float(value) for value in workflow["xgboost_weight_grid"]]
    )
    rows["nonnegative_blend"] = blend
    residual, residual_provenance = _cross_fitted_residual(
        rows,
        base_column="nonnegative_blend",
        maximum_iterations=int(workflow["residual_maximum_iterations"]),
        maximum_leaf_nodes=int(workflow["residual_maximum_leaf_nodes"]),
        l2_regularization=float(workflow["residual_l2_regularization"]),
        additional_features=residual_features,
    )
    rows["residual_corrected"] = residual

    method_columns = [
        "member_mean",
        "member_median",
        "trimmed_mean",
        "nonnegative_blend",
        "residual_corrected",
    ]
    method_frames = []
    fold_records = []
    for method in method_columns:
        frame = rows[[*ALIGNMENT_KEYS, "uncertainty_std", "uncertainty_range", "family_disagreement", *residual_features]].copy()
        frame["method"] = method
        frame["predicted_rul"] = rows[method]
        method_frames.append(frame)
        for outer_fold, fold_rows in frame.groupby("outer_fold"):
            fold_records.append(
                {
                    "method": method,
                    "outer_fold": int(outer_fold),
                    **regression_metrics(fold_rows["observed_rul"], fold_rows["predicted_rul"]),
                }
            )
    predictions = pd.concat(method_frames, ignore_index=True)
    folds = pd.DataFrame(fold_records)
    summary = folds.groupby("method", as_index=False).agg(
        folds=("outer_fold", "nunique"),
        mean_r2=("r2", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_bias=("bias", "mean"),
        mean_rms_overprediction=("rms_overprediction", "mean"),
    ).sort_values(["mean_rmse", "method"])
    winner = str(summary.iloc[0]["method"])
    control_path = repository_path(REPOSITORY_ROOT, workflow["control_predictions"])
    control = pd.read_csv(control_path)
    control = control.loc[control["method"].eq(workflow["control_method"])]
    control_folds = []
    for fold, fold_rows in control.groupby("outer_fold"):
        control_folds.append({"outer_fold": int(fold), **regression_metrics(fold_rows["observed_rul"], fold_rows["predicted_rul"])})
    control_metrics = pd.DataFrame(control_folds)
    winner_folds = folds.loc[folds["method"].eq(winner)]
    paired = winner_folds.merge(control_metrics, on="outer_fold", suffixes=("", "_control"))
    improvement = float((paired["rmse_control"].mean() - paired["rmse"].mean()) / paired["rmse_control"].mean())
    fold_wins = int((paired["rmse"] < paired["rmse_control"]).sum())
    promoted = fold_wins >= int(workflow["minimum_fold_wins"]) and improvement >= float(workflow["minimum_relative_rmse_improvement"])

    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(reporting / "method_predictions.csv.gz", index=False, compression="gzip")
    folds.to_csv(reporting / "fold_metrics.csv", index=False)
    summary.to_csv(reporting / "summary.csv", index=False)
    pd.DataFrame(blend_provenance).to_csv(reporting / "blend_provenance.csv", index=False)
    pd.DataFrame(residual_provenance).to_csv(reporting / "residual_provenance.csv", index=False)
    manifest = {
        "status": "promoted" if promoted else "no_promotion",
        "winner": winner,
        "promoted": promoted,
        "fold_wins_vs_control": fold_wins,
        "relative_rmse_improvement_vs_control": improvement,
        "control_method": workflow["control_method"],
        "uses_locked_evaluation": False,
        "member_count": expected_members,
    }
    write_json(reporting / "winner_manifest.json", manifest)
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.bar(summary["method"], summary["mean_rmse"], color=["#287271" if name == winner else "#68768a" for name in summary["method"]])
    axis.set_ylabel("Mean development RMSE")
    axis.set_title("PE_11 bagging and residual correction")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure_path = reporting / "bagging_residual_comparison.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    gallery = gallery_directory(EXPERIMENTS_DIR, args.workflow, definition)
    gallery.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure_path, gallery / figure_path.name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
