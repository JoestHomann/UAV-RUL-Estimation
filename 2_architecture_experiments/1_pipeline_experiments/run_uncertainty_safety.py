"""Cross-fit uncertainty-dependent conservative corrections for PE_13."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_config import read_experiment_config
from experiment_paths import gallery_directory, repository_path, run_directory
from oof_experiment_utils import regression_metrics, write_json


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]


def _adjust(rows: pd.DataFrame, strength: float, edges: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    bins = np.clip(np.digitize(rows["predicted_rul"].to_numpy(float), edges[1:-1]), 0, len(multipliers) - 1)
    correction = strength * multipliers[bins] * rows["uncertainty_std"].to_numpy(float)
    return np.maximum(rows["predicted_rul"].to_numpy(float) - correction, 0.0)


def _objective(rows: pd.DataFrame, prediction: np.ndarray, penalty: float, near_failure_limit: float) -> float:
    metrics = regression_metrics(rows["observed_rul"], prediction)
    near = rows["observed_rul"].to_numpy(float) <= near_failure_limit
    near_positive = np.maximum(prediction[near] - rows.loc[near, "observed_rul"].to_numpy(float), 0.0)
    near_rms = float(np.sqrt(np.mean(np.square(near_positive)))) if near.any() else 0.0
    return metrics["rmse"] + penalty * near_rms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_13")
    args = parser.parse_args()
    config = read_experiment_config(args.config.resolve())
    workflow = config["uncertainty_safety_workflows"][args.workflow]
    definition = config["run_definitions"][args.workflow]
    predictions = pd.read_csv(repository_path(REPOSITORY_ROOT, workflow["source_predictions"]))
    method = str(workflow["base_method"])
    if method == "auto":
        winner_path = repository_path(REPOSITORY_ROOT, workflow["source_winner_manifest"])
        winner = json.loads(winner_path.read_text(encoding="utf-8"))
        method = str(winner.get("winner", ""))
        if not method:
            raise ValueError("PE_11 winner manifest does not select a base method")
    rows = predictions.loc[predictions["method"].eq(method)].copy().reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"PE_13 base method {method!r} is missing")
    required = {"outer_fold", "inner_fold", "uav_id", "observed_rul", "predicted_rul", "uncertainty_std"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"PE_13 source predictions are missing {missing}")
    edges = np.asarray(workflow["prediction_bin_edges"], dtype=np.float64)
    multipliers = np.asarray(workflow["severity_multipliers"], dtype=np.float64)
    if len(multipliers) != len(edges) - 1 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("PE_13 bin edges and severity multipliers are inconsistent")
    strengths = [float(value) for value in workflow["strength_grid"]]
    selected = np.empty(len(rows), dtype=np.float64)
    provenance = []
    for outer_fold, outer_rows in rows.groupby("outer_fold"):
        for inner_fold, held in outer_rows.groupby("inner_fold"):
            training = outer_rows.loc[outer_rows["inner_fold"].ne(inner_fold)]
            if set(training["uav_id"]) & set(held["uav_id"]):
                raise ValueError("PE_13 cross-fitting has UAV overlap")
            scored = [
                (
                    _objective(
                        training,
                        _adjust(training, strength, edges, multipliers),
                        float(workflow["near_failure_penalty"]),
                        float(workflow["near_failure_rul"]),
                    ),
                    strength,
                )
                for strength in strengths
            ]
            _, chosen = min(scored, key=lambda item: (item[0], item[1]))
            selected[held.index] = _adjust(held, chosen, edges, multipliers)
            provenance.append({"outer_fold": int(outer_fold), "inner_fold": int(inner_fold), "selected_strength": chosen, "training_rows": len(training), "validation_rows": len(held), "uav_overlap": 0})
    methods = {"control": rows["predicted_rul"].to_numpy(float), "cross_fitted_uncertainty": selected}
    methods.update({f"strength_{strength:g}": _adjust(rows, strength, edges, multipliers) for strength in strengths})
    prediction_frames = []
    fold_records = []
    identity = [column for column in rows.columns if column not in {"method", "predicted_rul"}]
    for name, prediction in methods.items():
        frame = rows[identity].copy()
        frame["method"] = name
        frame["predicted_rul"] = prediction
        prediction_frames.append(frame)
        for fold, fold_rows in frame.groupby("outer_fold"):
            near = fold_rows.loc[
                fold_rows["observed_rul"] <= float(workflow["near_failure_rul"])
            ]
            near_residual = np.maximum(
                near["predicted_rul"].to_numpy(float)
                - near["observed_rul"].to_numpy(float),
                0.0,
            )
            fold_records.append(
                {
                    "method": name,
                    "outer_fold": int(fold),
                    **regression_metrics(
                        fold_rows["observed_rul"], fold_rows["predicted_rul"]
                    ),
                    "near_failure_rms_overprediction": float(
                        np.sqrt(np.mean(np.square(near_residual)))
                    ),
                }
            )
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    folds = pd.DataFrame(fold_records)
    summary = folds.groupby("method", as_index=False).agg(mean_r2=("r2", "mean"), mean_rmse=("rmse", "mean"), mean_overprediction_rate=("overprediction_rate", "mean"), mean_rms_overprediction=("rms_overprediction", "mean"), mean_near_failure_rms_overprediction=("near_failure_rms_overprediction", "mean")).sort_values("mean_rmse")
    control = summary.loc[summary["method"].eq("control")].iloc[0]
    candidate = summary.loc[summary["method"].eq("cross_fitted_uncertainty")].iloc[0]
    accuracy_regression = float((candidate.mean_rmse - control.mean_rmse) / control.mean_rmse)
    safety_improvement = float((control.mean_near_failure_rms_overprediction - candidate.mean_near_failure_rms_overprediction) / control.mean_near_failure_rms_overprediction)
    promoted = accuracy_regression <= float(workflow["maximum_relative_rmse_regression"]) and safety_improvement >= float(workflow["minimum_relative_rms_overprediction_improvement"])
    root = run_directory(EXPERIMENTS_DIR, args.workflow, definition)
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    all_predictions.to_csv(reporting / "method_predictions.csv.gz", index=False, compression="gzip")
    folds.to_csv(reporting / "fold_metrics.csv", index=False)
    summary.to_csv(reporting / "summary.csv", index=False)
    pd.DataFrame(provenance).to_csv(reporting / "cross_fit_provenance.csv", index=False)
    manifest = {"status": "promoted" if promoted else "no_promotion", "winner": "cross_fitted_uncertainty" if promoted else "control", "base_method": method, "promoted": promoted, "relative_rmse_regression": accuracy_regression, "relative_near_failure_rms_overprediction_improvement": safety_improvement, "uses_locked_evaluation": False}
    write_json(reporting / "winner_manifest.json", manifest)
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.scatter(summary["mean_rmse"], summary["mean_rms_overprediction"], color=["#287271" if name == manifest["winner"] else "#68768a" for name in summary["method"]])
    for row in summary.itertuples(index=False):
        axis.annotate(row.method, (row.mean_rmse, row.mean_rms_overprediction), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axis.set_xlabel("Mean development RMSE")
    axis.set_ylabel("Mean RMS overprediction")
    axis.set_title("PE_13 uncertainty-dependent safety trade-off")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure_path = reporting / "uncertainty_safety_tradeoff.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    gallery = gallery_directory(EXPERIMENTS_DIR, args.workflow, definition)
    gallery.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure_path, gallery / figure_path.name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
