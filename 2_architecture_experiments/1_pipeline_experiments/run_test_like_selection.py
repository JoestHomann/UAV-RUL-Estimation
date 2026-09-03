"""Rank existing OOF candidates under PE_12 test-like development weights."""

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
COMMON_KEYS = ["outer_fold", "inner_fold", "uav_id", "scenario", "cutoff", "observed_rul"]


def _read_source(source: dict) -> pd.DataFrame:
    path = repository_path(REPOSITORY_ROOT, str(source["path"]))
    if not path.is_file():
        if source.get("required", True):
            raise ValueError(f"Required PE_12 source is missing: {path}")
        return pd.DataFrame()
    table = pd.read_csv(path)
    prediction_column = str(source.get("prediction_column", "predicted_rul"))
    method_column = str(source.get("method_column", "method"))
    if prediction_column not in table:
        raise ValueError(f"PE_12 source {path} has no {prediction_column!r}")
    if method_column not in table:
        table[method_column] = str(source["name"])
    allowed = source.get("methods")
    if isinstance(allowed, list) and allowed:
        table = table.loc[table[method_column].astype(str).isin(map(str, allowed))]
    missing = sorted(set(COMMON_KEYS) - set(table.columns))
    if missing:
        raise ValueError(f"PE_12 source {path} is missing {missing}")
    result = table[[*COMMON_KEYS, method_column, prediction_column]].copy()
    result["candidate"] = str(source["name"]) + "::" + result[method_column].astype(str)
    result["predicted_rul"] = result[prediction_column].astype(float)
    return result[[*COMMON_KEYS, "candidate", "predicted_rul"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_12")
    args = parser.parse_args()
    config = read_experiment_config(args.config.resolve())
    workflow = config["test_like_selection_workflows"][args.workflow]
    definition = config["run_definitions"][args.workflow]
    source_frames = [_read_source(source) for source in workflow["sources"]]
    candidates = pd.concat([frame for frame in source_frames if not frame.empty], ignore_index=True)
    if candidates.empty:
        raise ValueError("PE_12 has no available OOF candidate predictions")
    propensity = pd.read_csv(repository_path(REPOSITORY_ROOT, workflow["domain_propensity"]))
    propensity = propensity.loc[propensity["domain"].eq("development"), ["uav_id", "scenario", "cutoff", "propensity"]]
    candidates = candidates.merge(
        propensity, on=["uav_id", "scenario", "cutoff"], how="left", validate="many_to_one"
    )
    if candidates["propensity"].isna().any():
        raise ValueError("PE_12 candidate rows do not align with PE_9 propensity")
    epsilon = float(workflow["propensity_epsilon"])
    clipped = candidates["propensity"].clip(epsilon, 1.0 - epsilon)
    lower, upper = map(float, workflow["weight_clip"])
    candidates["test_like_weight"] = (clipped / (1.0 - clipped)).clip(lower, upper)
    candidates["test_like_weight"] /= candidates.groupby("candidate")["test_like_weight"].transform("mean")
    threshold = float(propensity["propensity"].quantile(float(workflow["high_propensity_quantile"])))
    records = []
    for (candidate, fold), rows in candidates.groupby(["candidate", "outer_fold"]):
        residual = rows["predicted_rul"].to_numpy(float) - rows["observed_rul"].to_numpy(float)
        weights = rows["test_like_weight"].to_numpy(float)
        high = rows.loc[rows["propensity"] >= threshold]
        records.append(
            {
                "candidate": candidate,
                "outer_fold": int(fold),
                **regression_metrics(rows["observed_rul"], rows["predicted_rul"]),
                "test_like_rmse": float(np.sqrt(np.average(np.square(residual), weights=weights))),
                "high_propensity_rmse": regression_metrics(high["observed_rul"], high["predicted_rul"])["rmse"] if not high.empty else np.nan,
                "effective_sample_size": float(np.square(weights.sum()) / np.square(weights).sum()),
            }
        )
    folds = pd.DataFrame(records)
    summary = folds.groupby("candidate", as_index=False).agg(
        folds=("outer_fold", "nunique"),
        mean_rmse=("rmse", "mean"),
        mean_r2=("r2", "mean"),
        mean_test_like_rmse=("test_like_rmse", "mean"),
        mean_high_propensity_rmse=("high_propensity_rmse", "mean"),
        mean_effective_sample_size=("effective_sample_size", "mean"),
    ).sort_values(["mean_test_like_rmse", "mean_rmse"])
    control_name = str(workflow["control_candidate"])
    control = summary.loc[summary["candidate"].eq(control_name)]
    if len(control) != 1:
        raise ValueError(f"PE_12 control candidate {control_name!r} is missing")
    control_rmse = float(control.iloc[0]["mean_rmse"])
    minimum_ess = float(workflow["minimum_effective_sample_fraction"]) * float(candidates.groupby("candidate").size().min() / folds["outer_fold"].nunique())
    summary["passes_guard"] = (
        (summary["mean_rmse"] <= control_rmse * (1.0 + float(workflow["maximum_unweighted_rmse_regression"])))
        & (summary["mean_effective_sample_size"] >= minimum_ess)
    )
    eligible = summary.loc[summary["passes_guard"]]
    winner = control_name if eligible.empty else str(eligible.iloc[0]["candidate"])
    root = run_directory(EXPERIMENTS_DIR, args.workflow, definition)
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(reporting / "weighted_oof_predictions.csv.gz", index=False, compression="gzip")
    folds.to_csv(reporting / "test_like_fold_metrics.csv", index=False)
    summary.to_csv(reporting / "test_like_summary.csv", index=False)
    manifest = {
        "status": "complete",
        "winner": winner,
        "control": control_name,
        "changed_selection": winner != control_name,
        "uses_test_labels": False,
        "uses_locked_evaluation": False,
        "mean_domain_propensity": float(propensity["propensity"].mean()),
        "high_propensity_threshold": threshold,
    }
    write_json(reporting / "winner_manifest.json", manifest)
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    axis.scatter(summary["mean_rmse"], summary["mean_test_like_rmse"], color=["#287271" if name == winner else "#68768a" for name in summary["candidate"]])
    for row in summary.itertuples(index=False):
        axis.annotate(row.candidate, (row.mean_rmse, row.mean_test_like_rmse), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axis.set_xlabel("Ordinary development RMSE")
    axis.set_ylabel("Test-like weighted RMSE")
    axis.set_title("PE_12 test-like candidate selection")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure_path = reporting / "test_like_selection.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    gallery = gallery_directory(EXPERIMENTS_DIR, args.workflow, definition)
    gallery.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure_path, gallery / figure_path.name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

