"""Evaluate fixed blends and leakage-free nested OOF meta-models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from experiment_config import read_experiment_config
from experiment_paths import gallery_directory


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]


class StackingError(ValueError):
    """Explain an invalid or leakage-prone stacking request."""


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    denominator = float(np.square(observed - observed.mean()).sum())
    return {
        "r2": 1.0 - float(np.square(residual).sum()) / denominator,
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "overprediction_rate": float(np.mean(residual > 0.0)),
        "rms_overprediction": float(
            np.sqrt(np.mean(np.square(np.maximum(residual, 0.0))))
        ),
    }


def _meta_estimator(method: str, seed: int) -> Any:
    if method == "nonnegative_ridge":
        return Ridge(alpha=1.0, positive=True)
    if method == "shallow_xgboost":
        return XGBRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.03,
            min_child_weight=5.0,
            subsample=0.8,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=5.0,
            objective="reg:squarederror",
            tree_method="hist",
            device="cpu",
            random_state=seed,
            n_jobs=1,
        )
    raise StackingError(f"Unknown fitted stack method {method!r}")


def nested_meta_predictions(
    table: pd.DataFrame,
    *,
    method: str,
    feature_columns: list[str],
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Cross-fit a meta-model inside every outer fold and record provenance."""

    required = {"outer_fold", "inner_fold", "uav_id", "observed_rul", *feature_columns}
    missing = sorted(required - set(table.columns))
    if missing:
        raise StackingError(f"Aligned OOF table is missing {missing}")
    predictions = np.full(len(table), np.nan, dtype=np.float64)
    provenance: list[dict[str, Any]] = []
    for outer_fold, outer_rows in table.groupby("outer_fold", sort=True):
        for inner_fold, validation in outer_rows.groupby("inner_fold", sort=True):
            training = outer_rows.loc[~outer_rows["inner_fold"].eq(inner_fold)]
            overlap = set(training["uav_id"]) & set(validation["uav_id"])
            if overlap:
                raise StackingError(
                    f"Meta fold {outer_fold}/{inner_fold} shares validation UAVs"
                )
            if training.empty or validation.empty:
                raise StackingError(f"Meta fold {outer_fold}/{inner_fold} is empty")
            estimator = _meta_estimator(method, seed + int(outer_fold) * 10 + int(inner_fold))
            estimator.fit(
                training[feature_columns].to_numpy(dtype=np.float64),
                training["observed_rul"].to_numpy(dtype=np.float64),
            )
            predictions[validation.index] = estimator.predict(
                validation[feature_columns].to_numpy(dtype=np.float64)
            )
            provenance.append(
                {
                    "method": method,
                    "outer_fold": int(outer_fold),
                    "held_out_inner_fold": int(inner_fold),
                    "training_rows": len(training),
                    "validation_rows": len(validation),
                    "training_uavs": int(training["uav_id"].nunique()),
                    "validation_uavs": int(validation["uav_id"].nunique()),
                    "uav_overlap": 0,
                }
            )
    if not np.isfinite(predictions).all():
        raise StackingError(f"{method} did not produce one finite OOF prediction per row")
    return np.maximum(predictions, 0.0), pd.DataFrame.from_records(provenance)


def evaluate_stacks(
    aligned: pd.DataFrame,
    *,
    tree_source: str,
    temporal_source: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tree_column = f"prediction__{tree_source}"
    temporal_column = f"prediction__{temporal_source}"
    for column in (tree_column, temporal_column):
        if column not in aligned:
            raise StackingError(f"Aligned table has no {column!r}")
    predictions: dict[str, np.ndarray] = {
        "tree_control": aligned[tree_column].to_numpy(dtype=np.float64),
        "temporal_control": aligned[temporal_column].to_numpy(dtype=np.float64),
    }
    tree = predictions["tree_control"]
    temporal = predictions["temporal_control"]
    for weight in (0.25, 0.50, 0.75):
        predictions[f"blend_{int(weight * 100):03d}"] = (
            (1.0 - weight) * tree + weight * temporal
        )
    provenance_tables: list[pd.DataFrame] = []
    for method in ("nonnegative_ridge", "shallow_xgboost"):
        prediction, provenance = nested_meta_predictions(
            aligned,
            method=method,
            feature_columns=[tree_column, temporal_column],
            seed=seed,
        )
        predictions[method] = prediction
        provenance_tables.append(provenance)

    prediction_table = aligned.copy()
    for method, values in predictions.items():
        prediction_table[f"prediction__{method}"] = values
    records: list[dict[str, Any]] = []
    for method, values in predictions.items():
        for outer_fold, rows in aligned.groupby("outer_fold", sort=True):
            observed = rows["observed_rul"].to_numpy(dtype=np.float64)
            records.append(
                {
                    "method": method,
                    "outer_fold": int(outer_fold),
                    **_metrics(observed, values[rows.index]),
                }
            )
    return (
        prediction_table,
        pd.DataFrame.from_records(records),
        pd.concat(provenance_tables, ignore_index=True),
    )


def _report(folds: pd.DataFrame) -> pd.DataFrame:
    summary = folds.groupby("method", as_index=False).agg(
        folds=("outer_fold", "nunique"),
        mean_r2=("r2", "mean"),
        sd_r2=("r2", "std"),
        mean_rmse=("rmse", "mean"),
        sd_rmse=("rmse", "std"),
        mean_bias=("bias", "mean"),
        mean_rms_overprediction=("rms_overprediction", "mean"),
    )
    control = folds.loc[folds["method"].eq("tree_control"), ["outer_fold", "rmse"]]
    control = control.rename(columns={"rmse": "control_rmse"})
    wins: list[int] = []
    improvements: list[float] = []
    for method in summary["method"]:
        rows = folds.loc[folds["method"].eq(method), ["outer_fold", "rmse"]]
        paired = rows.merge(control, on="outer_fold", validate="one_to_one")
        wins.append(int((paired["rmse"] < paired["control_rmse"]).sum()))
        improvements.append(
            float((paired["control_rmse"].mean() - paired["rmse"].mean()) / paired["control_rmse"].mean())
        )
    summary["fold_wins_over_tree"] = wins
    summary["relative_rmse_improvement"] = improvements
    summary["passes_gate"] = (
        (~summary["method"].isin(["tree_control", "temporal_control"]))
        & (summary["mean_r2"] >= 0.91)
        & (summary["mean_rmse"] <= 9.5)
        & (summary["fold_wins_over_tree"] >= 4)
        & (summary["relative_rmse_improvement"] >= 0.03)
    )
    return summary.sort_values(["mean_rmse", "mean_r2"], ascending=[True, False])


def _plot(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.sort_values("mean_rmse")
    figure, axis = plt.subplots(figsize=(11, 5.5))
    colors = ["#27806b" if passed else "#386cb0" for passed in ordered["passes_gate"]]
    axis.bar(ordered["method"], ordered["mean_rmse"], color=colors)
    axis.axhline(9.5, color="#b33a3a", linestyle="--", linewidth=1.2)
    axis.set_ylabel("Mean development RMSE")
    axis.tick_params(axis="x", rotation=25)
    axis.set_title("PE_7 leakage-free OOF stacking")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_7")
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = read_experiment_config(args.config.resolve())
    workflow = config.get("stacking_workflows", {}).get(args.workflow)
    if not isinstance(workflow, dict):
        raise StackingError(f"Unknown stacking workflow {args.workflow!r}")
    aligned = pd.read_csv(args.aligned.resolve())
    predictions, folds, provenance = evaluate_stacks(
        aligned,
        tree_source=str(workflow.get("tree_source", "tree")),
        temporal_source=str(workflow.get("temporal_source", "temporal")),
        seed=int(workflow.get("seed", 13)),
    )
    summary = _report(folds)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "stack_oof_predictions.csv.gz", index=False, compression="gzip")
    folds.to_csv(output_dir / "stack_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "stack_summary.csv", index=False)
    provenance.to_csv(output_dir / "meta_fold_provenance.csv", index=False)
    figure = output_dir / "stacking_comparison.png"
    _plot(summary, figure)
    definition = config.get("run_definitions", {}).get(args.workflow)
    if isinstance(definition, dict):
        gallery = gallery_directory(EXPERIMENTS_DIR, args.workflow, definition)
        gallery.mkdir(parents=True, exist_ok=True)
        shutil.copy2(figure, gallery / figure.name)
    winner_rows = summary.loc[summary["passes_gate"]]
    winner = None if winner_rows.empty else str(winner_rows.iloc[0]["method"])
    manifest = {
        "status": "promoted" if winner else "no_promotion",
        "winner": winner,
        "locked_evaluation_opened": False,
        "artifacts": {
            "predictions": (output_dir / "stack_oof_predictions.csv.gz").relative_to(REPOSITORY_ROOT).as_posix(),
            "fold_metrics": (output_dir / "stack_fold_metrics.csv").relative_to(REPOSITORY_ROOT).as_posix(),
            "summary": (output_dir / "stack_summary.csv").relative_to(REPOSITORY_ROOT).as_posix(),
            "provenance": (output_dir / "meta_fold_provenance.csv").relative_to(REPOSITORY_ROOT).as_posix(),
        },
    }
    (output_dir / "stacking_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
