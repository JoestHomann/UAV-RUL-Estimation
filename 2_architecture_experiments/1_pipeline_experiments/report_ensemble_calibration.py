"""Evaluate fixed heterogeneous blends and cross-fitted residual calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_config import ExperimentConfigError, read_experiment_config
from experiment_paths import artifact_directory


ARCHITECTURE_EXPERIMENTS_ROOT = SCRIPT_DIR.parent
REPOSITORY_ROOT = ARCHITECTURE_EXPERIMENTS_ROOT.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "pipeline_experiments.toml"
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
PAIR_KEYS = [
    "outer_fold",
    "inner_fold",
    "validation_row",
    "uav_id",
    "scenario",
    "cutoff",
    "observed_rul",
]


class EnsembleReportError(ValueError):
    """Explain an invalid source or post-processing definition."""


def _read_config(path: Path) -> dict[str, Any]:
    try:
        payload = read_experiment_config(path)
    except ExperimentConfigError as error:
        raise EnsembleReportError(f"Cannot read experiment catalog: {error}") from error
    if not isinstance(payload, dict):
        raise EnsembleReportError("Experiment catalog must contain an object")
    return payload


def _paired_predictions(
    config: dict[str, Any],
    group_name: str,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    groups = config.get("experiment_groups", {})
    experiments = config.get("experiments", {})
    group = groups.get(group_name) if isinstance(groups, dict) else None
    if not isinstance(group, dict):
        raise EnsembleReportError(f"Unknown post-processing group {group_name!r}")
    source_name = group.get("control")
    source = experiments.get(source_name) if isinstance(experiments, dict) else None
    if not isinstance(source_name, str) or not isinstance(source, dict):
        raise EnsembleReportError("Ensemble group must name one source experiment")
    path = (
        artifact_directory(EXPERIMENTS_DIR, source_name, source)
        / "phase2"
        / "5_inner_model_selection"
        / "selected_inner_predictions.csv.gz"
    )
    try:
        predictions = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise EnsembleReportError(
            f"Cannot read selected development predictions at {path}: {error}"
        ) from error
    required = {*PAIR_KEYS, "model_family", "predicted_rul"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise EnsembleReportError(f"Selected predictions are missing {missing}")
    families = set(predictions["model_family"].astype(str))
    expected = {"extra_trees", "xgboost"}
    if not expected.issubset(families):
        raise EnsembleReportError(
            f"Ensemble source needs {sorted(expected)}, observed {sorted(families)}"
        )
    paired = predictions.pivot(
        index=PAIR_KEYS,
        columns="model_family",
        values="predicted_rul",
    ).reset_index()
    if paired[["extra_trees", "xgboost"]].isna().any().any():
        raise EnsembleReportError("XGBoost and ExtraTrees prediction rows do not align")
    return paired, group, source_name


def _metric_record(
    method: str,
    outer_fold: int,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    residual = predictions - targets
    positive = np.maximum(residual, 0.0)
    denominator = float(np.sum(np.square(targets - np.mean(targets))))
    return {
        "method": method,
        "outer_fold": int(outer_fold),
        "rows": int(len(targets)),
        "r2": 1.0 - float(np.sum(np.square(residual))) / denominator,
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "overprediction_rate": float(np.mean(residual > 0.0)),
        "rms_overprediction": float(np.sqrt(np.mean(np.square(positive)))),
        "overprediction_q95": float(np.quantile(positive, 0.95)),
        "maximum_overprediction": float(np.max(positive)),
    }


def _cross_fitted_calibration(
    table: pd.DataFrame,
    raw_column: str,
    *,
    degree: int,
    alpha: float,
) -> np.ndarray:
    adjusted = np.empty(len(table), dtype=np.float64)
    for outer_fold, outer_rows in table.groupby("outer_fold", sort=True):
        del outer_fold
        for inner_fold in sorted(outer_rows["inner_fold"].unique()):
            validation_mask = outer_rows["inner_fold"].eq(inner_fold)
            training = outer_rows.loc[~validation_mask]
            validation = outer_rows.loc[validation_mask]
            if training.empty or validation.empty:
                raise EnsembleReportError("Calibration fold is empty")
            model = make_pipeline(
                PolynomialFeatures(degree=degree, include_bias=False),
                StandardScaler(),
                Ridge(alpha=alpha),
            )
            features = [raw_column, "cutoff"]
            training_residual = (
                training[raw_column].to_numpy(dtype=np.float64)
                - training["observed_rul"].to_numpy(dtype=np.float64)
            )
            model.fit(training[features], training_residual)
            correction = model.predict(validation[features])
            adjusted[validation.index.to_numpy()] = np.maximum(
                validation[raw_column].to_numpy(dtype=np.float64) - correction,
                0.0,
            )
    return adjusted


def _plot(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.sort_values("mean_r2", ascending=False).reset_index(drop=True)
    x = np.arange(len(ordered))
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].bar(x, ordered["mean_r2"], color="#2878b5")
    axes[0].set_ylabel("Mean development-fold R2")
    axes[0].axhline(0.9, color="#c33c35", linestyle="--", linewidth=1.2)
    axes[1].bar(x, ordered["mean_overprediction_rate"], color="#d98b2b")
    axes[1].set_ylabel("Mean overprediction rate")
    axes[1].set_xticks(x, ordered["method"], rotation=25, ha="right")
    axes[1].set_xlabel("Predeclared post-processing method")
    figure.suptitle("Heterogeneous ensemble and residual-calibration study")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(
    config: dict[str, Any],
    group_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    paired, group, source_name = _paired_predictions(config, group_name)
    weights = group.get("blend_weights", [0.25, 0.5, 0.75])
    if (
        not isinstance(weights, list)
        or not weights
        or not all(isinstance(value, (int, float)) and 0.0 < value < 1.0 for value in weights)
    ):
        raise EnsembleReportError("blend_weights must be numeric values in (0, 1)")
    degree = int(group.get("calibration_degree", 2))
    alpha = float(group.get("calibration_ridge_alpha", 10.0))
    if degree not in {1, 2} or alpha < 0.0:
        raise EnsembleReportError("Calibration degree must be 1 or 2 and alpha nonnegative")

    method_columns = {
        "extra_trees": "extra_trees",
        "xgboost": "xgboost",
    }
    for weight in weights:
        label = f"blend_xgb_{float(weight):.2f}"
        paired[label] = (
            float(weight) * paired["xgboost"]
            + (1.0 - float(weight)) * paired["extra_trees"]
        )
        method_columns[label] = label

    output_rows: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    for method, column in list(method_columns.items()):
        for calibrated in (False, True):
            method_name = f"{method}__calibrated" if calibrated else method
            values = (
                _cross_fitted_calibration(
                    paired,
                    column,
                    degree=degree,
                    alpha=alpha,
                )
                if calibrated
                else paired[column].to_numpy(dtype=np.float64)
            )
            result = paired[PAIR_KEYS].copy()
            result["method"] = method_name
            result["predicted_rul"] = values
            result["residual"] = values - result["observed_rul"]
            output_rows.append(result)
            for outer_fold, rows in result.groupby("outer_fold", sort=True):
                fold_records.append(
                    _metric_record(
                        method_name,
                        int(outer_fold),
                        rows["observed_rul"].to_numpy(dtype=np.float64),
                        rows["predicted_rul"].to_numpy(dtype=np.float64),
                    )
                )

    predictions = pd.concat(output_rows, ignore_index=True)
    folds = pd.DataFrame.from_records(fold_records)
    summary = (
        folds.groupby("method", as_index=False)
        .agg(
            folds=("outer_fold", "nunique"),
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_bias=("bias", "mean"),
            mean_overprediction_rate=("overprediction_rate", "mean"),
            mean_rms_overprediction=("rms_overprediction", "mean"),
            mean_overprediction_q95=("overprediction_q95", "mean"),
        )
        .sort_values(["mean_r2", "mean_rmse"], ascending=[False, True])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "method_predictions.csv.gz"
    fold_path = output_dir / "fold_results.csv"
    summary_path = output_dir / "summary.csv"
    figure_path = output_dir / "ensemble_calibration_comparison.png"
    predictions.to_csv(prediction_path, index=False, compression="gzip")
    folds.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    _plot(summary, figure_path)

    best = summary.iloc[0]
    manifest = {
        "status": "complete",
        "group": group_name,
        "source_experiment": source_name,
        "uses_locked_evaluation": False,
        "selection_rule": "highest mean outer-fold R2, then lowest mean RMSE",
        "best_method": str(best["method"]),
        "best_mean_r2": float(best["mean_r2"]),
        "artifacts": {
            "predictions": prediction_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "fold_results": fold_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "summary": summary_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "figure": figure_path.relative_to(REPOSITORY_ROOT).as_posix(),
        },
    }
    (output_dir / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--group", default="PE3_ensemble_calibration")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = _read_config(args.config.resolve())
    groups = config.get("experiment_groups", {})
    group = groups.get(args.group) if isinstance(groups, dict) else None
    if not isinstance(group, dict):
        parser.error(f"Unknown experiment group {args.group!r}")
    output_dir = args.output_dir or (
        artifact_directory(EXPERIMENTS_DIR, args.group, group) / "reporting"
    )
    try:
        manifest = write_report(config, args.group, output_dir.resolve())
    except (EnsembleReportError, OSError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
