"""Cross-fit prediction-dependent conservative calibration for one Phase 3 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from experiment_config import ExperimentConfigError, read_experiment_config
from experiment_paths import pipeline_owner, pipeline_run_name, run_directory


SCRIPT_DIR = Path(__file__).resolve().parent
ARCHITECTURE_EXPERIMENTS_ROOT = SCRIPT_DIR.parent
REPOSITORY_ROOT = ARCHITECTURE_EXPERIMENTS_ROOT.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "pipeline_experiments.toml"
REQUIRED_OOF_COLUMNS = {
    "candidate_number",
    "configuration_id",
    "outer_fold",
    "uav_id",
    "scenario",
    "cutoff",
    "observed_rul",
    "predicted_rul",
}
RUL_BAND_EDGES = [-np.inf, 25.0, 50.0, 75.0, 100.0, np.inf]
RUL_BAND_LABELS = ["<=25", "26-50", "51-75", "76-100", ">100"]


class ConditionalCalibrationError(ValueError):
    """Explain an invalid source artifact or calibration configuration."""


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConditionalCalibrationError(
            f"Cannot read {description} at {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ConditionalCalibrationError(f"{description} must be a JSON object")
    return payload


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ConditionalCalibrationError(f"Path is outside the repository: {path}") from error


def load_workflow(config_path: Path, name: str) -> dict[str, Any]:
    """Load and validate one declarative conditional-calibration workflow."""

    try:
        config = read_experiment_config(config_path)
    except ExperimentConfigError as error:
        raise ConditionalCalibrationError(f"Cannot read experiment catalog: {error}") from error
    workflows = config.get("conditional_calibration_workflows", {})
    workflow = workflows.get(name) if isinstance(workflows, dict) else None
    if not isinstance(workflow, dict):
        raise ConditionalCalibrationError(f"Unknown conditional calibration workflow {name!r}")

    run_number = workflow.get("source_phase_3_run")
    if isinstance(run_number, bool) or not isinstance(run_number, int) or run_number <= 0:
        raise ConditionalCalibrationError("source_phase_3_run must be a positive integer")
    quantiles = workflow.get("quantiles")
    if (
        not isinstance(quantiles, list)
        or not quantiles
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.5 <= float(value) < 1.0
            for value in quantiles
        )
    ):
        raise ConditionalCalibrationError("quantiles must be numeric values in [0.5, 1.0)")
    normalized_quantiles = [float(value) for value in quantiles]
    if normalized_quantiles != sorted(set(normalized_quantiles)):
        raise ConditionalCalibrationError("quantiles must be unique and increasing")
    edges = workflow.get("prediction_bin_edges")
    if (
        not isinstance(edges, list)
        or len(edges) < 3
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in edges)
    ):
        raise ConditionalCalibrationError("prediction_bin_edges must be a numeric list")
    normalized_edges = [float(value) for value in edges]
    if normalized_edges != sorted(set(normalized_edges)):
        raise ConditionalCalibrationError("prediction_bin_edges must be unique and increasing")
    if normalized_edges[0] > 0.0:
        raise ConditionalCalibrationError("prediction_bin_edges must begin at or below zero")
    minimum_rows = workflow.get("minimum_bin_rows")
    if isinstance(minimum_rows, bool) or not isinstance(minimum_rows, int) or minimum_rows <= 0:
        raise ConditionalCalibrationError("minimum_bin_rows must be a positive integer")
    tolerance = workflow.get("r2_tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not 0.0 <= float(tolerance) < 1.0
    ):
        raise ConditionalCalibrationError("r2_tolerance must be in [0, 1)")
    return {
        **workflow,
        "source_phase_3_run": run_number,
        "quantiles": normalized_quantiles,
        "prediction_bin_edges": normalized_edges,
        "minimum_bin_rows": minimum_rows,
        "r2_tolerance": float(tolerance),
    }


def fit_correction_curve(
    predictions: NDArray[np.float64],
    residuals: NDArray[np.float64],
    *,
    quantile: float,
    edges: NDArray[np.float64],
    minimum_rows: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Estimate nonnegative residual quantiles in prediction-RUL bins."""

    corrections: list[float] = []
    counts: list[int] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (predictions >= lower) & (predictions < upper)
        values = residuals[in_bin]
        counts.append(int(len(values)))
        correction = float(np.quantile(values, quantile)) if len(values) >= minimum_rows else 0.0
        corrections.append(max(0.0, correction))
    return np.asarray(corrections), np.asarray(counts, dtype=np.int64)


def apply_correction_curve(
    predictions: NDArray[np.float64],
    corrections: NDArray[np.float64],
    edges: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate a smooth subtraction-only correction over predicted RUL."""

    centers = (edges[:-1] + edges[1:]) / 2.0
    adjustment = np.interp(
        predictions,
        centers,
        corrections,
        left=float(corrections[0]),
        right=float(corrections[-1]),
    )
    return np.maximum(predictions - np.maximum(adjustment, 0.0), 0.0)


def _metrics(targets: NDArray[np.float64], predictions: NDArray[np.float64]) -> dict[str, float]:
    residual = predictions - targets
    positive = np.maximum(residual, 0.0)
    denominator = float(np.square(targets - targets.mean()).sum())
    return {
        "r2": 1.0 - float(np.square(residual).sum()) / denominator,
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "overprediction_rate": float(np.mean(residual > 0.0)),
        "rms_overprediction": float(np.sqrt(np.mean(np.square(positive)))),
        "overprediction_q95": float(np.quantile(positive, 0.95)),
        "maximum_overprediction": float(np.max(positive)),
    }


def cross_fit_policy(
    table: pd.DataFrame,
    *,
    quantile: float | None,
    edges: NDArray[np.float64],
    minimum_rows: int,
) -> tuple[NDArray[np.float64], list[dict[str, Any]]]:
    """Apply fold-fitted calibration while keeping each validation fold unseen."""

    calibrated = np.empty(len(table), dtype=np.float64)
    curve_records: list[dict[str, Any]] = []
    for fold in sorted(table["outer_fold"].unique()):
        validation = table["outer_fold"].eq(fold).to_numpy()
        training = ~validation
        training_predictions = table.loc[training, "predicted_rul"].to_numpy(dtype=float)
        training_residuals = (
            table.loc[training, "predicted_rul"].to_numpy(dtype=float)
            - table.loc[training, "observed_rul"].to_numpy(dtype=float)
        )
        if quantile is None:
            corrections = np.zeros(len(edges) - 1, dtype=float)
            counts = np.asarray(
                [
                    np.count_nonzero(
                        (training_predictions >= lower) & (training_predictions < upper)
                    )
                    for lower, upper in zip(edges[:-1], edges[1:], strict=True)
                ],
                dtype=np.int64,
            )
        else:
            corrections, counts = fit_correction_curve(
                training_predictions,
                training_residuals,
                quantile=quantile,
                edges=edges,
                minimum_rows=minimum_rows,
            )
        validation_predictions = table.loc[validation, "predicted_rul"].to_numpy(dtype=float)
        calibrated[np.flatnonzero(validation)] = apply_correction_curve(
            validation_predictions,
            corrections,
            edges,
        )
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            curve_records.append(
                {
                    "outer_fold": int(fold),
                    "quantile": quantile,
                    "bin_lower": float(lower),
                    "bin_upper": float(upper),
                    "training_rows": int(counts[index]),
                    "correction": float(corrections[index]),
                }
            )
    return calibrated, curve_records


def _policy_name(quantile: float | None) -> str:
    return "control" if quantile is None else f"q_{quantile:.2f}"


def _select_policy(summary: pd.DataFrame, tolerance: float) -> dict[str, Any]:
    best_r2 = float(summary["mean_fold_r2"].max())
    eligible = summary.loc[summary["mean_fold_r2"] >= best_r2 - tolerance].copy()
    winner = eligible.sort_values(
        [
            "mean_fold_rms_overprediction",
            "mean_fold_overprediction_rate",
            "mean_fold_rmse",
            "mean_fold_r2",
            "policy",
        ],
        ascending=[True, True, True, False, True],
    ).iloc[0]
    quantile = None if pd.isna(winner["quantile"]) else float(winner["quantile"])
    return {
        "policy": str(winner["policy"]),
        "quantile": quantile,
        "best_observed_mean_fold_r2": best_r2,
        "minimum_eligible_mean_fold_r2": best_r2 - tolerance,
        "r2_tolerance": tolerance,
        "selection_rule": (
            "within the mean-fold R2 tolerance of the best policy, minimize RMS "
            "overprediction; then overprediction rate, RMSE, and maximize R2"
        ),
    }


def _plot_results(
    summary: pd.DataFrame,
    band_metrics: pd.DataFrame,
    curves: pd.DataFrame,
    selected_policy: str,
    output_dir: Path,
) -> list[Path]:
    figure_dir = output_dir / "reporting" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.scatter(summary["mean_fold_rms_overprediction"], summary["mean_fold_r2"], s=58)
    for row in summary.itertuples(index=False):
        offset = (-46, 4) if row.policy == "control" else (5, 4)
        axis.annotate(
            row.policy,
            (row.mean_fold_rms_overprediction, row.mean_fold_r2),
            xytext=offset,
            textcoords="offset points",
        )
    axis.set(xlabel="Mean-fold RMS overprediction", ylabel="Mean-fold R2", title="Conditional safety calibration trade-off")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = figure_dir / "calibration_tradeoff.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    labels = summary["policy"].tolist()
    axes[0].plot(labels, summary["mean_fold_r2"], marker="o")
    axes[0].set_ylabel("Mean-fold R2")
    axes[1].plot(labels, summary["mean_fold_overprediction_rate"], marker="o")
    axes[1].set_ylabel("Overprediction rate")
    axes[2].plot(labels, summary["mean_fold_rms_overprediction"], marker="o")
    axes[2].set_ylabel("RMS overprediction")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(alpha=0.25)
    figure.suptitle("Accuracy and safety across calibration quantiles")
    figure.tight_layout()
    path = figure_dir / "calibration_policy_metrics.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    selected_bands = band_metrics.loc[
        band_metrics["policy"].isin(["control", selected_policy])
    ].copy()
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for policy, rows in selected_bands.groupby("policy", sort=False):
        rows = rows.set_index("rul_band").reindex(RUL_BAND_LABELS).reset_index()
        axes[0].plot(rows["rul_band"], rows["rms_overprediction"], marker="o", label=policy)
        axes[1].plot(rows["rul_band"], rows["bias"], marker="o", label=policy)
    axes[0].set_ylabel("RMS overprediction")
    axes[1].set_ylabel("Signed bias")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Selected calibration by observed RUL band")
    figure.tight_layout()
    path = figure_dir / "selected_safety_by_rul_band.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    full_curves = curves.loc[curves["outer_fold"].eq("all")].copy()
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for policy, rows in full_curves.groupby("policy", sort=False):
        centers = (rows["bin_lower"].to_numpy() + rows["bin_upper"].to_numpy()) / 2.0
        axis.plot(centers, rows["correction"], marker="o", label=policy)
    axis.set(xlabel="Predicted RUL", ylabel="Subtracted RUL", title="Full-development conditional correction curves")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = figure_dir / "conditional_correction_curves.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    return paths


def run_workflow(name: str, config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Run all declared quantiles, select one, and create its submission."""

    workflow = load_workflow(config_path, name)
    source_run = int(workflow["source_phase_3_run"])
    source_root = REPOSITORY_ROOT / "3_final_model_training_and_inference" / "runs" / f"run_{source_run}"
    search_root = source_root / "2_final_configuration_search"
    manifest = _read_json(search_root / "final_search_manifest.json", "final search manifest")
    selected = _read_json(search_root / "artifacts" / "selected_configuration.json", "selected configuration")
    if manifest.get("status") != "complete" or manifest.get("locked_data_loaded") or manifest.get("test_data_loaded"):
        raise ConditionalCalibrationError("Source final search is incomplete or used locked/test data")
    if selected.get("locked_data_loaded") or selected.get("test_data_loaded"):
        raise ConditionalCalibrationError("Selected configuration used locked/test data")
    if manifest.get("phase_3_run_number") != source_run or selected.get("phase_3_run_number") != source_run:
        raise ConditionalCalibrationError("Source artifacts identify another Phase 3 run")

    oof_path = search_root / "artifacts" / "final_search_oof_predictions.csv"
    try:
        oof = pd.read_csv(oof_path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise ConditionalCalibrationError(f"Cannot read source OOF predictions: {error}") from error
    missing = sorted(REQUIRED_OOF_COLUMNS - set(oof.columns))
    if missing:
        raise ConditionalCalibrationError(f"Source OOF predictions are missing {missing}")
    candidate_number = int(selected["candidate_number"])
    configuration_id = str(selected["configuration_id"])
    oof = oof.loc[
        oof["candidate_number"].eq(candidate_number)
        & oof["configuration_id"].astype(str).eq(configuration_id)
    ].copy()
    if oof.empty or oof["outer_fold"].nunique() < 2:
        raise ConditionalCalibrationError("Selected candidate has incomplete OOF predictions")
    numeric_columns = ["cutoff", "observed_rul", "predicted_rul"]
    if not np.isfinite(oof[numeric_columns].to_numpy(dtype=float)).all():
        raise ConditionalCalibrationError("Selected OOF predictions contain non-finite values")
    oof = oof.sort_values(["outer_fold", "uav_id", "scenario", "cutoff"]).reset_index(drop=True)

    edges = np.asarray(workflow["prediction_bin_edges"], dtype=float)
    policies: list[float | None] = [None, *workflow["quantiles"]]
    prediction_tables: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    band_records: list[dict[str, Any]] = []
    curve_records: list[dict[str, Any]] = []
    targets = oof["observed_rul"].to_numpy(dtype=float)

    for quantile in policies:
        policy = _policy_name(quantile)
        predictions, policy_curves = cross_fit_policy(
            oof,
            quantile=quantile,
            edges=edges,
            minimum_rows=int(workflow["minimum_bin_rows"]),
        )
        for record in policy_curves:
            record["policy"] = policy
        curve_records.extend(policy_curves)
        prediction_table = oof[
            ["outer_fold", "uav_id", "scenario", "cutoff", "observed_rul", "predicted_rul"]
        ].copy()
        prediction_table["policy"] = policy
        prediction_table["quantile"] = quantile
        prediction_table["calibrated_rul"] = predictions
        prediction_table["residual"] = predictions - targets
        prediction_tables.append(prediction_table)

        policy_fold_records: list[dict[str, Any]] = []
        for fold, rows in prediction_table.groupby("outer_fold", sort=True):
            metrics = _metrics(
                rows["observed_rul"].to_numpy(dtype=float),
                rows["calibrated_rul"].to_numpy(dtype=float),
            )
            record = {"policy": policy, "quantile": quantile, "outer_fold": int(fold), **metrics}
            fold_records.append(record)
            policy_fold_records.append(record)
        fold_table = pd.DataFrame.from_records(policy_fold_records)
        pooled = _metrics(targets, predictions)
        summary_records.append(
            {
                "policy": policy,
                "quantile": quantile,
                "rows": len(oof),
                "outer_folds": oof["outer_fold"].nunique(),
                **{f"mean_fold_{key}": float(fold_table[key].mean()) for key in pooled},
                **{f"pooled_{key}": value for key, value in pooled.items()},
            }
        )
        bands = pd.cut(targets, RUL_BAND_EDGES, labels=RUL_BAND_LABELS)
        for band in RUL_BAND_LABELS:
            mask = np.asarray(bands == band)
            metrics = _metrics(targets[mask], predictions[mask])
            band_records.append(
                {"policy": policy, "quantile": quantile, "rul_band": band, "rows": int(mask.sum()), **metrics}
            )

    summary = pd.DataFrame.from_records(summary_records)
    selection = _select_policy(summary, float(workflow["r2_tolerance"]))
    selected_quantile = selection["quantile"]
    full_predictions = oof["predicted_rul"].to_numpy(dtype=float)
    full_residuals = full_predictions - targets
    for quantile in policies:
        policy = _policy_name(quantile)
        if quantile is None:
            corrections = np.zeros(len(edges) - 1, dtype=float)
            counts = np.asarray(
                [
                    np.count_nonzero((full_predictions >= lower) & (full_predictions < upper))
                    for lower, upper in zip(edges[:-1], edges[1:], strict=True)
                ]
            )
        else:
            corrections, counts = fit_correction_curve(
                full_predictions,
                full_residuals,
                quantile=quantile,
                edges=edges,
                minimum_rows=int(workflow["minimum_bin_rows"]),
            )
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            curve_records.append(
                {
                    "policy": policy,
                    "outer_fold": "all",
                    "quantile": quantile,
                    "bin_lower": float(lower),
                    "bin_upper": float(upper),
                    "training_rows": int(counts[index]),
                    "correction": float(corrections[index]),
                }
            )

    output_dir = run_directory(SCRIPT_DIR / "experiments", name, workflow)
    reporting_dir = output_dir / "reporting"
    artifact_dir = output_dir / "artifacts"
    reporting_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics = pd.DataFrame.from_records(fold_records)
    band_metrics = pd.DataFrame.from_records(band_records)
    curves = pd.DataFrame.from_records(curve_records)
    summary.to_csv(reporting_dir / "calibration_summary.csv", index=False)
    fold_metrics.to_csv(reporting_dir / "fold_metrics.csv", index=False)
    band_metrics.to_csv(reporting_dir / "metrics_by_rul_band.csv", index=False)
    curves.to_csv(reporting_dir / "correction_curves.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        artifact_dir / "cross_fitted_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    selected_policy = str(selection["policy"])
    selected_curve = curves.loc[
        curves["outer_fold"].eq("all") & curves["policy"].eq(selected_policy)
    ].sort_values("bin_lower")
    selected_calibrator = {
        "calibrator_version": 1,
        "pipeline_experiment": pipeline_owner(name, workflow)[0],
        "pipeline_run": pipeline_run_name(name, workflow),
        "source_phase_3_run": source_run,
        "source_configuration_id": configuration_id,
        "selection": selection,
        "prediction_bin_edges": workflow["prediction_bin_edges"],
        "minimum_bin_rows": workflow["minimum_bin_rows"],
        "corrections": selected_curve["correction"].astype(float).tolist(),
        "training_rows_by_bin": selected_curve["training_rows"].astype(int).tolist(),
        "correction_is_nonnegative": True,
        "calibration_can_increase_prediction": False,
    }
    _write_json(selected_calibrator, artifact_dir / "selected_calibrator.json")

    source_submission_path = source_root / "6_submission_verification" / "artifacts" / "submission.csv"
    source_submission_manifest = _read_json(
        source_root / "6_submission_verification" / "submission_manifest.json",
        "source submission manifest",
    )
    if source_submission_manifest.get("status") != "complete" or not source_submission_manifest.get("regenerated_prediction_equivalence"):
        raise ConditionalCalibrationError("Source Phase 3 submission is not verified")
    try:
        submission = pd.read_csv(source_submission_path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise ConditionalCalibrationError(f"Cannot read source submission: {error}") from error
    if list(submission.columns) != ["id", "RUL"] or submission["id"].duplicated().any():
        raise ConditionalCalibrationError("Source submission has an invalid schema")
    source_test_predictions = submission["RUL"].to_numpy(dtype=float)
    calibrated_test_predictions = apply_correction_curve(
        source_test_predictions,
        np.asarray(selected_calibrator["corrections"], dtype=float),
        edges,
    )
    if not np.isfinite(calibrated_test_predictions).all() or np.any(
        calibrated_test_predictions < 0.0
    ):
        raise ConditionalCalibrationError(
            "Conditional calibrator produced invalid submission predictions"
        )
    submission["RUL"] = calibrated_test_predictions
    submission.to_csv(artifact_dir / "submission.csv", index=False)
    submission_verification = {
        "manifest_version": 1,
        "status": "complete",
        "pipeline_experiment": pipeline_owner(name, workflow)[0],
        "pipeline_run": pipeline_run_name(name, workflow),
        "source_phase_3_run": source_run,
        "selected_policy": selected_policy,
        "rows": len(submission),
        "columns": ["id", "RUL"],
        "source_identifier_order_preserved": True,
        "identifier_uniqueness_verified": True,
        "finite_nonnegative_values_verified": True,
        "test_targets_loaded": False,
    }
    _write_json(
        submission_verification,
        artifact_dir / "submission_manifest.json",
    )

    figure_paths = _plot_results(summary, band_metrics, curves, selected_policy, output_dir)
    result = {
        "status": "complete",
        "pipeline_experiment": pipeline_owner(name, workflow)[0],
        "pipeline_run": pipeline_run_name(name, workflow),
        "experiment": "conditional conservative calibration",
        "source_phase_3_run": source_run,
        "source_configuration_id": configuration_id,
        "uses_locked_evaluation": False,
        "test_targets_loaded": False,
        "source_submission_predictions_loaded": True,
        "selection": selection,
        "quantiles": workflow["quantiles"],
        "artifacts": {
            "summary": _repo_relative(reporting_dir / "calibration_summary.csv"),
            "fold_metrics": _repo_relative(reporting_dir / "fold_metrics.csv"),
            "metrics_by_rul_band": _repo_relative(reporting_dir / "metrics_by_rul_band.csv"),
            "correction_curves": _repo_relative(reporting_dir / "correction_curves.csv"),
            "cross_fitted_predictions": _repo_relative(artifact_dir / "cross_fitted_predictions.csv.gz"),
            "selected_calibrator": _repo_relative(artifact_dir / "selected_calibrator.json"),
            "submission": _repo_relative(artifact_dir / "submission.csv"),
            "submission_manifest": _repo_relative(
                artifact_dir / "submission_manifest.json"
            ),
            "figures": [_repo_relative(path) for path in figure_paths],
        },
    }
    _write_json(result, output_dir / "conditional_calibration_manifest.json")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    try:
        run_workflow(args.run, args.config.resolve())
    except ConditionalCalibrationError as error:
        print(f"Conditional safety calibration failed:\n{error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
