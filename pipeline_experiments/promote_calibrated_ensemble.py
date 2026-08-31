"""Freeze and evaluate a development-selected calibrated tree ensemble."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_config import ExperimentConfigError, read_experiment_config
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "pipeline_experiments.toml"
RUNS_DIR = SCRIPT_DIR / "runs"
LOCKED_RUNNER = (
    REPOSITORY_ROOT
    / "2_model_architecture_study"
    / "6_locked_outer_evaluation"
    / "run_locked_outer_evaluation.py"
)
COMPONENT_FAMILIES = ("extra_trees", "xgboost")
LOCKED_PAIR_KEYS = [
    "seed",
    "outer_fold",
    "scenario",
    "sample_id",
    "uav_id",
    "cutoff",
    "terminal_lifetime",
    "lifetime_quantile",
    "y_true",
]
METHOD_PATTERN = re.compile(r"^blend_xgb_(0\.\d{2}|1\.00)__calibrated$")


class PromotionError(ValueError):
    """Explain why a selected policy cannot enter locked confirmation."""


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PromotionError(f"Cannot read {description} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise PromotionError(f"{description} must contain an object")
    return value


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise PromotionError(f"Artifact is outside the repository: {path}") from error


def _promotion_table(config: dict[str, Any], name: str) -> dict[str, Any]:
    promotions = config.get("promotions", {})
    promotion = promotions.get(name) if isinstance(promotions, dict) else None
    if not isinstance(promotion, dict):
        raise PromotionError(f"Unknown promotion {name!r}")
    return promotion


def _selected_contract(
    config: dict[str, Any],
    promotion_name: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    promotion = _promotion_table(config, promotion_name)
    workflow_name = promotion.get("workflow")
    if not isinstance(workflow_name, str) or not workflow_name:
        raise PromotionError("Promotion workflow must be a non-empty name")
    workflow_dir = RUNS_DIR / workflow_name / "workflow"
    selection_path = workflow_dir / "selection_manifest.json"
    selection = _read_json(selection_path, "workflow selection manifest")
    if selection.get("status") != "complete" or selection.get("uses_locked_evaluation") is not False:
        raise PromotionError("The development workflow is not complete and locked-free")

    selections = selection.get("selections")
    if not isinstance(selections, dict):
        raise PromotionError("Workflow selection manifest has no selections object")
    final = selections.get("final")
    ensemble = selections.get("ensemble_accuracy")
    feature = selections.get("feature")
    target_cap = selections.get("target_cap")
    if not all(isinstance(value, dict) for value in (final, ensemble, feature, target_cap)):
        raise PromotionError("Workflow selection manifest is missing a required decision")
    candidate = str(final.get("candidate", ""))
    method = str(ensemble.get("method", ""))
    if candidate != f"ensemble:{method}":
        raise PromotionError(
            "The final workflow winner is not the selected ensemble accuracy method"
        )
    match = METHOD_PATTERN.fullmatch(method)
    if match is None:
        raise PromotionError(
            "Locked promotion currently requires a calibrated fixed XGBoost/ExtraTrees blend"
        )
    weight = float(match.group(1))
    source_name = ensemble.get("source_experiment")
    if not isinstance(source_name, str) or not source_name:
        raise PromotionError("Selected ensemble has no source experiment")

    resolved_path = workflow_dir / "resolved_catalog.json"
    resolved = _read_json(resolved_path, "resolved workflow catalog")
    experiments = resolved.get("experiments")
    source = experiments.get(source_name) if isinstance(experiments, dict) else None
    if not isinstance(source, dict):
        raise PromotionError(f"Resolved workflow has no source experiment {source_name!r}")
    group_name = promotion.get("ensemble_group")
    groups = resolved.get("experiment_groups")
    group = groups.get(group_name) if isinstance(groups, dict) else None
    if not isinstance(group, dict):
        raise PromotionError(f"Resolved workflow has no ensemble group {group_name!r}")

    degree = int(group.get("calibration_degree", -1))
    alpha = float(group.get("calibration_ridge_alpha", -1.0))
    if degree not in {1, 2} or alpha < 0.0:
        raise PromotionError("Selected calibration settings are invalid")
    if str(source.get("feature_set")) != str(feature.get("feature_set")):
        raise PromotionError("Selected source and feature decisions disagree")
    if str(source.get("target_profile")) != str(target_cap.get("target_profile")):
        raise PromotionError("Selected source and target-cap decisions disagree")

    contract = {
        "contract_version": 1,
        "status": "frozen_before_locked_evaluation",
        "pipeline_run": workflow_name,
        "promotion": promotion_name,
        "source_experiment": source_name,
        "selected_candidate": candidate,
        "method": method,
        "component_families": list(COMPONENT_FAMILIES),
        "xgboost_weight": weight,
        "extra_trees_weight": 1.0 - weight,
        "feature_set": str(source["feature_set"]),
        "target_profile": str(source["target_profile"]),
        "calibration": {
            "type": "polynomial_residual_ridge",
            "features": ["raw_blend", "cutoff"],
            "degree": degree,
            "ridge_alpha": alpha,
            "fit_data": "all selected development OOF predictions",
        },
        "prediction_minimum": 0.0,
        "development_selection": {
            "mean_r2": float(ensemble["mean_r2"]),
            "mean_rmse": float(ensemble["mean_rmse"]),
        },
        "selection_manifest": _repo_relative(selection_path),
        "resolved_catalog": _repo_relative(resolved_path),
        "locked_results_used_for_selection": False,
    }
    return contract, source, selection_path


def _source_paths(source_name: str, source: dict[str, Any]) -> dict[str, Path]:
    pipeline_run = str(source.get("pipeline_run", source_name))
    source_root = RUNS_DIR / pipeline_run
    if pipeline_run != source_name:
        source_root /= source_name
    phase2 = source_root / "phase2"
    return {
        "root": source_root,
        "specification": phase2 / "1_architecture_study_settings" / "artifacts" / "experiment_specification.json",
        "tabular_manifest": phase2 / "2_tabular_data_adapter" / "artifacts" / "tabular_dataset_manifest.json",
        "sequence_manifest": phase2 / "3_sequence_data_adapter" / "artifacts" / "sequence_dataset_manifest.json",
        "trajectory_manifest": phase2 / "3_trajectory_data_adapter" / "artifacts" / "trajectory_dataset_manifest.json",
        "registry": phase2 / "4_model_adapters" / "artifacts" / "model_registry.json",
        "selection_manifest": phase2 / "5_inner_model_selection" / "selection_manifest.json",
        "selected_configurations": phase2 / "5_inner_model_selection" / "selected_configurations.csv",
        "development_predictions": phase2 / "5_inner_model_selection" / "selected_inner_predictions.csv.gz",
    }


def _development_pairs(path: Path, weight: float) -> pd.DataFrame:
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise PromotionError(f"Cannot read selected development predictions: {error}") from error
    keys = [
        "outer_fold", "inner_fold", "validation_row", "uav_id",
        "scenario", "cutoff", "observed_rul",
    ]
    required = {*keys, "model_family", "predicted_rul"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise PromotionError(f"Development predictions are missing {missing}")
    paired = table.pivot(index=keys, columns="model_family", values="predicted_rul").reset_index()
    if any(family not in paired for family in COMPONENT_FAMILIES):
        raise PromotionError("Development predictions need XGBoost and ExtraTrees")
    if paired[list(COMPONENT_FAMILIES)].isna().any().any():
        raise PromotionError("Development component predictions do not align")
    paired["raw_blend"] = weight * paired["xgboost"] + (1.0 - weight) * paired["extra_trees"]
    return paired


def _fit_calibrator(
    paired: pd.DataFrame,
    contract: dict[str, Any],
    model_path: Path,
) -> dict[str, Any]:
    calibration = contract["calibration"]
    model = make_pipeline(
        PolynomialFeatures(degree=int(calibration["degree"]), include_bias=False),
        StandardScaler(),
        Ridge(alpha=float(calibration["ridge_alpha"])),
    )
    residual = paired["raw_blend"].to_numpy(dtype=float) - paired["observed_rul"].to_numpy(dtype=float)
    model.fit(paired[["raw_blend", "cutoff"]], residual)
    correction = model.predict(paired[["raw_blend", "cutoff"]])
    calibrated = np.maximum(paired["raw_blend"].to_numpy(dtype=float) - correction, 0.0)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_name(f".{model_path.name}.{os.getpid()}.tmp")
    joblib.dump(model, temporary)
    os.replace(temporary, model_path)
    return {
        "rows": int(len(paired)),
        "model": _repo_relative(model_path),
        "uncalibrated": _metrics(paired["observed_rul"].to_numpy(dtype=float), paired["raw_blend"].to_numpy(dtype=float)),
        "in_sample_calibrated_diagnostic": _metrics(paired["observed_rul"].to_numpy(dtype=float), calibrated),
        "note": "The diagnostic is in-sample; method selection used the cross-fitted development report.",
    }


def _expected_pairs(paths: dict[str, Path]) -> tuple[list[tuple[str, int]], dict[tuple[str, int], set[str]], int]:
    specification = _read_json(paths["specification"], "source specification")
    registry = _read_json(paths["registry"], "source model registry")
    settings = specification.get("settings")
    if not isinstance(settings, dict):
        raise PromotionError("Source specification has no settings object")
    folds = range(int(settings["phase_1"]["expected_outer_folds"]))
    seeds = [int(value) for value in settings["tuning"]["retraining_seeds"]]
    registry_families = registry.get("families")
    if not isinstance(registry_families, dict):
        raise PromotionError("Source model registry has no families object")
    expected: dict[tuple[str, int], set[str]] = {}
    for family in COMPONENT_FAMILIES:
        details = registry_families.get(family)
        if not isinstance(details, dict) or details.get("enabled") is not True:
            raise PromotionError(f"Selected source does not enable {family}")
        family_seeds = seeds if details.get("stochastic") is True else seeds[:1]
        for fold in folds:
            expected[(family, fold)] = {
                f"{family}__outer_{fold:02d}__seed_{seed:03d}"
                for seed in family_seeds
            }
    return list(expected), expected, int(settings["settings_version"])


def _run_component_locked_evaluation(
    paths: dict[str, Path],
    output_dir: Path,
    *,
    force: bool,
    max_workers: int,
) -> dict[str, Any]:
    pairs, expected, settings_version = _expected_pairs(paths)
    manifest_path = output_dir / "locked_evaluation_manifest.json"
    completed: set[str] = set()
    if manifest_path.is_file() and not force:
        manifest = _read_json(manifest_path, "component locked manifest")
        if manifest.get("settings_version") == settings_version:
            value = manifest.get("completed_runs", [])
            if isinstance(value, list):
                completed = set(map(str, value))
    pending = [pair for pair in pairs if force or not expected[pair].issubset(completed)]
    if pending:
        print(
            f"Locked confirmation: {len(pending)}/{len(pairs)} component family/fold studies pending; "
            f"using up to {min(max_workers, len(pending))} workers",
            flush=True,
        )
    else:
        print("Locked confirmation: all component studies already complete", flush=True)

    environment = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = "1"

    def run_pair(pair: tuple[str, int]) -> None:
        family, fold = pair
        command = [
            sys.executable, str(LOCKED_RUNNER),
            "--specification", str(paths["specification"]),
            "--selection-manifest", str(paths["selection_manifest"]),
            "--selected-configurations", str(paths["selected_configurations"]),
            "--output-dir", str(output_dir),
            "--tabular-manifest", str(paths["tabular_manifest"]),
            "--sequence-manifest", str(paths["sequence_manifest"]),
            "--trajectory-manifest", str(paths["trajectory_manifest"]),
            "--family", family,
            "--outer-fold", str(fold),
        ]
        result = subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=False)
        if result.returncode != 0:
            raise PromotionError(f"Locked component {family} outer fold {fold} failed with exit code {result.returncode}")

    first_error: Exception | None = None
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(pending) or 1))) as executor:
        futures = {executor.submit(run_pair, pair): pair for pair in pending}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:  # preserve already-running atomic checkpoints
                if first_error is None:
                    first_error = error
                for other in futures:
                    other.cancel()
    if first_error is not None:
        raise PromotionError(str(first_error)) from first_error
    manifest = _read_json(manifest_path, "component locked manifest")
    expected_count = sum(len(value) for value in expected.values())
    if manifest.get("status") != "complete" or manifest.get("completed_run_count") != expected_count:
        raise PromotionError(
            "Component locked evaluation is incomplete: "
            f"{manifest.get('completed_run_count', 0)}/{expected_count} runs"
        )
    return manifest


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    positive = np.maximum(residual, 0.0)
    denominator = float(np.sum(np.square(target - np.mean(target))))
    return {
        "r2": 1.0 - float(np.sum(np.square(residual))) / denominator,
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "overprediction_rate": float(np.mean(residual > 0.0)),
        "rms_overprediction": float(np.sqrt(np.mean(np.square(positive)))),
        "overprediction_q95": float(np.quantile(positive, 0.95)),
        "maximum_overprediction": float(np.max(positive)),
    }


def _combine_locked_predictions(
    component_path: Path,
    model_path: Path,
    weight: float,
) -> pd.DataFrame:
    try:
        table = pd.read_csv(component_path)
    except (OSError, pd.errors.ParserError) as error:
        raise PromotionError(f"Cannot read component locked predictions: {error}") from error
    required = {*LOCKED_PAIR_KEYS, "model_family", "y_pred"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise PromotionError(f"Component locked predictions are missing {missing}")
    paired = table.pivot(index=LOCKED_PAIR_KEYS, columns="model_family", values="y_pred").reset_index()
    if any(family not in paired for family in COMPONENT_FAMILIES):
        raise PromotionError("Locked predictions need both component families")
    if paired[list(COMPONENT_FAMILIES)].isna().any().any():
        raise PromotionError("Locked component predictions do not align")
    paired["raw_blend_prediction"] = weight * paired["xgboost"] + (1.0 - weight) * paired["extra_trees"]
    calibrator = joblib.load(model_path)
    paired["calibration_correction"] = calibrator.predict(
        paired[["raw_blend_prediction", "cutoff"]].rename(columns={"raw_blend_prediction": "raw_blend"})
    )
    paired["y_pred"] = np.maximum(
        paired["raw_blend_prediction"] - paired["calibration_correction"],
        0.0,
    )
    paired["residual"] = paired["y_pred"] - paired["y_true"]
    paired.insert(0, "policy", "calibrated_tree_blend")
    return paired


def _write_reporting(
    predictions: pd.DataFrame,
    contract: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "locked_predictions.csv.gz"
    seed_path = output_dir / "locked_metrics_by_seed.csv"
    band_path = output_dir / "locked_metrics_by_rul_band.csv"
    predictions.to_csv(prediction_path, index=False, compression="gzip")

    seed_records = []
    for seed, rows in predictions.groupby("seed", sort=True):
        seed_records.append({"seed": int(seed), "rows": len(rows), **_metrics(rows["y_true"].to_numpy(float), rows["y_pred"].to_numpy(float))})
    seed_metrics = pd.DataFrame.from_records(seed_records)
    seed_metrics.to_csv(seed_path, index=False)
    metric_columns = [column for column in seed_metrics if column not in {"seed", "rows"}]
    mean_metrics = {column: float(seed_metrics[column].mean()) for column in metric_columns}
    sd_metrics = {column: float(seed_metrics[column].std(ddof=1)) for column in metric_columns}

    bins = [-np.inf, 25.0, 50.0, 75.0, 100.0, 125.0, np.inf]
    labels = ["<=25", "26-50", "51-75", "76-100", "101-125", ">125"]
    banded = predictions.copy()
    banded["rul_band"] = pd.cut(banded["y_true"], bins=bins, labels=labels)
    band_records = []
    for band, rows in banded.groupby("rul_band", observed=True, sort=True):
        band_records.append({"rul_band": str(band), "rows": len(rows), **_metrics(rows["y_true"].to_numpy(float), rows["y_pred"].to_numpy(float))})
    bands = pd.DataFrame.from_records(band_records)
    bands.to_csv(band_path, index=False)

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(predictions["y_true"], predictions["y_pred"], s=8, alpha=0.18, color="#2878b5")
    maximum = float(max(predictions["y_true"].max(), predictions["y_pred"].max()))
    axis.plot([0, maximum], [0, maximum], linestyle="--", color="#333333", linewidth=1)
    axis.set(xlabel="Observed RUL", ylabel="Calibrated ensemble RUL", title="PE_run_3 locked prediction alignment")
    figure.tight_layout()
    alignment_path = figure_dir / "locked_prediction_alignment.png"
    figure.savefig(alignment_path, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(seed_metrics["seed"].astype(str), seed_metrics["r2"], color="#2878b5")
    axes[0].axhline(0.9, linestyle="--", color="#c33c35", linewidth=1)
    axes[0].set(xlabel="Retraining seed", ylabel="Locked R2", title="Accuracy stability")
    axes[1].bar(seed_metrics["seed"].astype(str), seed_metrics["rms_overprediction"], color="#d98b2b")
    axes[1].set(xlabel="Retraining seed", ylabel="RMS overprediction", title="Safety stability")
    figure.tight_layout()
    stability_path = figure_dir / "locked_seed_stability.png"
    figure.savefig(stability_path, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(bands["rul_band"], bands["overprediction_rate"], color="#d98b2b")
    axis.axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    axis.set(xlabel="Observed RUL band", ylabel="Overprediction rate", title="Locked safety by RUL band")
    figure.tight_layout()
    safety_path = figure_dir / "locked_safety_by_rul_band.png"
    figure.savefig(safety_path, dpi=180)
    plt.close(figure)

    return {
        "status": "complete",
        "policy": contract["selected_candidate"],
        "uses_locked_evaluation": True,
        "locked_results_used_for_tuning": False,
        "primary_aggregation": "mean metric across retraining seeds",
        "seed_count": int(len(seed_metrics)),
        "prediction_rows": int(len(predictions)),
        "mean_locked_metrics": mean_metrics,
        "sd_locked_metrics": sd_metrics,
        "development_metrics": {
            "mean_r2": float(contract["development_selection"]["mean_r2"]),
            "mean_rmse": float(contract["development_selection"]["mean_rmse"]),
        },
        "artifacts": {
            "predictions": _repo_relative(prediction_path),
            "metrics_by_seed": _repo_relative(seed_path),
            "metrics_by_rul_band": _repo_relative(band_path),
            "figures": [_repo_relative(path) for path in (alignment_path, stability_path, safety_path)],
        },
    }


def run_promotion(
    config: dict[str, Any],
    promotion_name: str,
    *,
    force: bool = False,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Freeze, resume, and report one promoted ensemble policy."""

    contract, source, _ = _selected_contract(config, promotion_name)
    workflow_name = str(contract["pipeline_run"])
    root = RUNS_DIR / workflow_name / promotion_name
    contract_path = root / "promotion_contract.json"
    calibrator_path = root / "frozen_policy" / "residual_calibrator.joblib"
    calibration_summary_path = root / "frozen_policy" / "calibration_fit_summary.json"
    final_manifest_path = root / "locked_confirmation_manifest.json"
    component_dir = root / "locked_evaluation" / "components"
    reporting_dir = root / "reporting"
    paths = _source_paths(str(contract["source_experiment"]), source)

    existing_contract = _read_json(contract_path, "promotion contract") if contract_path.is_file() else None
    component_manifest_path = component_dir / "locked_evaluation_manifest.json"
    if existing_contract is not None and existing_contract != contract:
        locked_loaded = False
        if component_manifest_path.is_file():
            locked_loaded = bool(_read_json(component_manifest_path, "component locked manifest").get("locked_data_loaded"))
        if locked_loaded:
            raise PromotionError("The frozen policy changed after locked data was loaded; create a new experiment instead")
        if not force:
            raise PromotionError("The promotion contract changed; rerun with --force before locked evaluation starts")

    if final_manifest_path.is_file() and not force:
        manifest = _read_json(final_manifest_path, "locked confirmation manifest")
        if manifest.get("status") == "complete" and existing_contract == contract:
            print(f"{promotion_name}: locked confirmation already complete; skipping", flush=True)
            return manifest

    _write_json(contract, contract_path)
    paired = _development_pairs(paths["development_predictions"], float(contract["xgboost_weight"]))
    calibration_summary = _fit_calibrator(paired, contract, calibrator_path)
    _write_json(calibration_summary, calibration_summary_path)
    component_manifest = _run_component_locked_evaluation(
        paths,
        component_dir,
        force=force,
        max_workers=max_workers,
    )
    combined = _combine_locked_predictions(
        component_dir / "locked_predictions.csv.gz",
        calibrator_path,
        float(contract["xgboost_weight"]),
    )
    report = _write_reporting(combined, contract, reporting_dir)
    manifest = {
        **report,
        "promotion": promotion_name,
        "pipeline_run": workflow_name,
        "source_experiment": contract["source_experiment"],
        "contract": _repo_relative(contract_path),
        "calibration_fit_summary": _repo_relative(calibration_summary_path),
        "component_locked_manifest": _repo_relative(component_manifest_path),
        "component_completed_runs": component_manifest["completed_run_count"],
        "phase_3_created": False,
        "next_gate": "Review locked confirmation before creating Phase 3 Run 5",
    }
    _write_json(manifest, final_manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--promotion", default="PE3_final_ensemble")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        config = read_experiment_config(args.config.resolve())
        configured = config.get("execution", {}).get("max_workers", 1)
        if args.max_workers is not None:
            workers = args.max_workers
        elif configured == "auto":
            workers = os.cpu_count() or 1
        else:
            workers = int(configured)
        if workers < 1:
            raise PromotionError("max_workers must be positive")
        run_promotion(config, args.promotion, force=args.force, max_workers=workers)
    except (
        PromotionError,
        ExperimentConfigError,
        OSError,
        ValueError,
        pd.errors.ParserError,
    ) as error:
        print(f"Calibrated ensemble promotion stopped:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
