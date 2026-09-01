"""Retrain one fixed tree blend under four target policies and build submissions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
import sys
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
PHASE_2_DIR = SCRIPT_DIR.parent / "2_model_architecture_study"
for dependency in (
    SCRIPT_DIR,
    PHASE_2_DIR,
    PHASE_2_DIR / "2_tabular_data_adapter",
    PHASE_2_DIR / "4_model_adapters",
):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from experiment_config import read_experiment_config  # noqa: E402
from experiment_paths import repository_path, run_directory  # noqa: E402
from models.tabular.extra_trees import ExtraTreesAdapter  # noqa: E402
from models.tabular.xgboost import XGBoostAdapter  # noqa: E402
from policies import (  # noqa: E402
    ConditionalQuantileCalibrator,
    PredictionPolicy,
    TargetPolicy,
    cross_fit_conditional_calibration,
)
from tabular_data_adapter import TabularDataAdapter, TabularDataset  # noqa: E402
from tensorboard_monitoring import (  # noqa: E402
    TrainingRunContext,
    create_study_monitor,
)


EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
REQUIRED_VARIANTS = ("hard_cap_125", "raw", "weighted_raw", "soft_tail")


class TargetSubmissionError(ValueError):
    """Explain an invalid target-policy experiment or generated artifact."""


@dataclass(frozen=True)
class Variant:
    name: str
    target_policy: TargetPolicy
    high_rul_weight: float | None
    high_rul_threshold: float | None
    settings: dict[str, Any]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TargetSubmissionError(f"Cannot read {label} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise TargetSubmissionError(f"{label} must contain a JSON object")
    return value


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise TargetSubmissionError(f"Artifact escapes repository: {path}") from error


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TargetSubmissionError(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise TargetSubmissionError(f"{label} must be finite and positive")
    return result


def _variant(name: str, value: Any) -> Variant:
    if not isinstance(value, dict):
        raise TargetSubmissionError(f"Variant {name!r} must be a TOML table")
    mode = value.get("target_mode")
    if name == "hard_cap_125":
        if mode != "piecewise_cap":
            raise TargetSubmissionError("hard_cap_125 must use piecewise_cap")
        maximum = _positive_float(value.get("maximum_rul"), "maximum_rul")
        target = TargetPolicy.from_settings(
            {"mode": "piecewise_cap", "maximum_rul": maximum}
        )
        return Variant(name, target, None, None, dict(value))
    if name == "raw":
        if mode != "raw":
            raise TargetSubmissionError("raw must use raw target mode")
        return Variant(name, TargetPolicy.from_settings({"mode": "raw"}), None, None, dict(value))
    if name == "weighted_raw":
        if mode != "raw":
            raise TargetSubmissionError("weighted_raw must retain raw targets")
        threshold = _positive_float(value.get("tail_threshold"), "tail_threshold")
        weight = _positive_float(value.get("high_rul_weight"), "high_rul_weight")
        if weight >= 1.0:
            raise TargetSubmissionError("high_rul_weight must be below one")
        return Variant(
            name,
            TargetPolicy.from_settings({"mode": "raw"}),
            weight,
            threshold,
            dict(value),
        )
    if name == "soft_tail":
        if mode != "soft_tail":
            raise TargetSubmissionError("soft_tail must use soft_tail target mode")
        threshold = _positive_float(value.get("tail_threshold"), "tail_threshold")
        scale = _positive_float(value.get("tail_scale"), "tail_scale")
        target = TargetPolicy.from_settings(
            {
                "mode": "soft_tail",
                "tail_threshold": threshold,
                "tail_scale": scale,
            }
        )
        return Variant(name, target, None, None, dict(value))
    raise TargetSubmissionError(f"Unknown target variant {name!r}")


def _workflow(config: dict[str, Any], name: str) -> tuple[dict[str, Any], list[Variant]]:
    workflows = config.get("target_submission_workflows", {})
    workflow = workflows.get(name) if isinstance(workflows, dict) else None
    if not isinstance(workflow, dict):
        raise TargetSubmissionError(f"Unknown target submission workflow {name!r}")
    variants = workflow.get("variants")
    if not isinstance(variants, dict) or set(variants) != set(REQUIRED_VARIANTS):
        raise TargetSubmissionError(
            f"Workflow must define exactly these variants: {list(REQUIRED_VARIANTS)}"
        )
    return workflow, [_variant(item, variants[item]) for item in REQUIRED_VARIANTS]


def _prediction_policy(workflow: dict[str, Any]) -> PredictionPolicy:
    return PredictionPolicy.from_settings(
        {
            "loss": "symmetric_rmse",
            "overprediction_weight": 1.0,
            "quantile": 0.5,
            "severity_scale": 10.0,
            "calibration": "conditional_quantile",
            "safety_offset": 0.0,
            "non_overprediction_coverage": workflow.get("calibration_quantile"),
            "calibration_prediction_bin_edges": workflow.get(
                "calibration_prediction_bin_edges"
            ),
            "calibration_minimum_bin_rows": workflow.get(
                "calibration_minimum_bin_rows"
            ),
        }
    )


def _neutral_prediction_policy() -> PredictionPolicy:
    return PredictionPolicy.from_settings(
        {
            "loss": "symmetric_rmse",
            "overprediction_weight": 1.0,
            "quantile": 0.5,
            "severity_scale": 10.0,
            "calibration": "none",
            "safety_offset": 0.0,
            "non_overprediction_coverage": 0.5,
        }
    )


def _weighted_training_data(dataset: TabularDataset, variant: Variant) -> TabularDataset:
    if variant.high_rul_weight is None:
        return dataset
    if dataset.target is None or dataset.sample_weights is None:
        raise TargetSubmissionError("Weighted raw training data need targets and weights")
    assert variant.high_rul_threshold is not None
    weights = dataset.sample_weights.to_numpy(dtype=np.float64).copy()
    targets = dataset.target.to_numpy(dtype=np.float64)
    weights[targets > variant.high_rul_threshold] *= variant.high_rul_weight
    uav_ids = dataset.metadata["uav_id"].astype(str).reset_index(drop=True)
    weight_series = pd.Series(weights)
    totals = weight_series.groupby(uav_ids).transform("sum")
    if np.any(totals.to_numpy(dtype=np.float64) <= 0.0):
        raise TargetSubmissionError("Weighted raw policy produced an empty UAV weight")
    weights = weights / totals.to_numpy(dtype=np.float64)
    normalized = pd.Series(weights, name=dataset.sample_weights.name)
    sums = normalized.groupby(uav_ids).sum().to_numpy(dtype=np.float64)
    if not np.allclose(sums, np.ones_like(sums), rtol=0.0, atol=1e-12):
        raise TargetSubmissionError("Weighted raw policy broke equal UAV weighting")
    return TabularDataset(
        features=dataset.features,
        metadata=dataset.metadata,
        target=dataset.target,
        sample_weights=normalized,
    )


def _fit_blend(
    *,
    training: TabularDataset,
    prediction_data: TabularDataset,
    variant: Variant,
    extra_configuration: dict[str, Any],
    xgboost_configuration: dict[str, Any],
    blend_weight: float,
    seed: int,
    fold: int,
    log_root: Path,
) -> tuple[np.ndarray, ExtraTreesAdapter, XGBoostAdapter, dict[str, Any]]:
    training = _weighted_training_data(training, variant)
    neutral = _neutral_prediction_policy()
    extra = ExtraTreesAdapter(
        hyperparameters=dict(extra_configuration["hyperparameters"]),
        seed=seed,
        prediction_minimum=0.0,
    )
    extra.configure_policies(variant.target_policy, neutral)
    extra_summary = extra.fit(training, None)

    monitor_family = f"{variant.name}__xgboost"
    with create_study_monitor(
        stage="step_6",
        model_family=monitor_family,
        outer_fold=fold,
        log_root=log_root,
    ) as study_monitor:
        context = TrainingRunContext(
            stage="step_6",
            model_family=monitor_family,
            representation="tabular",
            outer_fold=fold,
            seed=seed,
            configuration_id=f"{variant.name}__fold_{fold:02d}",
            feature_set=str(extra_configuration.get("feature_set", "screened_drift_pruned")),
        )
        with study_monitor.fit(context) as fit_monitor:
            xgboost = XGBoostAdapter(
                hyperparameters=dict(xgboost_configuration["hyperparameters"]),
                seed=seed,
                prediction_minimum=0.0,
                early_stopping_patience=None,
                training_iterations=int(xgboost_configuration["training_iterations"]),
                training_monitor=fit_monitor,
            )
            xgboost.configure_policies(variant.target_policy, neutral)
            xgboost_summary = xgboost.fit(training, None)
            xgboost.detach_training_monitor()

    extra_prediction = extra.predict(prediction_data)
    xgboost_prediction = xgboost.predict(prediction_data)
    blend = blend_weight * xgboost_prediction + (1.0 - blend_weight) * extra_prediction
    if not np.isfinite(blend).all() or np.any(blend < 0.0):
        raise TargetSubmissionError(f"{variant.name} produced invalid blend predictions")
    return (
        blend,
        extra,
        xgboost,
        {
            "extra_trees_training_seconds": extra_summary.training_seconds,
            "xgboost_training_seconds": xgboost_summary.training_seconds,
        },
    )


def _metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    residual = predictions - targets
    denominator = float(np.square(targets - targets.mean()).sum())
    positive = np.maximum(residual, 0.0)
    return {
        "r2": 1.0 - float(np.square(residual).sum()) / denominator,
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "overprediction_rate": float(np.mean(residual > 0.0)),
        "rms_overprediction": float(np.sqrt(np.mean(np.square(positive)))),
    }


def _fold_metrics(table: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fold, rows in table.groupby("outer_fold", sort=True):
        targets = rows["observed_rul"].to_numpy(dtype=np.float64)
        for column, label in (
            ("uncalibrated_prediction", "uncalibrated"),
            ("predicted_rul", "conditional_q55"),
        ):
            records.append(
                {
                    "outer_fold": int(fold),
                    "prediction_policy": label,
                    "rows": len(rows),
                    **_metrics(targets, rows[column].to_numpy(dtype=np.float64)),
                }
            )
    return pd.DataFrame.from_records(records)


def _summary(variant: Variant, folds: pd.DataFrame, training: TabularDataset) -> dict[str, Any]:
    calibrated = folds.loc[folds["prediction_policy"] == "conditional_q55"]
    targets = training.target.to_numpy(dtype=np.float64) if training.target is not None else np.asarray([])
    threshold = float(variant.high_rul_threshold or variant.target_policy.tail_threshold or variant.target_policy.maximum_rul or 125.0)
    return {
        "variant": variant.name,
        "target_mode": variant.target_policy.mode,
        "mean_r2": float(calibrated["r2"].mean()),
        "mean_rmse": float(calibrated["rmse"].mean()),
        "mean_mae": float(calibrated["mae"].mean()),
        "mean_bias": float(calibrated["bias"].mean()),
        "mean_overprediction_rate": float(calibrated["overprediction_rate"].mean()),
        "mean_rms_overprediction": float(calibrated["rms_overprediction"].mean()),
        "training_rows_above_threshold": int(np.sum(targets > threshold)),
        "training_fraction_above_threshold": float(np.mean(targets > threshold)),
    }


def _oof_job(
    *,
    adapter: TabularDataAdapter,
    feature_set: str,
    variant: Variant,
    fold: int,
    extra_configuration: dict[str, Any],
    xgboost_configuration: dict[str, Any],
    blend_weight: float,
    seed: int,
    log_root: Path,
) -> tuple[str, int, pd.DataFrame, dict[str, Any]]:
    split = adapter.get_final_search_split(fold, feature_set)
    started = perf_counter()
    prediction, _, _, timings = _fit_blend(
        training=split.training,
        prediction_data=split.validation,
        variant=variant,
        extra_configuration=extra_configuration,
        xgboost_configuration=xgboost_configuration,
        blend_weight=blend_weight,
        seed=seed,
        fold=fold,
        log_root=log_root,
    )
    if split.validation.target is None:
        raise TargetSubmissionError("Development validation has no RUL target")
    table = split.validation.metadata.copy().reset_index(drop=True)
    table["outer_fold"] = fold
    table["observed_rul"] = split.validation.target.to_numpy(dtype=np.float64)
    table["uncalibrated_prediction"] = prediction
    return variant.name, fold, table, {**timings, "wall_seconds": perf_counter() - started}


def _fit_final_variant(
    *,
    adapter: TabularDataAdapter,
    feature_set: str,
    variant: Variant,
    oof: pd.DataFrame,
    q_policy: PredictionPolicy,
    extra_configuration: dict[str, Any],
    xgboost_configuration: dict[str, Any],
    blend_weight: float,
    seed: int,
    output_dir: Path,
    log_root: Path,
) -> dict[str, Any]:
    training = adapter.load_training(feature_set)
    test = adapter.load_test(feature_set)
    if test.target is not None or test.sample_weights is not None:
        raise TargetSubmissionError("Test data unexpectedly contain targets or weights")
    raw_prediction, extra, xgboost, timings = _fit_blend(
        training=training,
        prediction_data=test,
        variant=variant,
        extra_configuration=extra_configuration,
        xgboost_configuration=xgboost_configuration,
        blend_weight=blend_weight,
        seed=seed,
        fold=5,
        log_root=log_root,
    )
    calibrator = ConditionalQuantileCalibrator.fit(
        oof["observed_rul"].to_numpy(dtype=np.float64),
        oof["uncalibrated_prediction"].to_numpy(dtype=np.float64),
        q_policy,
    )
    prediction = calibrator.apply(raw_prediction, prediction_minimum=0.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "model_bundle.joblib"
    extra.detach_training_monitor()
    xgboost.detach_training_monitor()
    joblib.dump(
        {
            "bundle_version": 1,
            "variant": variant.name,
            "blend_weight": blend_weight,
            "extra_trees": extra,
            "xgboost": xgboost,
            "conditional_calibrator": calibrator.to_dict(),
        },
        bundle_path,
    )
    reloaded = joblib.load(bundle_path)
    regenerated_raw = (
        float(reloaded["blend_weight"]) * reloaded["xgboost"].predict(test)
        + (1.0 - float(reloaded["blend_weight"]))
        * reloaded["extra_trees"].predict(test)
    )
    regenerated = ConditionalQuantileCalibrator.from_dict(
        reloaded["conditional_calibrator"]
    ).apply(regenerated_raw, prediction_minimum=0.0)
    if not np.allclose(prediction, regenerated, rtol=1e-10, atol=1e-10):
        raise TargetSubmissionError(f"{variant.name} model reload changed predictions")

    prediction_table = test.metadata.copy().reset_index(drop=True)
    prediction_table["uncalibrated_prediction"] = raw_prediction
    prediction_table["calibration_adjustment"] = raw_prediction - prediction
    prediction_table["RUL"] = prediction
    prediction_table = prediction_table.sort_values("uav_id", kind="stable").reset_index(drop=True)
    prediction_path = output_dir / "test_predictions.csv"
    prediction_table.to_csv(prediction_path, index=False)
    submission = prediction_table.loc[:, ["uav_id", "RUL"]].rename(columns={"uav_id": "id"})
    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    reread = pd.read_csv(submission_path)
    if (
        list(reread.columns) != ["id", "RUL"]
        or len(reread) != 100
        or reread["id"].astype(str).duplicated().any()
        or not np.isfinite(reread["RUL"].to_numpy(dtype=np.float64)).all()
        or np.any(reread["RUL"].to_numpy(dtype=np.float64) < 0.0)
    ):
        raise TargetSubmissionError(f"{variant.name} submission verification failed")

    _write_json(calibrator.to_dict(), output_dir / "conditional_calibrator.json")
    manifest = {
        "status": "complete",
        "variant": variant.name,
        "target_policy": variant.target_policy.to_dict(),
        "weight_policy": (
            {
                "tail_threshold": variant.high_rul_threshold,
                "high_rul_weight": variant.high_rul_weight,
                "renormalized_to_equal_total_weight_per_uav": True,
            }
            if variant.high_rul_weight is not None
            else None
        ),
        "blend_weight_xgboost": blend_weight,
        "calibration": calibrator.to_dict(),
        "rows": len(reread),
        "prediction_minimum": float(reread["RUL"].min()),
        "prediction_maximum": float(reread["RUL"].max()),
        "prediction_mean": float(reread["RUL"].mean()),
        "reload_prediction_equivalence": True,
        "test_targets_loaded": False,
        "training_seconds": timings,
        "artifacts": {
            "model_bundle": _repo_relative(bundle_path),
            "test_predictions": _repo_relative(prediction_path),
            "submission": _repo_relative(submission_path),
            "conditional_calibrator": _repo_relative(output_dir / "conditional_calibrator.json"),
        },
    }
    _write_json(manifest, output_dir / "manifest.json")
    return manifest


def _plots(summary: pd.DataFrame, submissions: dict[str, pd.DataFrame], figure_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    ordered = summary.set_index("variant").loc[list(REQUIRED_VARIANTS)].reset_index()
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].bar(ordered["variant"], ordered["mean_r2"], color="#2878b5")
    axes[0].set_ylabel("Mean development R2")
    axes[1].bar(ordered["variant"], ordered["mean_overprediction_rate"], color="#d98b2b")
    axes[1].set_ylabel("Overprediction rate")
    axes[1].tick_params(axis="x", rotation=20)
    figure.suptitle("Target-policy development comparison")
    figure.tight_layout()
    path = figure_dir / "target_policy_development_metrics.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(11, 6))
    for name in REQUIRED_VARIANTS:
        values = submissions[name]["RUL"].to_numpy(dtype=np.float64)
        axis.plot(np.sort(values), np.linspace(0.01, 1.0, len(values)), label=name)
    axis.set_xlabel("Submitted RUL")
    axis.set_ylabel("Empirical cumulative probability")
    axis.set_title("Submission prediction distributions")
    axis.legend()
    figure.tight_layout()
    path = figure_dir / "submission_prediction_distributions.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)
    return paths


def run(config_path: Path, name: str, *, force: bool = False) -> dict[str, Any]:
    config = read_experiment_config(config_path)
    workflow, variants = _workflow(config, name)
    source_run = int(workflow["source_phase_3_run"])
    source_root = REPOSITORY_ROOT / "3_final_model_training_and_inference" / "runs" / f"run_{source_run}"
    contract_path = source_root / "3_final_training_contract" / "artifacts" / "final_training_contract.json"
    selection_path = source_root / "2_final_configuration_search" / "artifacts" / "selected_configuration.json"
    component_path = source_root / "1_winning_architecture_selection" / "artifacts" / "promoted_component_configurations.json"
    contract = _read_json(contract_path, "source training contract")
    selection = _read_json(selection_path, "source selected configuration")
    components = _read_json(component_path, "source component configurations")
    if contract.get("model_family") != "calibrated_tree_blend":
        raise TargetSubmissionError("Source Phase 3 run is not the calibrated tree blend")
    feature_set = str(workflow.get("feature_set", contract["input_schema"]["feature_set"]))
    if feature_set != str(contract["input_schema"]["feature_set"]):
        raise TargetSubmissionError("Configured feature set differs from the source contract")
    tabular_manifest = repository_path(
        REPOSITORY_ROOT,
        str(contract["data_manifests"]["tabular"]),
    )
    if not tabular_manifest.is_file():
        raise TargetSubmissionError(f"Source tabular manifest is missing: {tabular_manifest}")
    adapter = TabularDataAdapter(tabular_manifest)
    folds = adapter.outer_fold_labels()
    if folds != (0, 1, 2, 3, 4):
        raise TargetSubmissionError(f"Expected five outer folds, found {folds}")
    extra_index = int(selection["hyperparameters"]["extra_trees_configuration_index"])
    xgboost_index = int(selection["hyperparameters"]["xgboost_configuration_index"])
    extra_configuration = components["extra_trees"][extra_index]
    xgboost_configuration = components["xgboost"][xgboost_index]
    blend_weight = float(workflow.get("xgboost_weight", selection["hyperparameters"]["xgboost_weight"]))
    if not 0.0 < blend_weight < 1.0:
        raise TargetSubmissionError("xgboost_weight must be in (0, 1)")
    seed = int(workflow.get("model_seed", selection["model_seed"]))
    workers = int(config.get("execution", {}).get("max_workers", 4))
    if workers <= 0:
        raise TargetSubmissionError("execution.max_workers must be positive")
    q_policy = _prediction_policy(workflow)
    run_root = run_directory(EXPERIMENTS_DIR, name, workflow)
    run_root.mkdir(parents=True, exist_ok=True)
    log_root = run_root / "tensorboard_logs"

    if not force and (run_root / "target_submission_manifest.json").is_file():
        return _read_json(run_root / "target_submission_manifest.json", "target submission manifest")

    oof_by_variant: dict[str, list[pd.DataFrame]] = {item.name: [] for item in variants}
    timing_records: list[dict[str, Any]] = []
    jobs = []
    with ThreadPoolExecutor(max_workers=min(workers, len(variants) * len(folds))) as executor:
        for variant in variants:
            for fold in folds:
                jobs.append(
                    executor.submit(
                        _oof_job,
                        adapter=adapter,
                        feature_set=feature_set,
                        variant=variant,
                        fold=fold,
                        extra_configuration=extra_configuration,
                        xgboost_configuration=xgboost_configuration,
                        blend_weight=blend_weight,
                        seed=seed,
                        log_root=log_root,
                    )
                )
        for future in as_completed(jobs):
            variant_name, fold, table, timings = future.result()
            oof_by_variant[variant_name].append(table)
            timing_records.append({"variant": variant_name, "outer_fold": fold, **timings})
            print(f"{name}: completed OOF {variant_name}, fold {fold}", flush=True)

    training = adapter.load_training(feature_set)
    summary_records: list[dict[str, Any]] = []
    calibrated_oof: dict[str, pd.DataFrame] = {}
    for variant in variants:
        table = pd.concat(oof_by_variant[variant.name], ignore_index=True).sort_values(
            ["outer_fold", "sample_id"], kind="stable"
        ).reset_index(drop=True)
        calibrated, adjustment = cross_fit_conditional_calibration(
            table["observed_rul"].to_numpy(dtype=np.float64),
            table["uncalibrated_prediction"].to_numpy(dtype=np.float64),
            table["outer_fold"].to_numpy(),
            q_policy,
            prediction_minimum=0.0,
        )
        table["calibration_adjustment"] = adjustment
        table["predicted_rul"] = calibrated
        folds_table = _fold_metrics(table)
        variant_dir = run_root / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(variant_dir / "oof_predictions.csv.gz", index=False, compression="gzip")
        folds_table.to_csv(variant_dir / "fold_metrics.csv", index=False)
        summary_records.append(_summary(variant, folds_table, training))
        calibrated_oof[variant.name] = table

    manifests: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(variants))) as executor:
        future_map = {
            executor.submit(
                _fit_final_variant,
                adapter=adapter,
                feature_set=feature_set,
                variant=variant,
                oof=calibrated_oof[variant.name],
                q_policy=q_policy,
                extra_configuration=extra_configuration,
                xgboost_configuration=xgboost_configuration,
                blend_weight=blend_weight,
                seed=seed,
                output_dir=run_root / variant.name,
                log_root=log_root,
            ): variant.name
            for variant in variants
        }
        for future in as_completed(future_map):
            variant_name = future_map[future]
            manifests[variant_name] = future.result()
            print(f"{name}: verified submission {variant_name}", flush=True)

    summary_table = pd.DataFrame.from_records(summary_records).sort_values(
        ["mean_r2", "mean_rmse"], ascending=[False, True]
    )
    summary_path = run_root / "development_summary.csv"
    summary_table.to_csv(summary_path, index=False)
    pd.DataFrame.from_records(timing_records).sort_values(
        ["variant", "outer_fold"]
    ).to_csv(run_root / "oof_training_times.csv", index=False)
    submission_dir = run_root / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)
    submissions: dict[str, pd.DataFrame] = {}
    for variant in variants:
        source = pd.read_csv(run_root / variant.name / "submission.csv")
        source.to_csv(submission_dir / f"{variant.name}.csv", index=False)
        submissions[variant.name] = source
    figures = _plots(summary_table, submissions, run_root / "figures")
    manifest = {
        "status": "complete",
        "workflow": name,
        "pipeline_experiment": workflow.get("pipeline_experiment"),
        "pipeline_run": workflow.get("pipeline_run"),
        "source_phase_3_run": source_run,
        "architecture": "fixed 50/50 ExtraTrees/XGBoost blend",
        "feature_set": feature_set,
        "model_seed": seed,
        "locked_targets_loaded": False,
        "test_targets_loaded": False,
        "development_selection_performed": False,
        "leaderboard_comparison_required": True,
        "variants": {name: manifests[name] for name in REQUIRED_VARIANTS},
        "artifacts": {
            "development_summary": _repo_relative(summary_path),
            "submissions": {
                name: _repo_relative(submission_dir / f"{name}.csv")
                for name in REQUIRED_VARIANTS
            },
            "figures": [_repo_relative(path) for path in figures],
        },
    }
    _write_json(manifest, run_root / "target_submission_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        manifest = run(args.config.resolve(), args.run, force=args.force)
    except (TargetSubmissionError, OSError, ValueError, KeyError) as error:
        print(f"Target-policy submission experiment failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
