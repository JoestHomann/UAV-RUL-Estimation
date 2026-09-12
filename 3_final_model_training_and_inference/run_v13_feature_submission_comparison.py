"""Build two exploratory Phase 3 submissions from the standalone v13 model.

Run 9 preserves v13's current tiered feature recipe. Run 10 removes the raw
current-value and rolling-standard-deviation families identified as redundant
by the Run 10 historical feature-importance screen. All other v13 mechanics,
parameters, seeds, sample weights, feature weights, and early-specialist logic
remain unchanged.

These runs intentionally use a dedicated entry point because v13 is not a
registered Phase 2 model adapter. Artifacts explicitly record that limitation
and must not be described as promoted production Phase 3 models.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd


PHASE_DIR = Path(__file__).resolve().parent
ROOT = PHASE_DIR.parent
SOURCE = ROOT / "other_pipelines" / "uav_rul_pipeline_v13 (1).py"
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
IMPORTANCE_PATH = (
    ROOT
    / "2_architecture_experiments"
    / "2_model_architecture_study"
    / "runs"
    / "run_10"
    / "reporting"
    / "feature_importance_study"
    / "historical_feature_importance_consensus.csv"
)

VARIANTS = {
    "original": {
        "run_number": 9,
        "excluded_families": (),
        "expected_features": 153,
    },
    "reduced": {
        "run_number": 10,
        "excluded_families": (
            "raw",
            "roll5_std",
            "roll10_std",
            "roll20_std",
        ),
        "expected_features": 103,
    },
}

# User-reported Kaggle public leaderboard results. These are kept separate from
# model fitting and are never used to select or regenerate predictions.
KAGGLE_PUBLIC_SCORES = {
    "original": 0.89093,
    "reduced": 0.88701,
}


def load_v13():
    spec = importlib.util.spec_from_file_location("uav_rul_pipeline_v13", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_joblib(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        joblib.dump(payload, temporary, compress=3)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_root(run_number: int) -> Path:
    return PHASE_DIR / "runs" / f"run_{run_number}"


def step_dir(root: Path, number: int, name: str) -> Path:
    return root / f"{number}_{name}"


def feature_family_from_column(column: str) -> str:
    if column in {"flight_cycle", "flight_cycle_log"}:
        return "flight_age"
    if "__" not in column:
        return "raw"
    return column.split("__", maxsplit=1)[1]


def verify_reduction_source() -> dict:
    if not IMPORTANCE_PATH.is_file():
        raise FileNotFoundError(f"Feature-importance result missing: {IMPORTANCE_PATH}")
    table = pd.read_csv(IMPORTANCE_PATH)
    required = {
        "group_type",
        "group",
        "rmse_increase",
        "rmse_increase_ci_lower",
        "rmse_increase_ci_upper",
    }
    if not required.issubset(table.columns):
        raise ValueError("Feature-importance table has an unexpected schema")
    families = table.loc[table["group_type"] == "feature_family"].set_index("group")
    evidence = {}
    for group in ("Current value", "Rolling variability"):
        if group not in families.index:
            raise ValueError(f"Feature-importance table is missing {group!r}")
        row = families.loc[group]
        evidence[group] = {
            "rmse_increase": float(row["rmse_increase"]),
            "ci_lower": float(row["rmse_increase_ci_lower"]),
            "ci_upper": float(row["rmse_increase_ci_upper"]),
        }
    return evidence


def validate_submission(submission: pd.DataFrame, expected_ids: list[str]) -> None:
    if list(submission.columns) != ["id", "RUL"]:
        raise ValueError("Submission must contain exactly id,RUL")
    if submission["id"].astype(str).tolist() != expected_ids:
        raise ValueError("Submission IDs or ordering changed")
    if submission["id"].duplicated().any():
        raise ValueError("Submission IDs are duplicated")
    values = submission["RUL"].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Submission predictions must be finite and nonnegative")


def generate_predictions(v13, model_artifact: dict, test_raw: pd.DataFrame) -> pd.DataFrame:
    test_features = v13.build_features_tiered(
        test_raw,
        model_artifact["sensor_tiers"],
        excluded_families=model_artifact["excluded_families"],
    )
    feature_columns = model_artifact["feature_columns"]
    if [column for column in test_features if column != "uav_id"] != feature_columns:
        raise ValueError("Inference feature columns differ from the saved contract")
    test_last = (
        test_features.sort_values("flight_cycle")
        .groupby("uav_id")
        .tail(1)
        .sort_values("uav_id", kind="stable")
        .reset_index(drop=True)
    )
    main_prediction = model_artifact["main_model"].predict(
        test_last[feature_columns].to_numpy()
    )
    early_prediction = model_artifact["early_model"].predict(
        test_last[feature_columns].to_numpy()
    )
    gated = test_last["flight_cycle"].to_numpy() <= model_artifact["early_gate_max"]
    final_prediction = main_prediction.copy()
    gate_weight = model_artifact["early_gate_weight"]
    final_prediction[gated] = (
        (1.0 - gate_weight) * main_prediction[gated]
        + gate_weight * early_prediction[gated]
    )
    return pd.DataFrame(
        {
            "uav_id": test_last["uav_id"].astype(str),
            "flight_cycle": test_last["flight_cycle"].astype(int),
            "main_RUL": main_prediction,
            "early_specialist_RUL": early_prediction,
            "early_gate_applied": gated,
            "RUL": final_prediction,
        }
    )


def build_variant(
    variant: str,
    v13,
    train_uncapped: pd.DataFrame,
    test_raw: pd.DataFrame,
    sensor_tiers: dict[str, list[str]],
    importance_evidence: dict,
) -> dict:
    definition = VARIANTS[variant]
    run_number = definition["run_number"]
    root = run_root(run_number)
    final_manifest = root / "final_run_manifest.json"
    if final_manifest.is_file():
        existing = json.loads(final_manifest.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("variant") == variant:
            print(f"Reusing completed Phase 3 Run {run_number} ({variant})", flush=True)
            return existing
        raise ValueError(f"Phase 3 Run {run_number} already contains incompatible output")

    started = time.time()
    print("", flush=True)
    print("=" * 78, flush=True)
    print(f"Phase 3 Run {run_number}: v13 {variant} feature set", flush=True)
    print("=" * 78, flush=True)
    excluded = tuple(definition["excluded_families"])
    settings = {
        "settings_version": 1,
        "phase_3_run_number": run_number,
        "status": "exploratory_unpromoted",
        "model_family": "v13_xgboost_early_specialist",
        "variant": variant,
        "source_script": str(SOURCE.relative_to(ROOT).as_posix()),
        "source_script_sha256": sha256(SOURCE),
        "training_data_sha256": sha256(TRAIN_PATH),
        "test_data_sha256": sha256(TEST_PATH),
        "feature_importance_sha256": sha256(IMPORTANCE_PATH),
        "excluded_families": list(excluded),
        "expected_feature_count": definition["expected_features"],
        "standard_phase_3_model_registry_used": False,
        "test_target_loaded": False,
    }

    step1 = step_dir(root, 1, "winning_architecture_selection")
    write_json(step1 / "artifacts" / "resolved_phase_3_settings.json", settings)
    write_json(
        step1 / "artifacts" / "selected_architecture.json",
        {
            "model_family": settings["model_family"],
            "variant": variant,
            "promotion_status": "exploratory_unpromoted",
            "source": settings["source_script"],
        },
    )
    write_json(
        step1 / "architecture_selection_manifest.json",
        {"status": "complete", "settings_version": 1, "run_number": run_number},
    )

    train_raw = v13.apply_rul_cap(train_uncapped, v13.RUL_CAP)
    train_raw = train_raw.sort_values(["uav_id", "flight_cycle"]).reset_index(drop=True)
    test_sorted = test_raw.sort_values(["uav_id", "flight_cycle"]).reset_index(drop=True)
    test_cutoff_lengths = v13.get_test_cutoff_lengths(test_sorted)
    train_features = v13.build_features_tiered(
        train_raw, sensor_tiers, excluded_families=excluded
    )
    test_features = v13.build_features_tiered(
        test_sorted, sensor_tiers, excluded_families=excluded
    )
    train_features["RUL_raw"] = train_raw["RUL"].to_numpy()
    train_features["uav_id_check"] = train_raw["uav_id"].to_numpy()
    feature_columns = [
        column
        for column in train_features
        if column not in {"uav_id", "RUL_raw", "uav_id_check"}
    ]
    if len(feature_columns) != definition["expected_features"]:
        raise ValueError(
            f"{variant} expected {definition['expected_features']} features, "
            f"found {len(feature_columns)}"
        )
    if [column for column in test_features if column != "uav_id"] != feature_columns:
        raise ValueError("Training and test feature schemas differ")

    sample_weights = v13.compute_test_similarity_weights(
        train_features["flight_cycle"], test_cutoff_lengths
    )
    all_sensors = [sensor for tier in sensor_tiers.values() for sensor in tier]
    feature_weights = v13.compute_feature_weights(
        feature_columns, train_raw, all_sensors
    )
    feature_table = pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_family": [
                feature_family_from_column(column) for column in feature_columns
            ],
            "feature_weight": feature_weights,
        }
    )

    step2 = step_dir(root, 2, "final_configuration_search")
    configuration = {
        "configuration_id": f"v13_{variant}_frozen",
        "search_performed": False,
        "reason": "Use the submitted v13 configuration unchanged for the matched feature comparison",
        "xgb_parameters": v13.XGB_PARAMS,
        "early_stopping_rounds": v13.XGB_EARLY_STOPPING_ROUNDS,
        "rul_cap": v13.RUL_CAP,
        "early_gate_max": v13.EARLY_GATE_MAX,
        "early_train_max": v13.EARLY_TRAIN_MAX,
    }
    write_json(step2 / "artifacts" / "selected_configuration.json", configuration)
    write_json(
        step2 / "final_search_manifest.json",
        {"status": "complete", "settings_version": 1, "candidate_budget": 1},
    )

    step3 = step_dir(root, 3, "final_training_contract")
    contract = {
        "status": "frozen",
        "settings_version": 1,
        "run_number": run_number,
        "model_family": settings["model_family"],
        "variant": variant,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "excluded_families": list(excluded),
        "sensor_tiers": sensor_tiers,
        "training_uavs": int(train_raw["uav_id"].nunique()),
        "test_contract": {
            "expected_uavs": int(test_sorted["uav_id"].nunique()),
            "prediction_columns": [
                "uav_id",
                "flight_cycle",
                "main_RUL",
                "early_specialist_RUL",
                "early_gate_applied",
                "RUL",
            ],
            "submission_columns": ["id", "RUL"],
            "test_target_loaded": False,
        },
        "reduction_evidence": importance_evidence if variant == "reduced" else None,
        "protocol_limit": "Standalone v13 is absent from the registered Phase 2 adapter comparison",
    }
    write_json(step3 / "artifacts" / "final_training_contract.json", contract)
    write_csv(step3 / "artifacts" / "feature_contract.csv", feature_table)
    write_json(
        step3 / "final_training_contract_manifest.json",
        {"status": "complete", "settings_version": 1, "feature_count": len(feature_columns)},
    )

    submission_no_early, local_r2, extras = v13.run_full_pipeline(
        train_features,
        test_features,
        feature_columns,
        train_raw,
        test_cutoff_lengths,
        sample_weight=sample_weights,
        feature_weights=feature_weights,
        label=f"V13 {variant.upper()}",
    )
    early_model = v13.fit_early_specialist(
        train_features,
        feature_columns,
        sample_weight=sample_weights,
    )

    def main_prediction(eval_frame):
        return extras["xgb_final"].predict(eval_frame[feature_columns].to_numpy())

    gate_weight = v13.find_early_gate_weight(
        train_features,
        train_raw,
        test_cutoff_lengths,
        feature_columns,
        main_prediction,
        early_model,
    )
    model_payload = {
        "artifact_format_version": 1,
        "model_family": settings["model_family"],
        "variant": variant,
        "main_model": extras["xgb_final"],
        "early_model": early_model,
        "early_gate_weight": float(gate_weight),
        "early_gate_max": int(v13.EARLY_GATE_MAX),
        "early_train_max": int(v13.EARLY_TRAIN_MAX),
        "feature_columns": feature_columns,
        "excluded_families": excluded,
        "sensor_tiers": sensor_tiers,
        "source_sha256": settings["source_script_sha256"],
    }

    step4 = step_dir(root, 4, "final_model_training")
    model_path = step4 / "artifacts" / "final_model.joblib"
    write_joblib(model_path, model_payload)
    training_summary = {
        "status": "complete",
        "variant": variant,
        "feature_count": len(feature_columns),
        "training_rows": len(train_features),
        "training_uavs": int(train_features["uav_id_check"].nunique()),
        "local_test_like_main_r2": float(local_r2),
        "main_best_iteration": int(extras["xgb_final"].best_iteration),
        "early_best_iteration": int(early_model.best_iteration),
        "early_gate_weight": float(gate_weight),
        "elapsed_seconds_to_training_complete": float(time.time() - started),
    }
    write_json(step4 / "artifacts" / "training_summary.json", training_summary)
    write_json(
        step4 / "final_training_manifest.json",
        {
            "status": "complete",
            "settings_version": 1,
            "model_sha256": sha256(model_path),
            **training_summary,
        },
    )

    saved_model = joblib.load(model_path)
    predictions = generate_predictions(v13, saved_model, test_sorted)
    expected_ids = sorted(test_sorted["uav_id"].astype(str).unique().tolist())
    if predictions["uav_id"].tolist() != expected_ids:
        raise ValueError("Generated prediction IDs differ from test UAV IDs")
    if not np.isfinite(predictions.select_dtypes(include=[np.number])).all().all():
        raise ValueError("Generated predictions contain nonfinite values")
    if (predictions["RUL"] < 0).any():
        raise ValueError("v13 produced a negative prediction; no unregistered clipping was applied")

    step5 = step_dir(root, 5, "test_inference")
    prediction_path = step5 / "artifacts" / "test_predictions.csv"
    write_csv(prediction_path, predictions)
    write_json(
        step5 / "inference_manifest.json",
        {
            "status": "complete",
            "settings_version": 1,
            "model_family": settings["model_family"],
            "variant": variant,
            "feature_count": len(feature_columns),
            "prediction_rows": len(predictions),
            "test_uav_count": len(expected_ids),
            "prediction_minimum": float(predictions["RUL"].min()),
            "prediction_maximum": float(predictions["RUL"].max()),
            "early_gate_count": int(predictions["early_gate_applied"].sum()),
            "test_target_loaded": False,
            "test_metrics_calculated": False,
            "prediction_sha256": sha256(prediction_path),
        },
    )

    canonical = predictions[["uav_id", "RUL"]].rename(columns={"uav_id": "id"})
    canonical = canonical.sort_values("id", kind="stable").reset_index(drop=True)
    no_early = submission_no_early.sort_values("id", kind="stable").reset_index(drop=True)
    validate_submission(canonical, expected_ids)
    validate_submission(no_early, expected_ids)
    step6 = step_dir(root, 6, "submission_verification")
    submission_path = step6 / "artifacts" / "submission.csv"
    no_early_path = step6 / "artifacts" / "submission_no_early.csv"
    write_csv(submission_path, canonical)
    write_csv(no_early_path, no_early)
    validate_submission(pd.read_csv(submission_path), expected_ids)
    validate_submission(pd.read_csv(no_early_path), expected_ids)
    write_json(
        step6 / "submission_manifest.json",
        {
            "status": "complete",
            "settings_version": 1,
            "variant": variant,
            "rows": len(canonical),
            "columns": ["id", "RUL"],
            "identifier_set_verified": True,
            "finite_nonnegative_values_verified": True,
            "deterministic_order_verified": True,
            "regenerated_prediction_equivalence": True,
            "test_metrics_calculated": False,
            "artifacts": {
                "submission": "artifacts/submission.csv",
                "submission_no_early": "artifacts/submission_no_early.csv",
            },
        },
    )

    final = {
        "manifest_version": 1,
        "settings_version": 1,
        "phase_3_run_number": run_number,
        "status": "complete",
        "promotion_status": "exploratory_unpromoted",
        "standard_phase_3_model_registry_used": False,
        "model_family": settings["model_family"],
        "variant": variant,
        "feature_count": len(feature_columns),
        "local_test_like_main_r2": float(local_r2),
        "early_gate_weight": float(gate_weight),
        "elapsed_seconds": float(time.time() - started),
        "artifacts": {
            "settings": "1_winning_architecture_selection/artifacts/resolved_phase_3_settings.json",
            "configuration": "2_final_configuration_search/artifacts/selected_configuration.json",
            "training_contract": "3_final_training_contract/artifacts/final_training_contract.json",
            "feature_contract": "3_final_training_contract/artifacts/feature_contract.csv",
            "model": "4_final_model_training/artifacts/final_model.joblib",
            "test_predictions": "5_test_inference/artifacts/test_predictions.csv",
            "submission": "6_submission_verification/artifacts/submission.csv",
            "submission_no_early": "6_submission_verification/artifacts/submission_no_early.csv",
        },
        "test_target_loaded": False,
        "test_metrics_calculated": False,
    }
    write_json(final_manifest, final)
    print(f"Completed Phase 3 Run {run_number} ({variant})", flush=True)
    return final


def build_comparison(manifests: list[dict]) -> None:
    records = []
    submissions = {}
    for manifest in manifests:
        run_number = manifest["phase_3_run_number"]
        variant = manifest["variant"]
        root = run_root(run_number)
        submission = pd.read_csv(root / manifest["artifacts"]["submission"])
        submissions[variant] = submission.set_index("id")["RUL"].sort_index()
        inference = json.loads(
            (root / "5_test_inference" / "inference_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        records.append(
            {
                "phase_3_run": run_number,
                "variant": variant,
                "features": manifest["feature_count"],
                "local_test_like_main_r2": manifest["local_test_like_main_r2"],
                "early_gate_weight": manifest["early_gate_weight"],
                "early_gate_count": inference["early_gate_count"],
                "prediction_minimum": inference["prediction_minimum"],
                "prediction_maximum": inference["prediction_maximum"],
                "kaggle_public_score": KAGGLE_PUBLIC_SCORES.get(variant),
            }
        )
    if set(submissions) == {"original", "reduced"}:
        delta = submissions["reduced"] - submissions["original"]
        comparison = {
            "mean_prediction_delta_reduced_minus_original": float(delta.mean()),
            "mean_absolute_prediction_delta": float(delta.abs().mean()),
            "maximum_absolute_prediction_delta": float(delta.abs().max()),
            "prediction_correlation": float(
                submissions["original"].corr(submissions["reduced"])
            ),
            "identical_submission": bool((delta == 0).all()),
            "kaggle_score_delta_reduced_minus_original": float(
                KAGGLE_PUBLIC_SCORES["reduced"] - KAGGLE_PUBLIC_SCORES["original"]
            ),
            "local_r2_delta_reduced_minus_original": float(
                next(
                    item["local_test_like_main_r2"]
                    for item in records
                    if item["variant"] == "reduced"
                )
                - next(
                    item["local_test_like_main_r2"]
                    for item in records
                    if item["variant"] == "original"
                )
            ),
        }
    else:
        comparison = {}
    output = PHASE_DIR / "runs" / "v13_feature_comparison"
    write_csv(output / "run_comparison.csv", pd.DataFrame(records))
    write_json(
        output / "comparison_manifest.json",
        {
            "status": "complete",
            "runs": [manifest["phase_3_run_number"] for manifest in manifests],
            "comparison": comparison,
            "leaderboard_results": {
                "source": "user-reported Kaggle submission screenshot",
                "metric": "R2",
                "public_scores": KAGGLE_PUBLIC_SCORES,
                "used_for_model_fitting": False,
            },
        },
    )


def build_post_run_report(manifest: dict) -> dict:
    """Complete the seventh Phase 3 stage from verified stored artifacts."""

    run_number = manifest["phase_3_run_number"]
    root = run_root(run_number)
    training = json.loads(
        (root / "4_final_model_training" / "artifacts" / "training_summary.json").read_text(
            encoding="utf-8"
        )
    )
    inference = json.loads(
        (root / "5_test_inference" / "inference_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    submission_manifest = json.loads(
        (root / "6_submission_verification" / "submission_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    submission = pd.read_csv(
        root / "6_submission_verification" / "artifacts" / "submission.csv"
    )
    report = {
        "status": "complete",
        "promotion_status": manifest["promotion_status"],
        "phase_3_run_number": run_number,
        "variant": manifest["variant"],
        "model_family": manifest["model_family"],
        "feature_count": manifest["feature_count"],
        "local_test_like_main_r2": training["local_test_like_main_r2"],
        "early_gate_weight": training["early_gate_weight"],
        "early_gate_count": inference["early_gate_count"],
        "prediction_minimum": float(submission["RUL"].min()),
        "prediction_mean": float(submission["RUL"].mean()),
        "prediction_maximum": float(submission["RUL"].max()),
        "submission_rows": submission_manifest["rows"],
        "submission_verified": True,
        "test_target_loaded": False,
        "test_metrics_calculated": False,
        "protocol_limit": "Standalone v13 was not evaluated through the registered Phase 2 adapter protocol",
    }
    step7 = step_dir(root, 7, "post_run_reporting")
    write_json(step7 / "report_summary.json", report)
    write_json(
        step7 / "report_manifest.json",
        {
            "status": "complete",
            "settings_version": 1,
            "phase_3_run_number": run_number,
            "report": "report_summary.json",
        },
    )
    updated = dict(manifest)
    updated["artifacts"] = dict(updated["artifacts"])
    updated["artifacts"]["report"] = "7_post_run_reporting/report_summary.json"
    write_json(root / "final_run_manifest.json", updated)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=("original", "reduced"), help="Run one arm only"
    )
    args = parser.parse_args()
    importance_evidence = verify_reduction_source()
    v13 = load_v13()
    train_uncapped = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    if "RUL" in test_raw.columns:
        raise ValueError("Test data unexpectedly contains a target column")
    capped = v13.apply_rul_cap(train_uncapped, v13.RUL_CAP)
    sensors = v13.get_sensor_cols(capped)
    sensor_tiers = v13.compute_sensor_tiers(capped, sensors)
    order = [args.only] if args.only else ["original", "reduced"]
    manifests = [
        build_variant(
            variant,
            v13,
            train_uncapped,
            test_raw,
            sensor_tiers,
            importance_evidence,
        )
        for variant in order
    ]
    manifests = [build_post_run_report(manifest) for manifest in manifests]
    if len(manifests) == 2:
        build_comparison(manifests)
    print("Requested v13 Phase 3 submission run(s) completed", flush=True)


if __name__ == "__main__":
    main()
