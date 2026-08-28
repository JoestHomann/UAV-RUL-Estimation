"""Verify frozen validation artifacts and causal feature tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PHASE_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PHASE_ROOT / "5_prefix_feature_engineering"
PREPROCESSING_DIR = PHASE_ROOT / "7_fold_fitted_preprocessing"
for import_path in (PHASE_ROOT, FEATURE_DIR, PREPROCESSING_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from build_prefix_features import FEATURE_PREFIX, assert_feature_causality
from common import (
    DEFAULT_TRAIN_CSV,
    ID_COLUMN,
    STEP_1_ARTIFACT_DIR,
    STEP_2_ARTIFACT_DIR,
    STEP_3_ARTIFACT_DIR,
    STEP_4_ARTIFACT_DIR,
    STEP_5_ARTIFACT_DIR,
    STEP_6_ARTIFACT_DIR,
    STEP_7_ARTIFACT_DIR,
    STEP_9_ARTIFACT_DIR,
    STEP_10_ARTIFACT_DIR,
    load_dataset,
    save_json,
)
from feature_recipes import catalog_feature_sets
from preprocessing import (
    fit_robust_scaler,
    outer_fold_rows,
    selected_feature_names,
    transform_robust,
)


def finite_feature_file(path: Path, feature_names: list[str]) -> tuple[int, int]:
    rows = 0
    for chunk in pd.read_csv(path, chunksize=250):
        values = chunk[feature_names].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise AssertionError(f"{path} contains missing or non-finite features")
        rows += len(chunk)
    return rows, len(feature_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--audit-dir", type=Path, default=STEP_1_ARTIFACT_DIR)
    parser.add_argument("--fold-dir", type=Path, default=STEP_2_ARTIFACT_DIR)
    parser.add_argument("--scenario-dir", type=Path, default=STEP_3_ARTIFACT_DIR)
    parser.add_argument("--prefix-dir", type=Path, default=STEP_4_ARTIFACT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=STEP_5_ARTIFACT_DIR)
    parser.add_argument("--feature-set-dir", type=Path, default=STEP_6_ARTIFACT_DIR)
    parser.add_argument("--preprocessing-dir", type=Path, default=STEP_7_ARTIFACT_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=STEP_9_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=STEP_10_ARTIFACT_DIR)
    parser.add_argument(
        "--feature-profile",
        choices=("legacy", "extended"),
        default="legacy",
    )
    args = parser.parse_args()

    outer = pd.read_csv(args.fold_dir / "outer_folds.csv")
    inner = pd.read_csv(args.fold_dir / "inner_folds.csv")
    training_manifest = pd.read_csv(args.prefix_dir / "training_prefixes.csv")
    locked = pd.read_csv(args.scenario_dir / "locked_validation_scenarios.csv")
    development = pd.read_csv(
        args.scenario_dir / "development_validation_scenarios.csv"
    )
    test_summary = pd.read_csv(
        args.audit_dir / "test_fligh_cycles_cut_offs.csv"
    )
    catalog = pd.read_csv(args.feature_set_dir / "feature_catalog.csv")
    feature_sets = catalog_feature_sets(catalog)
    feature_names = catalog["feature_name"].tolist()
    forbidden = ("rul", "target", "terminal", "lifetime", "final", "future")
    invalid_names = [
        name for name in feature_names if any(token in name.lower() for token in forbidden)
    ]
    if invalid_names:
        raise AssertionError(f"Feature catalog contains future/target fields: {invalid_names}")
    if not all(name.startswith(FEATURE_PREFIX) for name in feature_names):
        raise AssertionError("Feature catalog contains names without the feature prefix")

    fold_counts = outer.groupby("outer_fold")[ID_COLUMN].nunique()
    if not fold_counts.eq(20).all():
        raise AssertionError(f"Outer fold UAV counts are not 20: {fold_counts.to_dict()}")
    if outer[ID_COLUMN].duplicated().any():
        raise AssertionError("Outer folds overlap by UAV")
    for outer_fold, group in inner.groupby("outer_fold"):
        outer_validation = set(outer.loc[outer["outer_fold"] == outer_fold, ID_COLUMN])
        if outer_validation & set(group[ID_COLUMN]):
            raise AssertionError(f"Outer fold {outer_fold} leaks into its inner folds")

    expected_test_lengths = np.sort(test_summary["final_cycle"].to_numpy(dtype=int))
    for table_name, table in [("locked", locked), ("development", development)]:
        for scenario, group in table.groupby("scenario"):
            if group[ID_COLUMN].nunique() != 100:
                raise AssertionError(f"{table_name}/{scenario} does not contain 100 UAVs")
            if not np.array_equal(
                np.sort(group["cutoff"].to_numpy(dtype=int)), expected_test_lengths
            ):
                raise AssertionError(
                    f"{table_name}/{scenario} does not exactly match test lengths"
                )

    feature_dir = args.feature_dir
    expected_rows = {
        "training_features.csv.gz": len(training_manifest),
        "locked_validation_features.csv.gz": len(locked),
        "development_validation_features.csv.gz": len(development),
        "test_features.csv.gz": 100,
    }
    feature_file_results: dict[str, dict[str, int]] = {}
    for filename, expected in expected_rows.items():
        rows, columns = finite_feature_file(feature_dir / filename, feature_names)
        if rows != expected:
            raise AssertionError(f"{filename} has {rows} rows; expected {expected}")
        feature_file_results[filename] = {"rows": rows, "features": columns}

    train = load_dataset(args.train_csv, require_target=True)
    assert_feature_causality(
        train,
        training_manifest,
        feature_profile=args.feature_profile,
    )
    total_weights = training_manifest.groupby(ID_COLUMN)["sample_weight"].sum()
    if not np.allclose(total_weights.to_numpy(dtype=float), 1.0):
        raise AssertionError("Training prefixes do not give every UAV total weight one")
    if training_manifest.duplicated([ID_COLUMN, "cutoff"]).any():
        raise AssertionError("Training prefixes contain duplicate UAV/cutoff rows")

    training_features = pd.read_csv(feature_dir / "training_features.csv.gz")
    saved_scalers = pd.read_csv(
        args.preprocessing_dir / "fold_scaler_parameters.csv.gz"
    )
    preprocessing_checks: dict[str, int] = {}
    for feature_set in feature_sets:
        names = selected_feature_names(catalog, feature_set)
        for outer_fold in sorted(outer["outer_fold"].unique()):
            fit_rows, held_out_rows = outer_fold_rows(
                training_features, outer, int(outer_fold)
            )
            parameters = fit_robust_scaler(fit_rows, names)
            recorded = saved_scalers.loc[
                (saved_scalers["feature_set"] == feature_set)
                & (saved_scalers["outer_fold"] == outer_fold)
            ].set_index("feature_name")
            if set(recorded.index) != set(parameters.feature_names):
                raise AssertionError(
                    f"Saved scaler features differ for {feature_set}/fold {outer_fold}"
                )
            ordered = recorded.loc[list(parameters.feature_names)]
            if not np.allclose(ordered["center"], parameters.centers):
                raise AssertionError(
                    f"Saved scaler centers differ for {feature_set}/fold {outer_fold}"
                )
            if not np.allclose(ordered["scale"], parameters.scales):
                raise AssertionError(
                    f"Saved scaler scales differ for {feature_set}/fold {outer_fold}"
                )
            if ordered["scale_method"].tolist() != list(parameters.scale_methods):
                raise AssertionError(
                    "Saved scaler fallback methods differ for "
                    f"{feature_set}/fold {outer_fold}"
                )
            if not np.allclose(
                ordered["variation_tolerance"],
                parameters.variation_tolerances,
                rtol=1e-12,
                atol=0.0,
            ):
                raise AssertionError(
                    "Saved scaler variation tolerances differ for "
                    f"{feature_set}/fold {outer_fold}"
                )
            unit_fallback = ordered["scale_method"] == "unit_fallback"
            if (
                ordered.loc[unit_fallback, "data_range"]
                > ordered.loc[unit_fallback, "variation_tolerance"]
            ).any():
                raise AssertionError(
                    "Unit fallback used despite meaningful feature variation for "
                    f"{feature_set}/fold {outer_fold}"
                )
            transformed = transform_robust(held_out_rows, parameters)
            if not np.isfinite(transformed).all():
                raise AssertionError(
                    f"Fold-fitted preprocessing failed for {feature_set}/fold {outer_fold}"
                )
        preprocessing_checks[feature_set] = len(names)

    baseline_predictions = pd.read_csv(
        args.baseline_dir / "locked_predictions.csv"
    )
    if len(baseline_predictions) != len(locked):
        raise AssertionError("Cycle baseline prediction count is incorrect")
    if baseline_predictions.duplicated(["scenario", ID_COLUMN]).any():
        raise AssertionError("Cycle baseline contains duplicate scenario/UAV predictions")
    expected_fold = baseline_predictions[ID_COLUMN].map(
        outer.set_index(ID_COLUMN)["outer_fold"]
    )
    if not expected_fold.eq(baseline_predictions["outer_fold"]).all():
        raise AssertionError("Cycle baseline predictions use an incorrect outer fold")

    report: dict[str, Any] = {
        "status": "passed",
        "assertions": {
            "outer_uav_folds_disjoint_and_balanced": True,
            "outer_validation_uavs_absent_from_inner_folds": True,
            "locked_cutoffs_exactly_match_test_history_lengths": True,
            "development_cutoffs_exactly_match_test_history_lengths": True,
            "feature_tables_finite": True,
            "training_prefixes_equalize_total_uav_weight": True,
            "feature_names_exclude_target_and_future_fields": True,
            "future_rows_cannot_change_prefix_features": True,
            "preprocessing_fitted_separately_for_each_outer_training_fold": True,
            "saved_preprocessing_parameters_match_recomputed_values": True,
            "cycle_baseline_predictions_are_group_held_out": True,
        },
        "outer_fold_uav_counts": {
            str(key): int(value) for key, value in fold_counts.items()
        },
        "feature_files": feature_file_results,
        "feature_sets": preprocessing_checks,
        "locked_scenarios": int(locked["scenario"].nunique()),
        "development_scenarios": int(development["scenario"].nunique()),
    }
    path = save_json(report, args.output_dir / "verification_report.json")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
