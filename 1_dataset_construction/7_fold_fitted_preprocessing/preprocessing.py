"""Fold-fitted feature selection and robust scaling utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PHASE_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from common import (  # noqa: E402
    STEP_2_ARTIFACT_DIR,
    STEP_5_ARTIFACT_DIR,
    STEP_6_ARTIFACT_DIR,
    STEP_7_ARTIFACT_DIR,
    save_json,
)


FEATURE_SETS = ("age_only", "last_values", "screened", "all_nonconstant")
RELATIVE_VARIATION_TOLERANCE = 1e-12


def selected_feature_names(catalog: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}; choose from {FEATURE_SETS}")
    if feature_set not in catalog.columns:
        raise ValueError(f"Feature catalog does not contain {feature_set!r}")
    mask = catalog[feature_set].astype(bool)
    names = catalog.loc[mask, "feature_name"].tolist()
    if not names:
        raise ValueError(f"Feature set {feature_set!r} is empty")
    return names


@dataclass(frozen=True)
class RobustScalerParameters:
    feature_names: tuple[str, ...]
    centers: np.ndarray
    scales: np.ndarray
    scale_methods: tuple[str, ...]
    iqrs: np.ndarray
    standard_deviations: np.ndarray
    data_ranges: np.ndarray
    variation_tolerances: np.ndarray


def fit_robust_scaler(
    training_rows: pd.DataFrame, feature_names: list[str]
) -> RobustScalerParameters:
    """Fit medians and IQR-based scales using training-fold rows only."""
    values = training_rows[feature_names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Training features contain missing or non-finite values")
    centers = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    iqrs = q75 - q25
    iqr_scales = iqrs / 1.349
    standard_deviations = np.std(values, axis=0, ddof=1)
    data_ranges = np.ptp(values, axis=0)
    feature_magnitudes = np.max(np.abs(values), axis=0)
    variation_tolerances = RELATIVE_VARIATION_TOLERANCE * np.maximum(
        feature_magnitudes, 1.0
    )
    has_meaningful_range = data_ranges > variation_tolerances
    use_iqr = has_meaningful_range & (iqr_scales > variation_tolerances)
    use_standard_deviation = has_meaningful_range & (~use_iqr)
    scales = np.where(
        use_iqr,
        iqr_scales,
        np.where(use_standard_deviation, standard_deviations, 1.0),
    )
    scale_methods = tuple(
        np.where(
            use_iqr,
            "iqr",
            np.where(
                use_standard_deviation,
                "standard_deviation_fallback",
                "unit_fallback",
            ),
        ).tolist()
    )
    return RobustScalerParameters(
        tuple(feature_names),
        centers,
        scales,
        scale_methods,
        iqrs,
        standard_deviations,
        data_ranges,
        variation_tolerances,
    )


def transform_robust(
    rows: pd.DataFrame, parameters: RobustScalerParameters
) -> np.ndarray:
    values = rows[list(parameters.feature_names)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Features contain missing or non-finite values")
    return (values - parameters.centers) / parameters.scales


def outer_fold_rows(
    table: pd.DataFrame,
    outer_folds: pd.DataFrame,
    outer_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_by_uav = outer_folds.set_index("uav_id")["outer_fold"]
    row_folds = table["uav_id"].map(fold_by_uav)
    if row_folds.isna().any():
        raise ValueError("Feature table contains UAVs absent from outer_folds.csv")
    return table.loc[row_folds != outer_fold], table.loc[row_folds == outer_fold]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", type=Path, default=STEP_2_ARTIFACT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=STEP_5_ARTIFACT_DIR)
    parser.add_argument("--feature-set-dir", type=Path, default=STEP_6_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=STEP_7_ARTIFACT_DIR)
    args = parser.parse_args()

    folds = pd.read_csv(args.fold_dir / "outer_folds.csv")
    training = pd.read_csv(args.feature_dir / "training_features.csv.gz")
    catalog = pd.read_csv(args.feature_set_dir / "feature_catalog.csv")
    records: list[dict[str, float | int | str]] = []
    feature_counts: dict[str, int] = {}

    for feature_set in FEATURE_SETS:
        names = selected_feature_names(catalog, feature_set)
        feature_counts[feature_set] = len(names)
        for outer_fold in sorted(folds["outer_fold"].unique()):
            fit_rows, _ = outer_fold_rows(training, folds, int(outer_fold))
            parameters = fit_robust_scaler(fit_rows, names)
            for (
                feature_name,
                center,
                scale,
                scale_method,
                iqr,
                standard_deviation,
                data_range,
                variation_tolerance,
            ) in zip(
                parameters.feature_names,
                parameters.centers,
                parameters.scales,
                parameters.scale_methods,
                parameters.iqrs,
                parameters.standard_deviations,
                parameters.data_ranges,
                parameters.variation_tolerances,
                strict=True,
            ):
                records.append(
                    {
                        "outer_fold": int(outer_fold),
                        "feature_set": feature_set,
                        "feature_name": feature_name,
                        "center": float(center),
                        "scale": float(scale),
                        "scale_method": scale_method,
                        "iqr": float(iqr),
                        "standard_deviation": float(standard_deviation),
                        "data_range": float(data_range),
                        "variation_tolerance": float(variation_tolerance),
                        "fit_uavs": int(fit_rows["uav_id"].nunique()),
                        "fit_rows": int(len(fit_rows)),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_path = args.output_dir / "fold_scaler_parameters.csv.gz"
    parameter_table = pd.DataFrame.from_records(records)
    parameter_table.to_csv(
        parameter_path,
        index=False,
        compression="gzip",
    )
    scale_method_counts = {
        str(method): int(count)
        for method, count in parameter_table["scale_method"].value_counts().items()
    }
    config_path = save_json(
        {
            "scaling": "median centering with IQR/1.349 scale",
            "fallback": "standard deviation, then 1.0",
            "relative_variation_tolerance": RELATIVE_VARIATION_TOLERANCE,
            "variation_tolerance_formula": "1e-12 * max(max_abs_value, 1.0)",
            "scale_method_counts": scale_method_counts,
            "fit_scope": "outer-training UAV prefixes only",
            "feature_counts": feature_counts,
            "outer_folds": int(folds["outer_fold"].nunique()),
        },
        args.output_dir / "preprocessing_config.json",
    )
    print(f"Saved {parameter_path}")
    print(f"Saved {config_path}")


if __name__ == "__main__":
    main()
