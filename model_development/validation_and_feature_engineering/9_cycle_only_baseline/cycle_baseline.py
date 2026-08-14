"""Evaluate a leakage-safe flight-cycle-only baseline on locked scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PHASE_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PHASE_ROOT / "8_validation_metrics"
for import_path in (PHASE_ROOT, METRICS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from common import (  # noqa: E402
    ID_COLUMN,
    STEP_2_ARTIFACT_DIR,
    STEP_5_ARTIFACT_DIR,
    STEP_9_ARTIFACT_DIR,
    save_csv,
    save_json,
)
from validation_metrics import evaluate_predictions


def weighted_linear_fit(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(x)), x.astype(float)])
    square_root_weights = np.sqrt(weights.astype(float))
    coefficients, *_ = np.linalg.lstsq(
        design * square_root_weights[:, None],
        y.astype(float) * square_root_weights,
        rcond=None,
    )
    return float(coefficients[0]), float(coefficients[1])


def predict(intercept: float, slope: float, cycles: pd.Series) -> np.ndarray:
    return np.maximum(0.0, intercept + slope * cycles.to_numpy(dtype=float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", type=Path, default=STEP_2_ARTIFACT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=STEP_5_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=STEP_9_ARTIFACT_DIR)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    training = pd.read_csv(args.feature_dir / "training_features.csv.gz")
    locked = pd.read_csv(args.feature_dir / "locked_validation_features.csv.gz")
    test = pd.read_csv(args.feature_dir / "test_features.csv.gz")
    folds = pd.read_csv(args.fold_dir / "outer_folds.csv")
    fold_by_uav = folds.set_index(ID_COLUMN)["outer_fold"]
    training_folds = training[ID_COLUMN].map(fold_by_uav)
    if training_folds.isna().any():
        raise ValueError("Training prefixes contain unknown UAV IDs")

    prediction_parts: list[pd.DataFrame] = []
    coefficient_records: list[dict[str, float | int]] = []
    for outer_fold in sorted(folds["outer_fold"].unique()):
        fit_rows = training.loc[training_folds != outer_fold]
        validation_rows = locked.loc[locked["outer_fold"] == outer_fold].copy()
        intercept, slope = weighted_linear_fit(
            fit_rows["feature__flight_cycle"].to_numpy(),
            fit_rows["RUL"].to_numpy(),
            fit_rows["sample_weight"].to_numpy(),
        )
        validation_rows["y_true"] = validation_rows["RUL"]
        validation_rows["y_pred"] = predict(
            intercept, slope, validation_rows["feature__flight_cycle"]
        )
        prediction_parts.append(
            validation_rows[
                [
                    "sample_id",
                    "scenario",
                    "outer_fold",
                    ID_COLUMN,
                    "cutoff",
                    "terminal_lifetime",
                    "lifetime_quantile",
                    "y_true",
                    "y_pred",
                ]
            ]
        )
        coefficient_records.append(
            {"outer_fold": int(outer_fold), "intercept": intercept, "slope": slope}
        )

    output_dir = args.output_dir
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["scenario", "outer_fold", ID_COLUMN]
    )
    paths = [save_csv(predictions, output_dir / "locked_predictions.csv")]
    paths.append(
        save_csv(
            pd.DataFrame.from_records(coefficient_records),
            output_dir / "fold_coefficients.csv",
        )
    )
    paths.extend(
        evaluate_predictions(
            predictions,
            output_dir / "metrics",
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        )
    )

    intercept, slope = weighted_linear_fit(
        training["feature__flight_cycle"].to_numpy(),
        training["RUL"].to_numpy(),
        training["sample_weight"].to_numpy(),
    )
    test_predictions = test[["sample_id", ID_COLUMN, "cutoff"]].copy()
    test_predictions["RUL"] = predict(
        intercept, slope, test["feature__flight_cycle"]
    )
    paths.append(save_csv(test_predictions, output_dir / "test_predictions.csv"))
    paths.append(
        save_json(
            {"intercept": intercept, "slope": slope},
            output_dir / "full_training_coefficients.json",
        )
    )
    print("\n".join(f"Saved {path}" for path in paths))


if __name__ == "__main__":
    main()
