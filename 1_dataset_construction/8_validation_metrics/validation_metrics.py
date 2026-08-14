"""Regression metrics for grouped, test-like UAV validation scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PHASE_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from common import STEP_8_ARTIFACT_DIR, ID_COLUMN, age_band, save_csv, save_json


def metric_specification() -> dict[str, Any]:
    return {
        "prediction_unit": "one UAV prefix per validation scenario",
        "metrics": {
            "r2": "1 - sum((y_true-y_pred)^2) / sum((y_true-mean(y_true))^2)",
            "rmse": "sqrt(mean((y_pred-y_true)^2))",
            "mae": "mean(abs(y_pred-y_true))",
            "bias": "mean(y_pred-y_true)",
        },
        "reported_groups": [
            "overall",
            "scenario",
            "outer_fold",
            "age_band",
            "lifetime_quantile",
        ],
        "uncertainty": "95% UAV-bootstrap interval for R2",
    }


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum(np.square(y_true - np.mean(y_true))))
    if denominator <= 0:
        return float("nan")
    return 1.0 - float(np.sum(np.square(y_true - y_pred))) / denominator


def regression_metrics(table: pd.DataFrame) -> dict[str, Any]:
    y_true = table["y_true"].to_numpy(dtype=float)
    y_pred = table["y_pred"].to_numpy(dtype=float)
    residual = y_pred - y_true
    return {
        "rows": int(len(table)),
        "uavs": int(table[ID_COLUMN].nunique()),
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


def grouped_metrics(table: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouper: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else group_columns
    )
    for keys, group in table.groupby(grouper, observed=True, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys, strict=True))
        record.update(regression_metrics(group))
        records.append(record)
    return pd.DataFrame.from_records(records)


def uav_bootstrap_r2(
    table: pd.DataFrame, *, repetitions: int, seed: int
) -> dict[str, float | int]:
    grouped = {uav: group for uav, group in table.groupby(ID_COLUMN, sort=True)}
    uavs = np.array(sorted(grouped), dtype=object)
    rng = np.random.default_rng(seed)
    scores = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled = rng.choice(uavs, size=len(uavs), replace=True)
        bootstrap = pd.concat([grouped[uav] for uav in sampled], ignore_index=True)
        scores[index] = regression_metrics(bootstrap)["r2"]
    finite = scores[np.isfinite(scores)]
    return {
        "repetitions": int(repetitions),
        "seed": int(seed),
        "r2_median": float(np.median(finite)),
        "r2_ci_lower_95": float(np.quantile(finite, 0.025)),
        "r2_ci_upper_95": float(np.quantile(finite, 0.975)),
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    output_dir: Path,
    *,
    bootstrap_repetitions: int = 1000,
    seed: int = 20260814,
) -> list[Path]:
    required = {
        "sample_id",
        "scenario",
        "outer_fold",
        ID_COLUMN,
        "cutoff",
        "terminal_lifetime",
        "lifetime_quantile",
        "y_true",
        "y_pred",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing columns: {missing}")
    if predictions.duplicated(["scenario", ID_COLUMN]).any():
        raise ValueError("Predictions contain duplicate scenario/UAV rows")
    numeric = predictions[["y_true", "y_pred"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Predictions contain missing or non-finite values")

    evaluated = predictions.copy()
    evaluated["age_band"] = age_band(evaluated["cutoff"])
    overall = regression_metrics(evaluated)
    overall["bootstrap"] = uav_bootstrap_r2(
        evaluated, repetitions=bootstrap_repetitions, seed=seed
    )
    paths = [save_json(overall, output_dir / "overall_metrics.json")]
    paths.append(
        save_csv(grouped_metrics(evaluated, ["scenario"]), output_dir / "scenario_metrics.csv")
    )
    paths.append(
        save_csv(grouped_metrics(evaluated, ["outer_fold"]), output_dir / "fold_metrics.csv")
    )
    paths.append(
        save_csv(grouped_metrics(evaluated, ["age_band"]), output_dir / "age_band_metrics.csv")
    )
    paths.append(
        save_csv(
            grouped_metrics(evaluated, ["lifetime_quantile"]),
            output_dir / "lifetime_quantile_metrics.csv",
        )
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_csv", type=Path, nargs="?")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STEP_8_ARTIFACT_DIR,
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if args.predictions_csv is None:
        path = save_json(metric_specification(), args.output_dir / "metric_specification.json")
        print(f"Saved {path}")
        return
    predictions = pd.read_csv(args.predictions_csv)
    paths = evaluate_predictions(
        predictions,
        args.output_dir,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    print("\n".join(f"Saved {path}" for path in paths))


if __name__ == "__main__":
    main()
