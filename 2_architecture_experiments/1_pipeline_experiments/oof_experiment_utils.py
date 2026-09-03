"""Shared metrics and artifact helpers for development-only OOF experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ALIGNMENT_KEYS = [
    "outer_fold",
    "inner_fold",
    "validation_row",
    "uav_id",
    "scenario",
    "cutoff",
    "observed_rul",
]


def regression_metrics(observed: Any, predicted: Any) -> dict[str, float]:
    """Return the accuracy and one-sided safety metrics used by the studies."""

    truth = np.asarray(observed, dtype=np.float64)
    estimate = np.asarray(predicted, dtype=np.float64)
    residual = estimate - truth
    positive = np.maximum(residual, 0.0)
    denominator = float(np.square(truth - truth.mean()).sum())
    return {
        "r2": 1.0 - float(np.square(residual).sum()) / denominator
        if denominator > 0.0
        else float("nan"),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "overprediction_rate": float(np.mean(residual > 0.0)),
        "rms_overprediction": float(np.sqrt(np.mean(np.square(positive)))),
        "overprediction_q95": float(np.quantile(positive, 0.95)),
        "maximum_overprediction": float(positive.max(initial=0.0)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one deterministic JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_prediction_table(table: pd.DataFrame, *, prediction: str) -> None:
    """Reject incomplete, duplicate, or non-finite OOF prediction tables."""

    required = {*ALIGNMENT_KEYS, prediction}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    if table.duplicated(ALIGNMENT_KEYS).any():
        raise ValueError("Prediction table contains duplicate OOF alignment keys")
    numeric = table[["observed_rul", prediction]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Prediction table contains non-finite targets or predictions")

