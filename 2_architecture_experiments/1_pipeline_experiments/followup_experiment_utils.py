"""Frozen inputs, exact prediction alignment and inner-only routing for PE_25/26."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from advanced_r2_utils import COMMON_KEYS, REPOSITORY_ROOT, equal_uav_weights
from confirmation_utils import evaluation_jobs, validate_partitions


CELL_KEYS = ["split_seed", "source_outer_fold", "evaluation_level", "inner_fold", "method"]
ENDPOINT_KEYS = ["uav_id", "scenario", "cutoff", "observed_rul"]


def input_path(value: str) -> Path:
    path = (REPOSITORY_ROOT / value).resolve()
    path.relative_to(REPOSITORY_ROOT.resolve())
    if not path.is_file():
        raise ValueError(f"Required experiment input is missing: {path}")
    return path


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, table: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def register_run(reporting: Path, workflow: dict, paths: list[Path]) -> dict:
    """Reject changed settings, code or source contents, including with --force."""
    hashes = {}
    for path in sorted(set(path.resolve() for path in paths)):
        with path.open("rb") as stream:
            hashes[str(path.relative_to(REPOSITORY_ROOT))] = hashlib.file_digest(stream, "sha256").hexdigest()
    registration = {"workflow": workflow, "input_sha256": hashes,
        "python_version": sys.version,
        "package_versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost")}}
    reporting.mkdir(parents=True, exist_ok=True)
    destination = reporting / "pre_registration.json"
    if destination.is_file():
        if json.loads(destination.read_text(encoding="utf-8")) != registration:
            raise ValueError("Registered settings, code or inputs changed; use a new pipeline.run directory")
    else:
        if (reporting / "fold_predictions.csv").exists():
            raise ValueError("Cannot trust a checkpoint without its pre-registration")
        atomic_json(destination, registration)
    return registration


def aligned_methods(rows: pd.DataFrame, challenger: str) -> pd.DataFrame:
    """Require identical labelled endpoints; never silently inner-join missing rows."""
    identity = [key for key in COMMON_KEYS if key != "observed_rul"]
    selected = []
    for name in ("control", challenger):
        frame = rows.loc[rows.method.eq(name)].sort_values(identity).reset_index(drop=True)
        if frame.empty or frame.duplicated(identity).any():
            raise ValueError(f"Missing or duplicate endpoints for {name}")
        if not np.isfinite(frame[["cutoff", "observed_rul", "predicted_rul"]].to_numpy(float)).all():
            raise ValueError(f"Non-finite predictions or labels for {name}")
        selected.append(frame)
    control, candidate = selected
    if not control[COMMON_KEYS].equals(candidate[COMMON_KEYS]):
        raise ValueError("Control and challenger endpoints/labels do not align")
    result = control[COMMON_KEYS].copy()
    result["control_prediction"] = control.predicted_rul.to_numpy(float)
    result["challenger_prediction"] = candidate.predicted_rul.to_numpy(float)
    return result


def validate_saved_nested(
    table: pd.DataFrame, outer: pd.DataFrame, inner: pd.DataFrame,
    *, methods: set[str], outer_count: int, inner_count: int,
) -> None:
    """Check every saved method/cell against the declared nested UAV partitions."""
    validate_partitions(outer, inner, expected_outer_folds=outer_count, expected_inner_folds=inner_count)
    if set(inner.split_seed) != set(outer.split_seed):
        raise ValueError("Inner/outer split seeds differ")
    expected_outer_keys = set(map(tuple, outer[["split_seed", "outer_fold"]].drop_duplicates().to_numpy()))
    if set(map(tuple, inner[["split_seed", "outer_fold"]].drop_duplicates().to_numpy())) != expected_outer_keys:
        raise ValueError("Inner/outer fold identities differ")
    if table.duplicated(CELL_KEYS + ["uav_id", "scenario", "cutoff"]).any():
        raise ValueError("Nested checkpoint contains duplicate endpoints")
    if not np.isfinite(table[["cutoff", "observed_rul", "predicted_rul"]].to_numpy(float)).all():
        raise ValueError("Nested checkpoint contains non-finite values")
    jobs = evaluation_jobs(outer, inner, include_inner=True)
    expected_cells = {
        (job.split_seed, job.outer_fold, job.evaluation_level, job.inner_fold, method)
        for job in jobs for method in methods
    }
    actual = set(map(tuple, table[CELL_KEYS].drop_duplicates().to_numpy()))
    if actual != expected_cells:
        raise ValueError("Nested checkpoint has incomplete or unexpected model cells")
    # The outer control table supplies the exact endpoints for each UAV/seed.
    reference = table.loc[table.evaluation_level.eq("outer") & table.method.eq("control")]
    for job in jobs:
        endpoints = reference.loc[
            reference.split_seed.eq(job.split_seed) & reference.uav_id.isin(job.validation_uavs), ENDPOINT_KEYS
        ].sort_values(ENDPOINT_KEYS).reset_index(drop=True)
        if set(endpoints.uav_id) != set(job.validation_uavs):
            raise ValueError("Outer checkpoint does not cover every held UAV")
        for method in methods:
            rows = cell_rows(table, job, method)
            if set(rows.outer_fold) != {job.report_fold}:
                raise ValueError("Checkpoint report fold does not match its split")
            actual_endpoints = rows[ENDPOINT_KEYS].sort_values(ENDPOINT_KEYS).reset_index(drop=True)
            if not actual_endpoints.equals(endpoints):
                raise ValueError("Nested endpoint coverage, labels or held UAV membership changed")


def cell_rows(table: pd.DataFrame, job: Any, method: str) -> pd.DataFrame:
    if table.empty:
        return table
    return table.loc[
        table.split_seed.eq(job.split_seed)
        & table.source_outer_fold.eq(job.outer_fold)
        & table.evaluation_level.eq(job.evaluation_level)
        & table.inner_fold.eq(job.inner_fold)
        & table.method.eq(method)
    ]


def complete_cell(table: pd.DataFrame, job: Any, method: str, expected: pd.DataFrame) -> bool:
    rows = cell_rows(table, job, method)
    if rows.empty:
        return False
    identity = ["outer_fold", *ENDPOINT_KEYS]
    if rows.duplicated(["uav_id", "scenario", "cutoff"]).any():
        raise ValueError("Checkpoint contains duplicate endpoints")
    if not np.isfinite(rows.predicted_rul.to_numpy(float)).all():
        raise ValueError("Checkpoint contains non-finite predictions")
    if not rows[identity].sort_values(identity).reset_index(drop=True).equals(
        expected[identity].sort_values(identity).reset_index(drop=True)
    ):
        raise ValueError("Checkpoint endpoints differ from the current validation dataset")
    return True


def route_predictions(rows: pd.DataFrame, *, weight: float, threshold: float, route_column: str) -> np.ndarray:
    """Use observable prediction/history only; labels are intentionally never read."""
    if route_column not in {"control_prediction", "cutoff"}:
        raise ValueError("Routing must use control prediction or observed history length")
    if not np.isfinite([weight, threshold]).all() or not 0 <= weight <= 1 or threshold <= 0:
        raise ValueError("Invalid routing weight or threshold")
    values = rows[[route_column, "control_prediction", "challenger_prediction"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Routing inputs must be finite")
    applied = weight * (rows[route_column].to_numpy(float) <= threshold)
    return rows.control_prediction.to_numpy(float) + applied * (
        rows.challenger_prediction.to_numpy(float) - rows.control_prediction.to_numpy(float)
    )


def select_routed_blend(
    outer_rows: pd.DataFrame, inner_rows: pd.DataFrame,
    *, weights: list[float], thresholds: list[float], route_column: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Pick one bounded recipe from training-side OOF predictions per outer fold."""
    weights = sorted(set(float(value) for value in weights))
    thresholds = sorted(set(float(value) for value in thresholds))
    if not weights or 0.0 not in weights or not thresholds:
        raise ValueError("Routing grid needs a zero-weight control and at least one threshold")
    if set(outer_rows.outer_fold) != set(inner_rows.outer_fold):
        raise ValueError("Inner and outer report folds differ")
    predictions = np.full(len(outer_rows), np.nan)
    provenance = []
    for fold in sorted(outer_rows.outer_fold.unique()):
        held_positions = np.flatnonzero(outer_rows.outer_fold.to_numpy() == fold)
        held = outer_rows.iloc[held_positions]
        training = inner_rows.loc[inner_rows.outer_fold.eq(fold)]
        if set(held.uav_id) & set(training.uav_id):
            raise ValueError("Routing selection includes outer-held UAVs")
        if training.groupby("uav_id").inner_fold.nunique().max() != 1:
            raise ValueError("An inner UAV occurs in multiple held folds")
        if not np.isfinite(training.observed_rul.to_numpy(float)).all():
            raise ValueError("Inner targets must be finite")
        sample_weights = equal_uav_weights(training)
        candidates = []
        for weight in weights:
            for threshold in thresholds:
                estimate = route_predictions(training, weight=weight, threshold=threshold, route_column=route_column)
                mse = float(np.average((estimate - training.observed_rul.to_numpy(float)) ** 2, weights=sample_weights))
                candidates.append((mse, weight, threshold))
        mse, weight, threshold = min(candidates)
        predictions[held_positions] = route_predictions(held, weight=weight, threshold=threshold, route_column=route_column)
        provenance.append({
            "outer_fold": int(fold), "selected_weight": weight, "selected_threshold": threshold,
            "route_column": route_column, "inner_rmse": float(np.sqrt(mse)),
            "training_uavs": int(training.uav_id.nunique()), "validation_uavs": int(held.uav_id.nunique()),
            "uav_overlap": 0, "selection_uses_outer_labels": False,
        })
    return predictions, pd.DataFrame(provenance)
