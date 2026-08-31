"""Verify causal trajectory construction and fold-safe references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from trajectory_data_adapter import (
    DEFAULT_MANIFEST_PATH,
    TrajectoryAdapterError,
    TrajectoryDataAdapter,
    TrajectoryDataset,
)


STEP_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT_PATH = STEP_DIR / "artifacts" / "trajectory_verification.json"


class TrajectoryVerificationError(ValueError):
    """Represent a failed runtime trajectory contract check."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrajectoryVerificationError(message)


def _verify_cutoffs(dataset: TrajectoryDataset) -> None:
    cutoffs = dataset.metadata["cutoff"].to_numpy(dtype=np.int64)
    for cycles, cutoff in zip(dataset.cycles, cutoffs, strict=True):
        _require(int(cycles[-1]) == int(cutoff), "Trajectory exceeds its cutoff")


def verify_trajectory_data_adapter(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Exercise one real inner split and the unlabelled test interface."""

    try:
        adapter = TrajectoryDataAdapter(manifest_path)
        outer_fold = adapter.outer_fold_labels()[0]
        inner_fold = adapter.inner_fold_labels(outer_fold)[0]
        split = adapter.get_inner_selection_split(outer_fold, inner_fold)
        test = adapter.load_test()
    except (TrajectoryAdapterError, IndexError) as error:
        raise TrajectoryVerificationError(str(error)) from error

    training_ids = set(split.training.metadata["uav_id"].astype(str))
    validation_ids = set(split.validation.metadata["uav_id"].astype(str))
    _require(not training_ids & validation_ids, "Training and validation UAVs overlap")
    references = split.training.reference_library
    validation_references = split.validation.reference_library
    _require(references is not None, "Training reference library is missing")
    _require(validation_references is not None, "Validation references are missing")
    reference_ids = set(references.metadata["uav_id"].astype(str))
    _require(reference_ids == training_ids, "References do not match training UAVs")
    _require(
        not reference_ids & validation_ids,
        "Validation UAV entered the training reference library",
    )
    _require(split.training.scaled, "Training trajectories are not scaled")
    _require(split.validation.scaled, "Validation trajectories are not scaled")
    _require(references.scaled, "Reference trajectories are not scaled")
    _verify_cutoffs(split.training)
    _verify_cutoffs(split.validation)
    _verify_cutoffs(test)
    _require(test.target is None, "Test trajectory data unexpectedly has targets")

    for dataset in (split.training, split.validation, test):
        for trajectory in dataset.trajectories:
            _require(np.isfinite(trajectory).all(), "Trajectory contains non-finite values")

    return {
        "status": "passed",
        "settings_version": adapter.manifest.get("settings_version"),
        "outer_fold_checked": outer_fold,
        "inner_fold_checked": inner_fold,
        "training_queries_checked": len(split.training),
        "validation_queries_checked": len(split.validation),
        "reference_trajectories_checked": len(references),
        "test_queries_checked": len(test),
        "checks": {
            "causal_cutoffs": True,
            "fold_disjoint_uavs": True,
            "training_only_references": True,
            "training_only_channel_scaling": True,
            "finite_values": True,
            "unlabelled_test_interface": True,
        },
    }


def _write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    try:
        report = verify_trajectory_data_adapter(args.manifest)
        _write_report(report, args.report)
    except TrajectoryVerificationError as error:
        print(f"Trajectory adapter verification failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("Trajectory data adapter verification passed")
    print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
