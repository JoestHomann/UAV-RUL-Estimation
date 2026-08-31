"""Verify the trajectory DTW-kNN adapter on one real inner split."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


STEP_DIR = Path(__file__).resolve().parent
PHASE_DIR = STEP_DIR.parent
for dependency_dir in (
    PHASE_DIR / "3_trajectory_data_adapter",
    STEP_DIR,
):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from base import load_model_adapter  # noqa: E402
from model_registry import ModelAdapterFactory  # noqa: E402
from trajectory_data_adapter import TrajectoryDataAdapter  # noqa: E402


HYPERPARAMETERS = {
    "neighbors": 5,
    "reference_pool_size": 20,
    "max_points": 48,
    "warping_window": 8,
    "distance_power": 1.0,
}


def main() -> None:
    adapter = TrajectoryDataAdapter()
    split = adapter.get_inner_selection_split(0, 0)
    model = ModelAdapterFactory().create(
        "trajectory_dtw_knn",
        HYPERPARAMETERS,
        seed=13,
    )
    summary = model.fit(split.training, split.validation)
    predictions = model.predict(split.validation)
    if len(predictions) != len(split.validation):
        raise RuntimeError("Trajectory DTW-kNN prediction length changed")
    if not np.isfinite(predictions).all():
        raise RuntimeError("Trajectory DTW-kNN produced non-finite predictions")
    path = STEP_DIR / ".trajectory_dtw_knn_verification.joblib"
    try:
        model.save(path)
        restored = load_model_adapter(path)
        restored_predictions = restored.predict(split.validation)
    finally:
        path.unlink(missing_ok=True)
    if not np.allclose(predictions, restored_predictions, rtol=1e-10, atol=1e-10):
        raise RuntimeError("Reloaded trajectory model changed predictions")
    print("Trajectory DTW-kNN verification passed")
    print(f"Training queries: {summary.training_rows}")
    print(f"Validation queries: {summary.validation_rows}")
    print(f"Validation RMSE: {summary.best_validation_rmse:.6f}")


if __name__ == "__main__":
    main()
