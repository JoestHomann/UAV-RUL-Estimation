"""Smoke-test censored and horizon RUL adapters, including persistence."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STEP_DIR.parent / "2_tabular_data_adapter"))

from no_op_training_monitor import NoOpTrainingMonitor  # noqa: E402
from models.tabular.censored_rul import (  # noqa: E402
    HorizonXGBoostAdapter,
    XGBoostAFTAdapter,
)
from tabular_data_adapter import TabularDataset  # noqa: E402


def _dataset() -> TabularDataset:
    rng = np.random.default_rng(13)
    rows = 90
    target = np.linspace(2.0, 180.0, rows)
    features = pd.DataFrame(
        {
            "feature__health": target + rng.normal(0.0, 2.0, rows),
            "feature__age": np.arange(rows, dtype=float),
            "feature__noise": rng.normal(size=rows),
        }
    )
    metadata = pd.DataFrame(
        {"uav_id": [f"UAV_{index:04d}" for index in range(rows)], "cutoff": np.arange(rows) + 1}
    )
    return TabularDataset(
        features,
        metadata,
        pd.Series(target),
        pd.Series(np.ones(rows)),
        None,
    )


def main() -> None:
    data = _dataset()
    shared = {
        "maximum_trees": 8,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_child_weight": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 1.0e-4,
        "reg_lambda": 1.0,
    }
    adapters = [
        XGBoostAFTAdapter(
            hyperparameters={
                **shared,
                "aft_loss_distribution": "normal",
                "aft_loss_distribution_scale": 1.0,
                "censoring_threshold": 125.0,
            },
            seed=13,
            early_stopping_patience=3,
            training_monitor=NoOpTrainingMonitor(),
            device="cpu",
        ),
        HorizonXGBoostAdapter(
            hyperparameters={**shared, "horizons": "10,25,50,75,100,125"},
            seed=13,
            training_monitor=NoOpTrainingMonitor(),
        ),
    ]
    for adapter in adapters:
        adapter.fit(data, data)
        prediction = adapter.predict(data)
        if prediction.shape != (len(data),) or not np.isfinite(prediction).all():
            raise AssertionError(f"{adapter.family} produced invalid predictions")
        artifact = STEP_DIR / f".{adapter.family}_verification.joblib"
        adapter.detach_training_monitor()
        adapter.save(artifact)
        restored = type(adapter).load(artifact)
        if not np.allclose(prediction, restored.predict(data), rtol=0.0, atol=1.0e-10):
            raise AssertionError(f"{adapter.family} persistence changed predictions")
        artifact.unlink()
    print("Censored and horizon RUL adapter verification passed")


if __name__ == "__main__":
    main()

