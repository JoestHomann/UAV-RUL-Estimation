"""Verify the GRU adapter on one real grouped inner split."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


STEP_DIR = Path(__file__).resolve().parent
PHASE_DIR = STEP_DIR.parent
for dependency_dir in (PHASE_DIR / "3_sequence_data_adapter", STEP_DIR):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from base import load_model_adapter  # noqa: E402
from models.neural.gru import GRUAdapter  # noqa: E402
from models.neural.neural_base import NeuralTrainingConfig  # noqa: E402
from sequence_data_adapter import SequenceDataAdapter  # noqa: E402


HYPERPARAMETERS = {
    "layers": 1,
    "hidden_units": 16,
    "direction": "unidirectional",
    "dropout": 0.0,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
}


class _VerificationMonitor:
    def log_training_step(self, **_: object) -> bool:
        return False


def main() -> None:
    adapter = SequenceDataAdapter()
    lookback = min(adapter.lookbacks)
    split = adapter.get_inner_selection_split(0, 0, lookback)
    model = GRUAdapter(
        hyperparameters=HYPERPARAMETERS,
        seed=13,
        training_config=NeuralTrainingConfig(
            batch_size=128,
            maximum_epochs=2,
            early_stopping_patience=1,
            gradient_clip_global_norm=1.0,
        ),
        training_epochs=2,
        training_monitor=_VerificationMonitor(),
    )
    summary = model.fit(split.training, split.validation)
    predictions = model.predict(split.validation)
    if len(predictions) != len(split.validation) or not np.isfinite(predictions).all():
        raise RuntimeError("GRU prediction contract failed")

    path = STEP_DIR / ".gru_verification.joblib"
    try:
        model.save(path)
        restored = load_model_adapter(path)
        restored_predictions = restored.predict(split.validation)
    finally:
        path.unlink(missing_ok=True)
    if not np.allclose(predictions, restored_predictions, rtol=1e-7, atol=1e-7):
        raise RuntimeError("Reloaded GRU changed predictions")

    replay = GRUAdapter(
        hyperparameters=HYPERPARAMETERS,
        seed=13,
        training_config=model.training_config,
        training_epochs=2,
        training_monitor=_VerificationMonitor(),
    )
    replay.fit(split.training, split.validation)
    replay_predictions = replay.predict(split.validation)
    if not np.allclose(predictions, replay_predictions, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Same-seed GRU replay changed predictions")
    print("GRU verification passed")
    print(f"Lookback: {lookback}")
    print(f"Training rows: {summary.training_rows}")
    print(f"Validation rows: {summary.validation_rows}")


if __name__ == "__main__":
    main()
