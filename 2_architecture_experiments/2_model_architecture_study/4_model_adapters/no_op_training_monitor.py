"""Provide the complete training-monitor contract when live logging is disabled."""

from __future__ import annotations

from typing import Any, Callable


class NoOpTrainingMonitor:
    """Accept model progress hooks without writing monitoring artifacts."""

    def log_training_step(self, **_: object) -> bool:
        return False

    def create_xgboost_callback(
        self,
        progress_reporter: Callable[..., bool],
    ) -> Any:
        # XGBoost requires every callbacks entry to implement its callback API,
        # even when the surrounding workflow deliberately disables logging.
        from xgboost.callback import TrainingCallback

        class _NoOpXGBoostCallback(TrainingCallback):
            def log_final_round(self, evaluations: object) -> None:
                del evaluations

        del progress_reporter
        return _NoOpXGBoostCallback()
