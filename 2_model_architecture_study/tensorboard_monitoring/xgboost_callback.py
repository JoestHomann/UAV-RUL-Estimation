"""Adapt XGBoost's official callback events to the shared training monitor."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

from xgboost.callback import TrainingCallback


class XGBoostProgressCallback(TrainingCallback):
    """Translate XGBoost evaluation history into stable scalar event names."""

    def __init__(self, progress_reporter: Callable[..., bool]) -> None:
        self.progress_reporter = progress_reporter
        self.last_logged_at = perf_counter()
        self.last_logged_step = 0

    @staticmethod
    def _round_scalars(
        evaluations: dict[str, dict[str, list[float]]],
    ) -> dict[str, float]:
        """Map XGBoost evaluation-set names to dashboard scalar tags."""

        scalars: dict[str, float] = {}
        training = evaluations.get("validation_0", {}).get("rmse", [])
        validation = evaluations.get("validation_1", {}).get("rmse", [])
        if training:
            scalars["optimization/training_rmse"] = float(training[-1])
        if validation:
            scalars["optimization/validation_rmse"] = float(validation[-1])
        return scalars

    def _log(
        self,
        *,
        step: int,
        evaluations: dict[str, dict[str, list[float]]],
        force: bool,
    ) -> None:
        """Add average iteration time and apply the shared sampling policy."""

        current_time = perf_counter()
        rounds_since_last_log = max(1, step - self.last_logged_step)
        scalars = self._round_scalars(evaluations)
        scalars["timing/seconds_per_iteration"] = (
            current_time - self.last_logged_at
        ) / rounds_since_last_log
        written = self.progress_reporter(
            step=step,
            scalars=scalars,
            force=force,
        )
        if written:
            self.last_logged_at = current_time
            self.last_logged_step = step

    def after_iteration(
        self,
        model: Any,
        epoch: int,
        evals_log: dict[str, dict[str, list[float]]],
    ) -> bool:
        """Receive the official zero-based XGBoost training iteration."""

        self._log(step=epoch + 1, evaluations=evals_log, force=False)
        return False

    def log_final_round(
        self,
        evaluations: dict[str, dict[str, list[float]]],
    ) -> None:
        """Guarantee that a final round outside the ten-round interval is shown."""

        training = evaluations.get("validation_0", {}).get("rmse", [])
        if training:
            self._log(step=len(training), evaluations=evaluations, force=True)
