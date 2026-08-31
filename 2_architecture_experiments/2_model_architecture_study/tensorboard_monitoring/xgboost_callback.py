"""Adapt XGBoost's official callback events to the shared training monitor."""

from __future__ import annotations

from typing import Any, Callable

from xgboost.callback import TrainingCallback


class XGBoostProgressCallback(TrainingCallback):
    """Translate XGBoost evaluation history into the two shared curve tags."""

    def __init__(self, progress_reporter: Callable[..., bool]) -> None:
        self.progress_reporter = progress_reporter

    @staticmethod
    def _round_scalars(
        evaluations: dict[str, dict[str, list[float]]],
    ) -> dict[str, float]:
        """Map XGBoost evaluation-set names to the shared curve tags.

        The first evaluation set is the training data and the second, when
        present, is the inner validation fold that drives early stopping.
        """

        scalars: dict[str, float] = {}
        training = evaluations.get("validation_0", {}).get("rmse", [])
        validation = evaluations.get("validation_1", {}).get("rmse", [])
        if training:
            scalars["train/loss"] = float(training[-1])
        if validation:
            scalars["val/rmse"] = float(validation[-1])
        return scalars

    def after_iteration(
        self,
        model: Any,
        epoch: int,
        evals_log: dict[str, dict[str, list[float]]],
    ) -> bool:
        """Receive the official zero-based XGBoost training iteration."""

        self.progress_reporter(
            step=epoch + 1,
            scalars=self._round_scalars(evals_log),
            force=False,
        )
        return False

    def log_final_round(
        self,
        evaluations: dict[str, dict[str, list[float]]],
    ) -> None:
        """Guarantee that a final round outside the ten-round interval is shown."""

        training = evaluations.get("validation_0", {}).get("rmse", [])
        if not training:
            return
        self.progress_reporter(
            step=len(training),
            scalars=self._round_scalars(evaluations),
            force=True,
        )
