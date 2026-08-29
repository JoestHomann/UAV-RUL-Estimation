"""Validate and apply shared RUL target, loss, and prediction policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


class PolicyError(ValueError):
    """Explain an invalid or unsupported model policy."""


@dataclass(frozen=True)
class TargetPolicy:
    mode: str = "raw"
    maximum_rul: float | None = None

    @classmethod
    def from_settings(cls, value: dict[str, Any]) -> "TargetPolicy":
        mode = value.get("mode")
        maximum = value.get("maximum_rul")
        if mode not in {"raw", "piecewise_cap"}:
            raise PolicyError(f"Unknown target mode {mode!r}")
        if mode == "raw" and maximum is not None:
            raise PolicyError("Raw target policy cannot define maximum_rul")
        if mode == "piecewise_cap":
            if maximum is None or not np.isfinite(maximum) or float(maximum) <= 0:
                raise PolicyError("Piecewise target cap must be finite and positive")
            maximum = float(maximum)
        return cls(mode=mode, maximum_rul=maximum)

    def transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        result = np.asarray(values, dtype=np.float64).copy()
        if self.mode == "piecewise_cap":
            assert self.maximum_rul is not None
            result = np.minimum(result, self.maximum_rul)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "maximum_rul": self.maximum_rul}


@dataclass(frozen=True)
class PredictionPolicy:
    loss: str = "symmetric_rmse"
    overprediction_weight: float = 1.0
    quantile: float = 0.5
    calibration: str = "none"
    safety_offset: float = 0.0
    non_overprediction_coverage: float = 0.5

    @classmethod
    def from_settings(cls, value: dict[str, Any]) -> "PredictionPolicy":
        policy = cls(
            loss=str(value.get("loss")),
            overprediction_weight=float(value.get("overprediction_weight")),
            quantile=float(value.get("quantile")),
            calibration=str(value.get("calibration")),
            safety_offset=float(value.get("safety_offset")),
            non_overprediction_coverage=float(
                value.get("non_overprediction_coverage")
            ),
        )
        if policy.loss not in {"symmetric_rmse", "asymmetric_mse", "quantile"}:
            raise PolicyError(f"Unknown loss policy {policy.loss!r}")
        if policy.overprediction_weight < 1.0:
            raise PolicyError("Overprediction weight cannot be below one")
        if not 0.0 < policy.quantile <= 0.5:
            raise PolicyError("Conservative quantile must be in (0, 0.5]")
        if policy.calibration not in {"none", "fixed_offset"}:
            raise PolicyError(f"Unknown calibration policy {policy.calibration!r}")
        if policy.safety_offset < 0.0:
            raise PolicyError("Safety offset cannot be negative")
        if not 0.0 < policy.non_overprediction_coverage < 1.0:
            raise PolicyError("Calibration coverage must be between zero and one")
        return policy

    def adjust_predictions(
        self,
        predictions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        values = np.asarray(predictions, dtype=np.float64)
        if self.calibration == "fixed_offset":
            return values - self.safety_offset
        return values.copy()

    def numpy_losses(
        self,
        targets: NDArray[np.float64],
        predictions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        residual = np.asarray(predictions) - np.asarray(targets)
        if self.loss == "asymmetric_mse":
            weights = np.where(residual > 0.0, self.overprediction_weight, 1.0)
            return weights * np.square(residual)
        if self.loss == "quantile":
            return np.where(
                residual > 0.0,
                (1.0 - self.quantile) * residual,
                self.quantile * -residual,
            )
        return np.square(residual)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "overprediction_weight": self.overprediction_weight,
            "quantile": self.quantile,
            "calibration": self.calibration,
            "safety_offset": self.safety_offset,
            "non_overprediction_coverage": self.non_overprediction_coverage,
        }


TARGET_CAPABLE_FAMILIES = {
    "mean_baseline",
    "cycle_only_baseline",
    "regularized_linear",
    "random_forest",
    "extra_trees",
    "xgboost",
    "catboost",
    "mlp",
    "tcn",
    "multiscale_cnn",
    "sensor_graph_tcn",
    "lstm",
    "transformer",
    "rbf_svr",
}

LOSS_CAPABILITIES = {
    "xgboost": {"symmetric_rmse", "asymmetric_mse", "quantile"},
    "catboost": {"symmetric_rmse", "quantile"},
    "mlp": {"symmetric_rmse", "asymmetric_mse", "quantile"},
    "tcn": {"symmetric_rmse", "asymmetric_mse", "quantile"},
    "multiscale_cnn": {"symmetric_rmse", "asymmetric_mse", "quantile"},
    "sensor_graph_tcn": {"symmetric_rmse", "asymmetric_mse", "quantile"},
    "lstm": {"symmetric_rmse", "asymmetric_mse", "quantile"},
    "transformer": {"symmetric_rmse", "asymmetric_mse", "quantile"},
}


def verify_family_policies(
    family: str,
    target: TargetPolicy,
    prediction: PredictionPolicy,
) -> None:
    if target.mode != "raw" and family not in TARGET_CAPABLE_FAMILIES:
        raise PolicyError(f"{family} does not support transformed RUL targets")
    supported_losses = LOSS_CAPABILITIES.get(family, {"symmetric_rmse"})
    if prediction.loss not in supported_losses:
        raise PolicyError(
            f"{family} does not support {prediction.loss}; "
            f"supported losses are {sorted(supported_losses)}"
        )


def calibrated_safety_offset(
    targets: NDArray[np.float64],
    predictions: NDArray[np.float64],
    coverage: float,
) -> float:
    """Return the nonnegative one-sided offset for a requested coverage."""

    if not 0.0 < coverage < 1.0:
        raise PolicyError("Calibration coverage must be between zero and one")
    residual = np.asarray(predictions, dtype=np.float64) - np.asarray(
        targets,
        dtype=np.float64,
    )
    if residual.shape != np.asarray(targets).shape or residual.size == 0:
        raise PolicyError("Calibration targets and predictions must align")
    return max(0.0, float(np.quantile(residual, coverage)))
