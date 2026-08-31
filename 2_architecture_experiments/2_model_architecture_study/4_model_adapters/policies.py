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
        if mode not in {"raw", "piecewise_cap", "failure_cycle"}:
            raise PolicyError(f"Unknown target mode {mode!r}")
        if mode in {"raw", "failure_cycle"} and maximum is not None:
            raise PolicyError(f"{mode} target policy cannot define maximum_rul")
        if mode == "piecewise_cap":
            if maximum is None or not np.isfinite(maximum) or float(maximum) <= 0:
                raise PolicyError("Piecewise target cap must be finite and positive")
            maximum = float(maximum)
        return cls(mode=mode, maximum_rul=maximum)

    def transform(
        self,
        values: NDArray[np.float64],
        cutoffs: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        result = np.asarray(values, dtype=np.float64).copy()
        if self.mode == "piecewise_cap":
            assert self.maximum_rul is not None
            result = np.minimum(result, self.maximum_rul)
        elif self.mode == "failure_cycle":
            result = result + self._validated_cutoffs(cutoffs, result)
        return result

    def inverse_predictions(
        self,
        predictions: NDArray[np.float64],
        cutoffs: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return raw-RUL predictions for evaluation and downstream use."""

        result = np.asarray(predictions, dtype=np.float64).copy()
        if self.mode == "failure_cycle":
            result = result - self._validated_cutoffs(cutoffs, result)
        return result

    @staticmethod
    def _validated_cutoffs(
        cutoffs: NDArray[np.float64] | None,
        reference: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if cutoffs is None:
            raise PolicyError("failure_cycle target mode requires endpoint cutoffs")
        values = np.asarray(cutoffs, dtype=np.float64).reshape(-1)
        if values.shape != reference.reshape(-1).shape:
            raise PolicyError("Endpoint cutoffs do not align with target rows")
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise PolicyError("Endpoint cutoffs must be finite and nonnegative")
        return values

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "maximum_rul": self.maximum_rul}


@dataclass(frozen=True)
class PredictionPolicy:
    loss: str = "symmetric_rmse"
    overprediction_weight: float = 1.0
    quantile: float = 0.5
    severity_scale: float = 10.0
    calibration: str = "none"
    safety_offset: float = 0.0
    non_overprediction_coverage: float = 0.5
    calibration_prediction_bin_edges: tuple[float, ...] = (
        0.0,
        25.0,
        50.0,
        75.0,
        100.0,
        125.0,
        150.0,
    )
    calibration_minimum_bin_rows: int = 10

    @classmethod
    def from_settings(cls, value: dict[str, Any]) -> "PredictionPolicy":
        policy = cls(
            loss=str(value.get("loss")),
            overprediction_weight=float(value.get("overprediction_weight")),
            quantile=float(value.get("quantile")),
            severity_scale=float(value.get("severity_scale", 10.0)),
            calibration=str(value.get("calibration")),
            safety_offset=float(value.get("safety_offset")),
            non_overprediction_coverage=float(
                value.get("non_overprediction_coverage")
            ),
            calibration_prediction_bin_edges=tuple(
                float(edge)
                for edge in value.get(
                    "calibration_prediction_bin_edges",
                    [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0],
                )
            ),
            calibration_minimum_bin_rows=int(
                value.get("calibration_minimum_bin_rows", 10)
            ),
        )
        if policy.loss not in {
            "symmetric_rmse",
            "asymmetric_mse",
            "severity_asymmetric_mse",
            "quantile",
        }:
            raise PolicyError(f"Unknown loss policy {policy.loss!r}")
        if policy.overprediction_weight < 1.0:
            raise PolicyError("Overprediction weight cannot be below one")
        if (
            policy.loss == "severity_asymmetric_mse"
            and policy.overprediction_weight <= 1.0
        ):
            raise PolicyError(
                "Severity-asymmetric loss requires overprediction weight above one"
            )
        if policy.severity_scale <= 0.0:
            raise PolicyError("Severity scale must be positive")
        if not 0.0 < policy.quantile <= 0.5:
            raise PolicyError("Conservative quantile must be in (0, 0.5]")
        if policy.calibration not in {
            "none",
            "fixed_offset",
            "conditional_quantile",
        }:
            raise PolicyError(f"Unknown calibration policy {policy.calibration!r}")
        if policy.safety_offset < 0.0:
            raise PolicyError("Safety offset cannot be negative")
        if not 0.0 < policy.non_overprediction_coverage < 1.0:
            raise PolicyError("Calibration coverage must be between zero and one")
        edges = policy.calibration_prediction_bin_edges
        if len(edges) < 3 or any(not np.isfinite(edge) for edge in edges):
            raise PolicyError("Calibration bin edges must contain at least three finite values")
        if any(upper <= lower for lower, upper in zip(edges[:-1], edges[1:], strict=True)):
            raise PolicyError("Calibration bin edges must be strictly increasing")
        if policy.calibration_minimum_bin_rows <= 0:
            raise PolicyError("Calibration minimum bin rows must be positive")
        if policy.calibration == "conditional_quantile":
            if policy.safety_offset != 0.0:
                raise PolicyError(
                    "Conditional-quantile calibration requires safety_offset = 0"
                )
            if policy.non_overprediction_coverage < 0.5:
                raise PolicyError(
                    "Conditional-quantile calibration coverage must be at least 0.5"
                )
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
        if self.loss == "severity_asymmetric_mse":
            positive = np.maximum(residual, 0.0)
            severity = (self.overprediction_weight - 1.0) / self.severity_scale
            return np.square(residual) + severity * np.power(positive, 3.0)
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
            "severity_scale": self.severity_scale,
            "calibration": self.calibration,
            "safety_offset": self.safety_offset,
            "non_overprediction_coverage": self.non_overprediction_coverage,
            "calibration_prediction_bin_edges": list(
                self.calibration_prediction_bin_edges
            ),
            "calibration_minimum_bin_rows": self.calibration_minimum_bin_rows,
        }


@dataclass(frozen=True)
class ConditionalQuantileCalibrator:
    """Store one subtraction-only correction curve fitted from OOF residuals."""

    quantile: float
    prediction_bin_edges: tuple[float, ...]
    minimum_bin_rows: int
    corrections: tuple[float, ...]
    training_rows_by_bin: tuple[int, ...]

    @classmethod
    def fit(
        cls,
        targets: NDArray[np.float64],
        predictions: NDArray[np.float64],
        policy: PredictionPolicy,
    ) -> "ConditionalQuantileCalibrator":
        if policy.calibration != "conditional_quantile":
            raise PolicyError("Conditional calibrator requires conditional_quantile mode")
        observed = np.asarray(targets, dtype=np.float64).reshape(-1)
        predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
        if observed.shape != predicted.shape or observed.size == 0:
            raise PolicyError("Calibration targets and predictions must align")
        if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
            raise PolicyError("Calibration targets and predictions must be finite")

        edges = np.asarray(policy.calibration_prediction_bin_edges, dtype=np.float64)
        residuals = predicted - observed
        corrections: list[float] = []
        counts: list[int] = []
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            mask = (predicted >= lower) & (predicted < upper)
            values = residuals[mask]
            counts.append(int(values.size))
            correction = (
                float(np.quantile(values, policy.non_overprediction_coverage))
                if values.size >= policy.calibration_minimum_bin_rows
                else 0.0
            )
            corrections.append(max(0.0, correction))
        return cls(
            quantile=policy.non_overprediction_coverage,
            prediction_bin_edges=policy.calibration_prediction_bin_edges,
            minimum_bin_rows=policy.calibration_minimum_bin_rows,
            corrections=tuple(corrections),
            training_rows_by_bin=tuple(counts),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConditionalQuantileCalibrator":
        try:
            calibrator = cls(
                quantile=float(value["quantile"]),
                prediction_bin_edges=tuple(
                    float(edge) for edge in value["prediction_bin_edges"]
                ),
                minimum_bin_rows=int(value["minimum_bin_rows"]),
                corrections=tuple(float(item) for item in value["corrections"]),
                training_rows_by_bin=tuple(
                    int(item) for item in value["training_rows_by_bin"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PolicyError("Invalid conditional calibrator artifact") from error
        expected = len(calibrator.prediction_bin_edges) - 1
        if (
            expected < 2
            or len(calibrator.corrections) != expected
            or len(calibrator.training_rows_by_bin) != expected
            or not 0.5 <= calibrator.quantile < 1.0
            or calibrator.minimum_bin_rows <= 0
            or any(
                upper <= lower
                for lower, upper in zip(
                    calibrator.prediction_bin_edges[:-1],
                    calibrator.prediction_bin_edges[1:],
                    strict=True,
                )
            )
            or any(
                not np.isfinite(value) or value < 0.0
                for value in calibrator.corrections
            )
            or any(value < 0 for value in calibrator.training_rows_by_bin)
        ):
            raise PolicyError("Conditional calibrator dimensions are inconsistent")
        return calibrator

    def apply(
        self,
        predictions: NDArray[np.float64],
        *,
        prediction_minimum: float = 0.0,
    ) -> NDArray[np.float64]:
        values = np.asarray(predictions, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise PolicyError("Predictions supplied to calibration must be finite")
        edges = np.asarray(self.prediction_bin_edges, dtype=np.float64)
        centers = (edges[:-1] + edges[1:]) / 2.0
        corrections = np.asarray(self.corrections, dtype=np.float64)
        adjustment = np.interp(
            values,
            centers,
            corrections,
            left=float(corrections[0]),
            right=float(corrections[-1]),
        )
        return np.maximum(values - adjustment, float(prediction_minimum))

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibrator_version": 1,
            "mode": "conditional_quantile",
            "quantile": self.quantile,
            "prediction_bin_edges": list(self.prediction_bin_edges),
            "minimum_bin_rows": self.minimum_bin_rows,
            "corrections": list(self.corrections),
            "training_rows_by_bin": list(self.training_rows_by_bin),
            "correction_is_nonnegative": True,
            "calibration_can_increase_prediction": False,
        }


def cross_fit_conditional_calibration(
    targets: NDArray[np.float64],
    predictions: NDArray[np.float64],
    fold_labels: NDArray[Any],
    policy: PredictionPolicy,
    *,
    prediction_minimum: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calibrate each validation fold using residuals from all other folds."""

    observed = np.asarray(targets, dtype=np.float64).reshape(-1)
    raw = np.asarray(predictions, dtype=np.float64).reshape(-1)
    folds = np.asarray(fold_labels).reshape(-1)
    if observed.shape != raw.shape or observed.shape != folds.shape:
        raise PolicyError("Cross-fit calibration arrays must align")
    unique_folds = np.unique(folds)
    if unique_folds.size < 2:
        raise PolicyError("Cross-fit calibration requires at least two folds")
    calibrated = np.empty_like(raw)
    for fold in unique_folds:
        validation = folds == fold
        calibrator = ConditionalQuantileCalibrator.fit(
            observed[~validation],
            raw[~validation],
            policy,
        )
        calibrated[validation] = calibrator.apply(
            raw[validation],
            prediction_minimum=prediction_minimum,
        )
    return calibrated, raw - calibrated


TARGET_CAPABLE_FAMILIES = {
    "mean_baseline",
    "cycle_only_baseline",
    "regularized_linear",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "catboost",
    "mlp",
    "tcn",
    "multiscale_cnn",
    "sensor_graph_tcn",
    "lstm",
    "transformer",
    "rbf_svr",
    "calibrated_tree_blend",
}

LOSS_CAPABILITIES = {
    "xgboost": {
        "symmetric_rmse",
        "asymmetric_mse",
        "severity_asymmetric_mse",
        "quantile",
    },
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
