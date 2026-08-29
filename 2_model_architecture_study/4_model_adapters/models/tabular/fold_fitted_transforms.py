"""Fold-fitted telemetry transforms shared by tabular tree adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from base import ModelAdapterError


SIGNAL_FAMILIES = {
    "13_16_22_25_28": (13, 16, 22, 25, 28),
    "19_21": (19, 21),
    "15_23": (15, 23),
    "07": (7,),
}
FAULT_MODE_STRATEGIES = {"none", "indicator", "experts"}
SIGNAL_COMPRESSION_STRATEGIES = {
    "none",
    "median_only",
    "pca_only",
    "individual_plus_median",
    "individual_plus_pca",
}


def _score_name(channel: int) -> str:
    return f"feature__telemetry_{channel:02d}__degradation_score"


def _signal_control_mask(feature_names: tuple[str, ...]) -> NDArray[np.bool_]:
    return np.asarray(
        [
            name in {"feature__flight_cycle", "feature__log1p_flight_cycle"}
            or name.endswith("__last")
            for name in feature_names
        ],
        dtype=bool,
    )


class SignalCompressionTransformer:
    """Add or substitute family-level median/PCA health indices."""

    def __init__(self, strategy: str) -> None:
        if strategy not in SIGNAL_COMPRESSION_STRATEGIES:
            raise ModelAdapterError(
                f"Unknown signal compression strategy {strategy!r}"
            )
        self.strategy = strategy
        self._fitted = False

    def fit_transform(
        self,
        values: NDArray[np.float64],
        feature_names: tuple[str, ...],
    ) -> tuple[NDArray[np.float64], tuple[str, ...]]:
        self.feature_names = feature_names
        if self.strategy == "none":
            self._fitted = True
            return values, feature_names
        self.family_indices = self._family_indices(feature_names)
        self.scalers: dict[str, StandardScaler] = {}
        self.pcas: dict[str, PCA] = {}
        if "pca" in self.strategy:
            for family, indices in self.family_indices.items():
                scaler = StandardScaler().fit(values[:, indices])
                scaled = scaler.transform(values[:, indices])
                pca = PCA(n_components=1, svd_solver="full").fit(scaled)
                self.scalers[family] = scaler
                self.pcas[family] = pca
        self._fitted = True
        return self.transform(values, feature_names)

    def transform(
        self,
        values: NDArray[np.float64],
        feature_names: tuple[str, ...],
    ) -> tuple[NDArray[np.float64], tuple[str, ...]]:
        if self.strategy == "none":
            return values, feature_names
        if not self._fitted or feature_names != self.feature_names:
            raise ModelAdapterError("Signal compression feature columns changed")

        derived_values: list[NDArray[np.float64]] = []
        derived_names: list[str] = []
        for family, indices in self.family_indices.items():
            family_values = values[:, indices]
            if "median" in self.strategy:
                derived_values.append(np.median(family_values, axis=1))
                derived_names.append(f"derived__signal_{family}__median_health")
            if "pca" in self.strategy:
                transformed = self.scalers[family].transform(family_values)
                component = self.pcas[family].transform(transformed).reshape(-1)
                derived_values.append(component)
                derived_names.append(f"derived__signal_{family}__pca_health")

        if self.strategy.endswith("_only"):
            keep = _signal_control_mask(feature_names)
            base_values = values[:, keep]
            base_names = tuple(np.asarray(feature_names, dtype=object)[keep])
        else:
            base_values = values
            base_names = feature_names
        additions = np.column_stack(derived_values)
        return (
            np.column_stack([base_values, additions]),
            (*base_names, *derived_names),
        )

    @staticmethod
    def _family_indices(feature_names: tuple[str, ...]) -> dict[str, list[int]]:
        positions = {name: index for index, name in enumerate(feature_names)}
        result: dict[str, list[int]] = {}
        for family, channels in SIGNAL_FAMILIES.items():
            names = [_score_name(channel) for channel in channels]
            missing = [name for name in names if name not in positions]
            if missing:
                raise ModelAdapterError(
                    "Signal compression requires degradation-score features: "
                    f"{missing}"
                )
            result[family] = [positions[name] for name in names]
        return result


@dataclass(frozen=True)
class FaultModeAssignments:
    modes: NDArray[np.int64]
    trusted: NDArray[np.bool_]


class FaultModeTransformer:
    """Infer two modes from training UAVs and assign later rows by proximity."""

    def __init__(self, strategy: str, *, seed: int) -> None:
        if strategy not in FAULT_MODE_STRATEGIES:
            raise ModelAdapterError(f"Unknown fault-mode strategy {strategy!r}")
        self.strategy = strategy
        self.seed = int(seed)
        self._fitted = False

    def fit(
        self,
        values: NDArray[np.float64],
        feature_names: tuple[str, ...],
        metadata: Any,
    ) -> FaultModeAssignments:
        if self.strategy == "none":
            self._fitted = True
            return FaultModeAssignments(
                modes=np.zeros(len(values), dtype=np.int64),
                trusted=np.ones(len(values), dtype=bool),
            )
        required = {"uav_id", "cutoff"}
        if metadata is None or not required.issubset(metadata.columns):
            raise ModelAdapterError(
                "Fault-mode inference requires uav_id and cutoff metadata"
            )
        self.feature_names = feature_names
        positions = {name: index for index, name in enumerate(feature_names)}
        score_names = [
            _score_name(channel)
            for channels in SIGNAL_FAMILIES.values()
            for channel in channels
        ]
        missing = [name for name in score_names if name not in positions]
        if missing:
            raise ModelAdapterError(
                f"Fault-mode inference requires degradation scores: {missing}"
            )
        self.score_indices = [positions[name] for name in score_names]
        latest_indices = (
            metadata.assign(_row=np.arange(len(metadata)))
            .sort_values(["uav_id", "cutoff"], kind="stable")
            .groupby("uav_id", sort=True)["_row"]
            .last()
            .to_numpy(dtype=int)
        )
        if len(latest_indices) < 10:
            raise ModelAdapterError("Fault-mode inference needs at least ten UAVs")
        latest = values[latest_indices][:, self.score_indices]
        self.scaler = StandardScaler().fit(latest)
        standardized = self.scaler.transform(latest)
        self.cluster = KMeans(
            n_clusters=2,
            random_state=self.seed,
            n_init=20,
        ).fit(standardized)
        distances = np.min(self.cluster.transform(standardized), axis=1)
        self.distance_threshold = float(np.quantile(distances, 0.95))
        self._fitted = True
        return self.assign(values, feature_names)

    def assign(
        self,
        values: NDArray[np.float64],
        feature_names: tuple[str, ...],
    ) -> FaultModeAssignments:
        if self.strategy == "none":
            return FaultModeAssignments(
                modes=np.zeros(len(values), dtype=np.int64),
                trusted=np.ones(len(values), dtype=bool),
            )
        if not self._fitted or feature_names != self.feature_names:
            raise ModelAdapterError("Fault-mode feature columns changed")
        standardized = self.scaler.transform(values[:, self.score_indices])
        distances = self.cluster.transform(standardized)
        modes = np.argmin(distances, axis=1).astype(np.int64)
        nearest = distances[np.arange(len(values)), modes]
        return FaultModeAssignments(
            modes=modes,
            trusted=nearest <= self.distance_threshold,
        )

    def append_indicator(
        self,
        values: NDArray[np.float64],
        feature_names: tuple[str, ...],
        assignments: FaultModeAssignments,
    ) -> tuple[NDArray[np.float64], tuple[str, ...]]:
        indicator = assignments.modes.astype(np.float64)
        indicator = np.where(assignments.trusted, indicator, -1.0)
        return (
            np.column_stack([values, indicator]),
            (*feature_names, "derived__fault_mode"),
        )
