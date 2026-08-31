"""Implement deterministic trajectory-retrieval RUL prediction with DTW."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from base import (
    ModelAdapter,
    ModelAdapterError,
    TrainingSummary,
    root_mean_squared_error,
    target_values,
)


class TrajectoryDTWKNNAdapter(ModelAdapter):
    """Retrieve similar training UAV trajectories and average their RULs.

    The trajectory data adapter supplies complete, scaled reference histories
    from the active training UAVs only. A query is compared with the reference
    prefix ending near the query cutoff, using a constrained multivariate DTW
    distance. The matched reference endpoint's cycle-wise remaining life is the
    prediction target for the distance-weighted k-nearest-neighbour estimate.
    """

    family = "trajectory_dtw_knn"
    representation = "trajectory"
    stochastic = False

    def __init__(
        self,
        *,
        hyperparameters: dict[str, Any],
        seed: int,
        prediction_minimum: float = 0.0,
        training_monitor: Any | None = None,
    ) -> None:
        """Store fixed retrieval settings and the training reference library."""

        super().__init__(
            hyperparameters=hyperparameters,
            seed=seed,
            prediction_minimum=prediction_minimum,
            training_monitor=training_monitor,
        )
        self.reference_library: Any | None = None
        self.channel_names: tuple[str, ...] | None = None

        neighbors = int(hyperparameters["neighbors"])
        reference_pool_size = int(hyperparameters["reference_pool_size"])
        max_points = int(hyperparameters["max_points"])
        warping_window = int(hyperparameters["warping_window"])
        distance_power = float(hyperparameters["distance_power"])
        if neighbors <= 0:
            raise ModelAdapterError("Trajectory neighbors must be positive")
        if reference_pool_size < neighbors:
            raise ModelAdapterError(
                "Trajectory reference_pool_size must be at least neighbors"
            )
        if max_points < 2:
            raise ModelAdapterError("Trajectory max_points must be at least two")
        if warping_window < 0:
            raise ModelAdapterError("Trajectory warping_window cannot be negative")
        if not np.isfinite(distance_power) or distance_power <= 0.0:
            raise ModelAdapterError(
                "Trajectory distance_power must be finite and positive"
            )

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        """Store the fold-safe reference library and score development queries."""

        started_at = self.start_timer()
        references = getattr(training_data, "reference_library", None)
        if references is None or len(references) == 0:
            raise ModelAdapterError(
                "Trajectory training data has no complete reference library"
            )
        if not getattr(training_data, "scaled", False) or not getattr(
            references, "scaled", False
        ):
            raise ModelAdapterError(
                "Trajectory DTW requires fold-scaled queries and references"
            )
        channel_names = tuple(getattr(training_data, "channel_names", ()))
        if not channel_names or tuple(references.channel_names) != channel_names:
            raise ModelAdapterError("Trajectory reference channels do not match data")
        self.reference_library = references
        self.channel_names = channel_names
        self._is_fitted = True

        validation_rmse = None
        validation_rows = 0
        if validation_data is not None:
            validation_rows = len(validation_data)
            validation_rmse = root_mean_squared_error(
                target_values(validation_data),
                self.predict(validation_data),
            )
        self.training_summary = TrainingSummary(
            model_family=self.family,
            seed=self.seed,
            training_rows=len(training_data),
            validation_rows=validation_rows,
            training_seconds=self.elapsed_seconds(started_at),
            epochs_or_iterations=None,
            best_epoch_or_iteration=None,
            best_validation_rmse=validation_rmse,
            trainable_parameters=None,
        )
        return self.training_summary

    @staticmethod
    def _downsample(
        trajectory: NDArray[np.float64],
        maximum_points: int,
    ) -> NDArray[np.float64]:
        """Keep ordered endpoints while bounding DTW work per comparison."""

        if len(trajectory) <= maximum_points:
            return trajectory
        indices = np.linspace(
            0,
            len(trajectory) - 1,
            num=maximum_points,
            dtype=np.int64,
        )
        return trajectory[indices]

    @staticmethod
    def _dtw_distance(
        query: NDArray[np.float64],
        reference: NDArray[np.float64],
        warping_window: int,
    ) -> float:
        """Return normalized multivariate DTW distance in a diagonal band."""

        query_length, reference_length = len(query), len(reference)
        previous = np.full(reference_length + 1, np.inf, dtype=np.float64)
        previous[0] = 0.0
        # A fixed-width band must also span the two endpoints when the query
        # and reference prefixes have different lengths. Without this floor,
        # the first or last DP row can be unreachable for short queries.
        window = max(0, int(warping_window), abs(reference_length - query_length))

        for query_index in range(1, query_length + 1):
            current = np.full(reference_length + 1, np.inf, dtype=np.float64)
            center = int(
                round(query_index * reference_length / query_length)
            )
            start = max(1, center - window)
            stop = min(reference_length, center + window)
            query_point = query[query_index - 1]
            for reference_index in range(start, stop + 1):
                point_difference = query_point - reference[reference_index - 1]
                point_cost = float(
                    np.sqrt(np.mean(point_difference * point_difference))
                )
                current[reference_index] = point_cost + min(
                    previous[reference_index],
                    current[reference_index - 1],
                    previous[reference_index - 1],
                )
            previous = current

        if not np.isfinite(previous[reference_length]):
            return float("inf")
        return float(previous[reference_length] / max(query_length, reference_length))

    @staticmethod
    def _endpoint_index(
        cycles: NDArray[np.int64],
        query_cutoff: int,
    ) -> int:
        """Find the reference state at or immediately before a query cutoff."""

        index = int(np.searchsorted(cycles, query_cutoff, side="right") - 1)
        return min(max(index, 0), len(cycles) - 1)

    def _predict_one(
        self,
        query: NDArray[np.float64],
        query_cycles: NDArray[np.int64],
    ) -> float:
        """Retrieve one query's nearest reference lifetimes."""

        if self.reference_library is None:
            raise ModelAdapterError("Trajectory reference library is not fitted")
        maximum_points = int(self.hyperparameters["max_points"])
        warping_window = int(self.hyperparameters["warping_window"])
        neighbors = int(self.hyperparameters["neighbors"])
        pool_size = int(self.hyperparameters["reference_pool_size"])
        distance_power = float(self.hyperparameters["distance_power"])
        query_values = np.asarray(query, dtype=np.float64)
        query_cutoff = int(query_cycles[-1])

        candidates: list[tuple[float, int, int]] = []
        for reference_number, (trajectory, cycles, remaining_life) in enumerate(
            zip(
                self.reference_library.trajectories,
                self.reference_library.cycles,
                self.reference_library.remaining_life,
                strict=True,
            )
        ):
            endpoint_index = self._endpoint_index(cycles, query_cutoff)
            point_difference = query_values[-1] - trajectory[endpoint_index]
            quick_distance = float(
                np.sqrt(np.mean(point_difference * point_difference))
            )
            candidates.append((quick_distance, reference_number, endpoint_index))

        candidates.sort(key=lambda item: (item[0], item[1]))
        query_compact = self._downsample(query_values, maximum_points)
        matches: list[tuple[float, float]] = []
        for _, reference_number, endpoint_index in candidates[:pool_size]:
            reference_values = np.asarray(
                self.reference_library.trajectories[reference_number][
                    : endpoint_index + 1
                ],
                dtype=np.float64,
            )
            reference_compact = self._downsample(reference_values, maximum_points)
            distance = self._dtw_distance(
                query_compact,
                reference_compact,
                warping_window,
            )
            if np.isfinite(distance):
                remaining = float(
                    self.reference_library.remaining_life[reference_number][
                        endpoint_index
                    ]
                )
                matches.append((distance, remaining))

        if len(matches) < neighbors:
            raise ModelAdapterError(
                "Trajectory DTW could not produce enough finite reference matches"
            )
        matches.sort(key=lambda item: item[0])
        selected = matches[:neighbors]
        distances = np.asarray([item[0] for item in selected], dtype=np.float64)
        remaining = np.asarray([item[1] for item in selected], dtype=np.float64)
        exact = distances <= 1e-12
        if exact.any():
            return float(np.mean(remaining[exact]))
        weights = np.power(1.0 / (distances + 1e-12), distance_power)
        return float(np.dot(weights, remaining) / np.sum(weights))

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        """Predict one RUL value for every variable-length trajectory query."""

        if self.channel_names is None:
            raise ModelAdapterError("Trajectory DTW has no fitted channel schema")
        if tuple(getattr(data, "channel_names", ())) != self.channel_names:
            raise ModelAdapterError("Trajectory query channels do not match training")
        predictions = [
            self._predict_one(trajectory, cycles)
            for trajectory, cycles in zip(
                data.trajectories,
                data.cycles,
                strict=True,
            )
        ]
        return np.asarray(predictions, dtype=np.float64)
