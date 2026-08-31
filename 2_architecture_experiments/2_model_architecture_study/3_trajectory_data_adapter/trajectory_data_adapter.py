"""Construct causal trajectory queries and fold-safe reference libraries."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from typing import Any, Collection

import numpy as np
from numpy.typing import NDArray
import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STEP_DIR.parents[2]
SEQUENCE_STEP_DIR = STEP_DIR.parent / "3_sequence_data_adapter"
if str(SEQUENCE_STEP_DIR) not in sys.path:
    sys.path.insert(0, str(SEQUENCE_STEP_DIR))

from sequence_data_adapter import (  # noqa: E402
    RobustChannelScaler,
    SequenceAdapterError,
    SequenceDataAdapter,
)


DEFAULT_ARTIFACT_DIR = STEP_DIR / "artifacts"
DEFAULT_MANIFEST_PATH = DEFAULT_ARTIFACT_DIR / "trajectory_dataset_manifest.json"


class TrajectoryAdapterError(ValueError):
    """Represent a readable trajectory configuration or data failure."""


@dataclass(frozen=True)
class TrajectoryReferenceLibrary:
    """Hold complete run-to-failure histories from active training UAVs."""

    trajectories: tuple[NDArray[np.float32], ...]
    cycles: tuple[NDArray[np.int64], ...]
    remaining_life: tuple[NDArray[np.float32], ...]
    metadata: pd.DataFrame
    channel_names: tuple[str, ...]
    scaled: bool

    def __post_init__(self) -> None:
        """Reject incomplete or misaligned reference trajectories."""

        row_count = len(self.trajectories)
        if not (
            len(self.cycles)
            == len(self.remaining_life)
            == len(self.metadata)
            == row_count
        ):
            raise TrajectoryAdapterError("Reference library rows are misaligned")
        for trajectory, cycles, remaining_life in zip(
            self.trajectories,
            self.cycles,
            self.remaining_life,
            strict=True,
        ):
            if trajectory.ndim != 2 or trajectory.shape[1] != len(self.channel_names):
                raise TrajectoryAdapterError("Reference trajectory shape is invalid")
            if len(trajectory) == 0 or len(cycles) != len(trajectory):
                raise TrajectoryAdapterError("Reference cycle count is invalid")
            if len(remaining_life) != len(trajectory):
                raise TrajectoryAdapterError("Reference RUL count is invalid")
            if np.any(np.diff(cycles) <= 0):
                raise TrajectoryAdapterError("Reference cycles are not ordered")

    def __len__(self) -> int:
        """Return the number of complete reference UAV histories."""

        return len(self.trajectories)


@dataclass(frozen=True)
class TrajectoryDataset:
    """Hold variable-length causal trajectories for prediction endpoints."""

    trajectories: tuple[NDArray[np.float32], ...]
    cycles: tuple[NDArray[np.int64], ...]
    side_features: NDArray[np.float32]
    metadata: pd.DataFrame
    target: pd.Series | None
    sample_weights: pd.Series | None
    channel_names: tuple[str, ...]
    side_feature_names: tuple[str, ...]
    reference_library: TrajectoryReferenceLibrary | None
    scaled: bool

    def __post_init__(self) -> None:
        """Reject trajectory and endpoint alignment errors immediately."""

        row_count = len(self.trajectories)
        if len(self.cycles) != row_count or len(self.metadata) != row_count:
            raise TrajectoryAdapterError("Trajectory dataset rows are misaligned")
        if self.side_features.shape != (row_count, len(self.side_feature_names)):
            raise TrajectoryAdapterError("Trajectory side-feature shape is invalid")
        if self.target is not None and len(self.target) != row_count:
            raise TrajectoryAdapterError("Trajectory target rows are misaligned")
        if self.sample_weights is not None and len(self.sample_weights) != row_count:
            raise TrajectoryAdapterError("Trajectory weights are misaligned")
        for trajectory, cycles in zip(self.trajectories, self.cycles, strict=True):
            if trajectory.ndim != 2 or trajectory.shape[1] != len(self.channel_names):
                raise TrajectoryAdapterError("Query trajectory shape is invalid")
            if len(trajectory) == 0 or len(cycles) != len(trajectory):
                raise TrajectoryAdapterError("Query trajectory cycle count is invalid")
            if np.any(np.diff(cycles) <= 0):
                raise TrajectoryAdapterError("Query trajectory cycles are not ordered")

    def __len__(self) -> int:
        """Return the number of prediction queries."""

        return len(self.trajectories)


@dataclass(frozen=True)
class TrajectoryChannelScaler:
    """Apply one sequence-compatible fold-fitted scaler to trajectories."""

    parameters: RobustChannelScaler

    def _transform_trajectory(
        self,
        trajectory: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        transformed = (
            trajectory.astype(np.float64) - self.parameters.centers
        ) / self.parameters.scales
        if not np.isfinite(transformed).all():
            raise TrajectoryAdapterError("Scaled trajectory values are not finite")
        return transformed.astype(np.float32)

    def transform(self, dataset: TrajectoryDataset) -> TrajectoryDataset:
        """Scale every query and its optional training reference library."""

        if dataset.channel_names != self.parameters.channel_names:
            raise TrajectoryAdapterError("Scaler channels do not match trajectories")
        references = dataset.reference_library
        scaled_references = None
        if references is not None:
            scaled_references = replace(
                references,
                trajectories=tuple(
                    self._transform_trajectory(value)
                    for value in references.trajectories
                ),
                scaled=True,
            )
        return replace(
            dataset,
            trajectories=tuple(
                self._transform_trajectory(value) for value in dataset.trajectories
            ),
            reference_library=scaled_references,
            scaled=True,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return the same readable channel parameters as the sequence path."""

        return self.parameters.to_frame()


@dataclass(frozen=True)
class TrajectorySplit:
    """Pair fold-scaled trajectory datasets with their fitted scaler."""

    training: TrajectoryDataset
    validation: TrajectoryDataset
    channel_scaler: TrajectoryChannelScaler


class TrajectoryDataAdapter:
    """Build variable-length trajectory views using Step 3's verified inputs."""

    def __init__(self, manifest_path: Path = DEFAULT_MANIFEST_PATH) -> None:
        self.manifest_path = manifest_path.resolve()
        self.manifest = self._load_manifest()
        sequence_manifest = self._repository_path(
            self.manifest.get("source_sequence_manifest"),
            "source_sequence_manifest",
        )
        try:
            self.sequence_adapter = SequenceDataAdapter(sequence_manifest)
        except SequenceAdapterError as error:
            raise TrajectoryAdapterError(str(error)) from error
        self.channel_names = self.sequence_adapter.channel_names
        self.side_feature_names = self.sequence_adapter.side_feature_names

    def _load_manifest(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrajectoryAdapterError(
                "Trajectory adapter artifacts are missing or invalid. Run "
                f"build_trajectory_data_adapter.py first. Details: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise TrajectoryAdapterError("Trajectory manifest must be an object")
        return payload

    @staticmethod
    def _repository_path(value: Any, field: str) -> Path:
        if not isinstance(value, str) or Path(value).is_absolute():
            raise TrajectoryAdapterError(f"Manifest field {field!r} is invalid")
        supplied = Path(value)
        resolved = (REPOSITORY_ROOT / supplied).resolve()
        if not resolved.exists() and supplied.parts:
            moved_prefixes = {
                "pipeline_experiments": Path(
                    "2_architecture_experiments/1_pipeline_experiments"
                ),
                "2_model_architecture_study": Path(
                    "2_architecture_experiments/2_model_architecture_study"
                ),
            }
            replacement = moved_prefixes.get(supplied.parts[0])
            if replacement is not None:
                resolved = (
                    REPOSITORY_ROOT
                    / replacement.joinpath(*supplied.parts[1:])
                ).resolve()
        try:
            resolved.relative_to(REPOSITORY_ROOT.resolve())
        except ValueError as error:
            raise TrajectoryAdapterError(
                f"Manifest field {field!r} escapes the repository"
            ) from error
        if not resolved.is_file():
            raise TrajectoryAdapterError(f"Required adapter input is missing: {resolved}")
        return resolved

    def _build_reference_library(
        self,
        uav_ids: Collection[str],
    ) -> TrajectoryReferenceLibrary:
        """Load complete histories for training UAVs and derive cycle-wise RUL."""

        try:
            histories = self.sequence_adapter._load_histories("raw_train", uav_ids)
        except SequenceAdapterError as error:
            raise TrajectoryAdapterError(str(error)) from error
        trajectories: list[NDArray[np.float32]] = []
        cycles: list[NDArray[np.int64]] = []
        remaining_life: list[NDArray[np.float32]] = []
        metadata_rows: list[dict[str, Any]] = []
        for uav_id in sorted(histories):
            cycle_values, trajectory = histories[uav_id]
            terminal_cycle = int(cycle_values[-1])
            trajectories.append(trajectory.astype(np.float32))
            cycles.append(cycle_values.copy())
            remaining_life.append(
                (terminal_cycle - cycle_values).astype(np.float32)
            )
            metadata_rows.append(
                {
                    "uav_id": uav_id,
                    "terminal_cycle": terminal_cycle,
                    "trajectory_length": len(cycle_values),
                }
            )
        return TrajectoryReferenceLibrary(
            trajectories=tuple(trajectories),
            cycles=tuple(cycles),
            remaining_life=tuple(remaining_life),
            metadata=pd.DataFrame(metadata_rows),
            channel_names=self.channel_names,
            scaled=False,
        )

    def _build_dataset(
        self,
        endpoint_name: str,
        uav_ids: Collection[str] | None = None,
        reference_uav_ids: Collection[str] | None = None,
    ) -> TrajectoryDataset:
        """Build complete causal prefixes ending exactly at each cutoff."""

        try:
            endpoints, entry = self.sequence_adapter._load_endpoints(
                endpoint_name,
                uav_ids,
            )
            history_name = entry.get("history_file")
            if not isinstance(history_name, str):
                raise TrajectoryAdapterError(
                    f"Endpoint table {endpoint_name!r} has no history file"
                )
            endpoint_uavs = endpoints["uav_id"].astype(str).unique().tolist()
            histories = self.sequence_adapter._load_histories(
                history_name,
                endpoint_uavs,
            )
        except SequenceAdapterError as error:
            raise TrajectoryAdapterError(str(error)) from error

        trajectories: list[NDArray[np.float32]] = []
        cycle_parts: list[NDArray[np.int64]] = []
        for _, endpoint in endpoints.iterrows():
            uav_id = str(endpoint["uav_id"])
            cutoff = int(endpoint["cutoff"])
            cycles, values = histories[uav_id]
            stop = int(np.searchsorted(cycles, cutoff, side="right"))
            if stop == 0 or int(cycles[stop - 1]) != cutoff:
                raise TrajectoryAdapterError(
                    f"History {uav_id!r} has no reading at cutoff {cutoff}"
                )
            trajectories.append(values[:stop].astype(np.float32))
            cycle_parts.append(cycles[:stop].copy())

        metadata_columns = entry.get("metadata_columns")
        if not isinstance(metadata_columns, list):
            raise TrajectoryAdapterError(
                f"Endpoint table {endpoint_name!r} has invalid metadata columns"
            )
        target_name = entry.get("target_column")
        weight_name = entry.get("sample_weight_column")
        target = (
            endpoints[target_name].copy()
            if isinstance(target_name, str) and target_name in endpoints.columns
            else None
        )
        weights = (
            endpoints[weight_name].copy()
            if isinstance(weight_name, str) and weight_name in endpoints.columns
            else None
        )
        cutoffs = endpoints["cutoff"].to_numpy(dtype=np.int64)
        references = (
            self._build_reference_library(reference_uav_ids)
            if reference_uav_ids is not None
            else None
        )
        return TrajectoryDataset(
            trajectories=tuple(trajectories),
            cycles=tuple(cycle_parts),
            side_features=self.sequence_adapter._side_feature_values(cutoffs),
            metadata=endpoints.loc[:, metadata_columns].copy(),
            target=target,
            sample_weights=weights,
            channel_names=self.channel_names,
            side_feature_names=self.side_feature_names,
            reference_library=references,
            scaled=False,
        )

    def fit_channel_scaler(
        self,
        training_uav_ids: Collection[str],
    ) -> TrajectoryChannelScaler:
        """Fit robust telemetry scaling on active training UAV histories only."""

        try:
            parameters = self.sequence_adapter.fit_channel_scaler(training_uav_ids)
        except SequenceAdapterError as error:
            raise TrajectoryAdapterError(str(error)) from error
        return TrajectoryChannelScaler(parameters)

    def load_training(self) -> TrajectoryDataset:
        """Build all training queries with all training UAVs as references."""

        endpoints, _ = self.sequence_adapter._load_endpoints(
            "training_endpoints",
            None,
        )
        uav_ids = endpoints["uav_id"].astype(str).unique().tolist()
        return self._build_dataset(
            "training_endpoints",
            reference_uav_ids=uav_ids,
        )

    def load_development(self) -> TrajectoryDataset:
        """Build unscaled development queries without a reference library."""

        return self._build_dataset("development_endpoints")

    def load_locked(self) -> TrajectoryDataset:
        """Explicitly build unscaled locked queries without references."""

        return self._build_dataset("locked_endpoints")

    def load_test(self) -> TrajectoryDataset:
        """Build unlabelled test trajectories ending at each test cutoff."""

        return self._build_dataset("test_endpoints")

    def outer_fold_labels(self) -> tuple[int, ...]:
        """Return labels from the shared copied outer-fold table."""

        return self.sequence_adapter.outer_fold_labels()

    def inner_fold_labels(self, outer_fold: int) -> tuple[int, ...]:
        """Return labels from the shared copied inner-fold table."""

        return self.sequence_adapter.inner_fold_labels(outer_fold)

    def _split(
        self,
        *,
        outer_fold: int,
        inner_fold: int | None,
        validation_endpoint: str,
    ) -> TrajectorySplit:
        try:
            training_ids, validation_ids = self.sequence_adapter._fold_uav_ids(
                outer_fold=outer_fold,
                inner_fold=inner_fold,
            )
        except SequenceAdapterError as error:
            raise TrajectoryAdapterError(str(error)) from error
        raw_training = self._build_dataset(
            "training_endpoints",
            training_ids,
            training_ids,
        )
        raw_validation = self._build_dataset(
            validation_endpoint,
            validation_ids,
            training_ids,
        )
        scaler = self.fit_channel_scaler(training_ids)
        return TrajectorySplit(
            training=scaler.transform(raw_training),
            validation=scaler.transform(raw_validation),
            channel_scaler=scaler,
        )

    def get_inner_selection_split(
        self,
        outer_fold: int,
        inner_fold: int,
    ) -> TrajectorySplit:
        """Return a fold-safe inner-training/development trajectory split."""

        return self._split(
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            validation_endpoint="development_endpoints",
        )

    def get_final_search_split(self, outer_fold: int) -> TrajectorySplit:
        """Return one outer-training/development split for Phase 3 search."""

        return self._split(
            outer_fold=outer_fold,
            inner_fold=None,
            validation_endpoint="development_endpoints",
        )

    def get_locked_outer_evaluation_split(
        self,
        outer_fold: int,
    ) -> TrajectorySplit:
        """Return a fold-safe outer-training/locked trajectory split."""

        return self._split(
            outer_fold=outer_fold,
            inner_fold=None,
            validation_endpoint="locked_endpoints",
        )
