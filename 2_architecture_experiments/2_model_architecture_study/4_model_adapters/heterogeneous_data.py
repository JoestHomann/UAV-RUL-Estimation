"""Align tabular and sequence views for joint model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from base import ModelAdapterError


REQUIRED_ALIGNMENT_COLUMNS = ("uav_id", "cutoff")
OPTIONAL_ALIGNMENT_COLUMNS = ("scenario",)


@dataclass(frozen=True)
class HeterogeneousDataset:
    """Expose aligned tabular and sequence views through one dataset contract."""

    tabular: Any
    sequence: Any

    def __post_init__(self) -> None:
        if len(self.tabular) != len(self.sequence):
            raise ModelAdapterError("Tabular and sequence row counts differ")
        for column in REQUIRED_ALIGNMENT_COLUMNS:
            if column not in self.tabular.metadata or column not in self.sequence.metadata:
                raise ModelAdapterError(
                    f"Heterogeneous data is missing alignment column {column!r}"
                )
        alignment_columns = list(REQUIRED_ALIGNMENT_COLUMNS)
        for column in OPTIONAL_ALIGNMENT_COLUMNS:
            availability = (
                column in self.tabular.metadata,
                column in self.sequence.metadata,
            )
            if availability[0] != availability[1]:
                raise ModelAdapterError(
                    f"Tabular and sequence alignment column {column!r} availability differs"
                )
            if availability[0]:
                alignment_columns.append(column)
        tabular_keys = _alignment_keys(self.tabular.metadata, alignment_columns)
        sequence_keys = _alignment_keys(self.sequence.metadata, alignment_columns)
        if not tabular_keys.equals(sequence_keys):
            raise ModelAdapterError(
                "Tabular and sequence rows are not aligned by UAV, scenario, and cutoff"
            )
        _matching_optional_values(
            self.tabular.target,
            self.sequence.target,
            label="target",
        )
        _matching_optional_values(
            self.tabular.sample_weights,
            self.sequence.sample_weights,
            label="sample weights",
        )
        _matching_optional_values(
            self.tabular.fitting_target,
            self.sequence.fitting_target,
            label="fitting target",
        )

    def __len__(self) -> int:
        return len(self.tabular)

    @property
    def metadata(self) -> pd.DataFrame:
        return self.tabular.metadata

    @property
    def target(self) -> pd.Series | None:
        return self.tabular.target

    @property
    def sample_weights(self) -> pd.Series | None:
        return self.tabular.sample_weights

    @property
    def fitting_target(self) -> pd.Series | None:
        return self.tabular.fitting_target


@dataclass(frozen=True)
class HeterogeneousSplit:
    """Pair aligned heterogeneous training and validation datasets."""

    training: HeterogeneousDataset
    validation: HeterogeneousDataset


def align_split(tabular_split: Any, sequence_split: Any) -> HeterogeneousSplit:
    """Build one strict aligned split from the existing public adapters."""

    return HeterogeneousSplit(
        training=HeterogeneousDataset(
            tabular=tabular_split.training,
            sequence=sequence_split.training,
        ),
        validation=HeterogeneousDataset(
            tabular=tabular_split.validation,
            sequence=sequence_split.validation,
        ),
    )


def _alignment_keys(metadata: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values: dict[str, np.ndarray] = {
        "uav_id": metadata["uav_id"].astype(str).to_numpy(),
        "cutoff": pd.to_numeric(metadata["cutoff"], errors="raise").to_numpy(),
    }
    if "scenario" in columns:
        values["scenario"] = metadata["scenario"].astype(str).to_numpy()
    return pd.DataFrame({column: values[column] for column in columns})


def _matching_optional_values(left: Any, right: Any, *, label: str) -> None:
    if (left is None) != (right is None):
        raise ModelAdapterError(f"Tabular and sequence {label} availability differs")
    if left is None:
        return
    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_values.shape != right_values.shape or not np.allclose(
        left_values,
        right_values,
        rtol=0.0,
        atol=1e-7,
    ):
        raise ModelAdapterError(f"Tabular and sequence {label} values differ")
