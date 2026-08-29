"""Construct the frozen Phase 3 training and test model inputs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from sequence_data_adapter import (
    RobustChannelScaler,
    SequenceDataAdapter,
    SequenceDataset,
)
from tabular_data_adapter import TabularDataAdapter, TabularDataset
from trajectory_data_adapter import (
    TrajectoryChannelScaler,
    TrajectoryDataAdapter,
    TrajectoryDataset,
)

from phase_3_common import Phase3Error
from phase_3_common import REPOSITORY_ROOT


def _manifest_path(contract: dict[str, Any], name: str) -> Path:
    """Resolve one run-specific adapter manifest recorded in the contract."""

    manifests = contract.get("data_manifests", {})
    value = manifests.get(name) if isinstance(manifests, dict) else None
    if not isinstance(value, str) or not value:
        raise Phase3Error(f"Final training contract has no {name} data manifest")
    path = (REPOSITORY_ROOT / value).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise Phase3Error(f"{name} data manifest escapes the repository") from error
    return path


def _tabular_feature_set(contract: dict[str, Any]) -> str:
    value = contract["input_schema"].get("feature_set")
    return str(value or "age_only")


def load_final_training_data(
    contract: dict[str, Any],
) -> tuple[Any, RobustChannelScaler | TrajectoryChannelScaler | None]:
    """Load all training prefixes and fit only permitted preprocessing."""

    representation = contract["representation"]
    if representation in {"none", "tabular"}:
        dataset = TabularDataAdapter(
            _manifest_path(contract, "tabular")
        ).load_training(_tabular_feature_set(contract))
        expected = list(contract["input_schema"]["feature_names"])
        if list(dataset.features.columns) != expected:
            raise Phase3Error("Final tabular training feature order changed")
        return dataset, None
    if representation == "sequence":
        lookback = int(contract["input_schema"]["lookback"])
        adapter = SequenceDataAdapter(_manifest_path(contract, "sequence"))
        dataset = adapter.load_training(lookback)
        if list(dataset.channel_names) != contract["input_schema"]["channel_names"]:
            raise Phase3Error("Final sequence channel order changed")
        scaler = adapter.fit_channel_scaler(contract["training"]["uav_ids"])
        return scaler.transform(dataset), scaler
    if representation == "trajectory":
        adapter = TrajectoryDataAdapter(_manifest_path(contract, "trajectory"))
        dataset = adapter.load_training()
        if list(dataset.channel_names) != contract["input_schema"]["channel_names"]:
            raise Phase3Error("Final trajectory channel order changed")
        scaler = adapter.fit_channel_scaler(contract["training"]["uav_ids"])
        return scaler.transform(dataset), scaler
    raise Phase3Error(f"Unsupported representation {representation!r}")


def load_final_test_data(
    contract: dict[str, Any],
    preprocessor: RobustChannelScaler | TrajectoryChannelScaler | None,
) -> Any:
    """Load test endpoints only after the frozen contract gate passes."""

    representation = contract["representation"]
    if representation in {"none", "tabular"}:
        dataset = TabularDataAdapter(
            _manifest_path(contract, "tabular")
        ).load_test(_tabular_feature_set(contract))
        expected = list(contract["input_schema"]["feature_names"])
        if list(dataset.features.columns) != expected:
            raise Phase3Error("Final test feature order differs from the contract")
        return dataset
    if representation == "sequence":
        if not isinstance(preprocessor, RobustChannelScaler):
            raise Phase3Error("Sequence inference requires its fitted channel scaler")
        lookback = int(contract["input_schema"]["lookback"])
        dataset = SequenceDataAdapter(
            _manifest_path(contract, "sequence")
        ).load_test(lookback)
        if list(dataset.channel_names) != contract["input_schema"]["channel_names"]:
            raise Phase3Error("Final test channel order differs from the contract")
        return preprocessor.transform(dataset)
    if representation == "trajectory":
        if not isinstance(preprocessor, TrajectoryChannelScaler):
            raise Phase3Error("Trajectory inference requires its fitted channel scaler")
        adapter = TrajectoryDataAdapter(_manifest_path(contract, "trajectory"))
        dataset = adapter.load_test()
        if list(dataset.channel_names) != contract["input_schema"]["channel_names"]:
            raise Phase3Error("Final test trajectory channel order differs from the contract")
        return preprocessor.transform(dataset)
    raise Phase3Error(f"Unsupported representation {representation!r}")


def first_rows(dataset: Any, count: int) -> Any:
    """Return a stable leading subset for save/reload prediction checks."""

    size = min(max(1, count), len(dataset))
    if isinstance(dataset, TabularDataset):
        target = (
            dataset.target.iloc[:size].reset_index(drop=True)
            if dataset.target is not None
            else None
        )
        weights = (
            dataset.sample_weights.iloc[:size].reset_index(drop=True)
            if dataset.sample_weights is not None
            else None
        )
        return TabularDataset(
            features=dataset.features.iloc[:size].reset_index(drop=True),
            metadata=dataset.metadata.iloc[:size].reset_index(drop=True),
            target=target,
            sample_weights=weights,
        )
    if isinstance(dataset, SequenceDataset):
        target = (
            dataset.target.iloc[:size].reset_index(drop=True)
            if dataset.target is not None
            else None
        )
        weights = (
            dataset.sample_weights.iloc[:size].reset_index(drop=True)
            if dataset.sample_weights is not None
            else None
        )
        return replace(
            dataset,
            sequences=dataset.sequences[:size].copy(),
            padding_mask=dataset.padding_mask[:size].copy(),
            side_features=dataset.side_features[:size].copy(),
            metadata=dataset.metadata.iloc[:size].reset_index(drop=True),
            target=target,
            sample_weights=weights,
        )
    if isinstance(dataset, TrajectoryDataset):
        target = (
            dataset.target.iloc[:size].reset_index(drop=True)
            if dataset.target is not None
            else None
        )
        weights = (
            dataset.sample_weights.iloc[:size].reset_index(drop=True)
            if dataset.sample_weights is not None
            else None
        )
        return replace(
            dataset,
            trajectories=dataset.trajectories[:size],
            cycles=dataset.cycles[:size],
            side_features=dataset.side_features[:size].copy(),
            metadata=dataset.metadata.iloc[:size].reset_index(drop=True),
            target=target,
            sample_weights=weights,
        )
    raise Phase3Error(f"Unsupported dataset type {type(dataset).__name__}")
