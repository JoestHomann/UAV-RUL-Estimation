"""Construct the frozen Phase 3 training and test model inputs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from sequence_data_adapter import (
    RobustChannelScaler,
    SequenceDataAdapter,
    SequenceDataset,
)
from tabular_data_adapter import TabularDataAdapter, TabularDataset

from phase_3_common import Phase3Error


def _tabular_feature_set(contract: dict[str, Any]) -> str:
    value = contract["input_schema"].get("feature_set")
    return str(value or "age_only")


def load_final_training_data(
    contract: dict[str, Any],
) -> tuple[Any, RobustChannelScaler | None]:
    """Load all training prefixes and fit only permitted preprocessing."""

    representation = contract["representation"]
    if representation in {"none", "tabular"}:
        dataset = TabularDataAdapter().load_training(_tabular_feature_set(contract))
        expected = list(contract["input_schema"]["feature_names"])
        if list(dataset.features.columns) != expected:
            raise Phase3Error("Final tabular training feature order changed")
        return dataset, None
    if representation == "sequence":
        lookback = int(contract["input_schema"]["lookback"])
        adapter = SequenceDataAdapter()
        dataset = adapter.load_training(lookback)
        if list(dataset.channel_names) != contract["input_schema"]["channel_names"]:
            raise Phase3Error("Final sequence channel order changed")
        scaler = adapter.fit_channel_scaler(contract["training"]["uav_ids"])
        return scaler.transform(dataset), scaler
    raise Phase3Error(f"Unsupported representation {representation!r}")


def load_final_test_data(
    contract: dict[str, Any],
    preprocessor: RobustChannelScaler | None,
) -> Any:
    """Load test endpoints only after the frozen contract gate passes."""

    representation = contract["representation"]
    if representation in {"none", "tabular"}:
        dataset = TabularDataAdapter().load_test(_tabular_feature_set(contract))
        expected = list(contract["input_schema"]["feature_names"])
        if list(dataset.features.columns) != expected:
            raise Phase3Error("Final test feature order differs from the contract")
        return dataset
    if representation == "sequence":
        if not isinstance(preprocessor, RobustChannelScaler):
            raise Phase3Error("Sequence inference requires its fitted channel scaler")
        lookback = int(contract["input_schema"]["lookback"])
        dataset = SequenceDataAdapter().load_test(lookback)
        if list(dataset.channel_names) != contract["input_schema"]["channel_names"]:
            raise Phase3Error("Final test channel order differs from the contract")
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
    raise Phase3Error(f"Unsupported dataset type {type(dataset).__name__}")
