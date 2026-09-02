"""Verify heterogeneous alignment and hybrid neural input contracts."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


STEP_DIR = Path(__file__).resolve().parent
for dependency in (
    STEP_DIR,
    STEP_DIR.parent / "2_tabular_data_adapter",
    STEP_DIR.parent / "3_sequence_data_adapter",
):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from heterogeneous_data import HeterogeneousDataset  # noqa: E402
from models.neural.hybrid import (  # noqa: E402
    HybridCNNRegressor,
    HybridGRURegressor,
    build_resolution_view,
)
from sequence_data_adapter import SequenceDataset  # noqa: E402
from tabular_data_adapter import TabularDataset  # noqa: E402


def main() -> None:
    metadata = pd.DataFrame(
        {
            "uav_id": ["1", "2"],
            "scenario": ["development_01", "development_02"],
            "cutoff": [6, 8],
        }
    )
    target = pd.Series([10.0, 20.0])
    weights = pd.Series([1.0, 1.0])
    tabular = TabularDataset(
        features=pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]}),
        metadata=metadata.copy(),
        target=target.copy(),
        sample_weights=weights.copy(),
    )
    values = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
    mask = np.zeros((2, 8), dtype=np.bool_)
    mask[0, :2] = True
    values[mask] = 0.0
    sequence = SequenceDataset(
        sequences=values,
        padding_mask=mask,
        side_features=np.array([[6.0, np.log1p(6)], [8.0, np.log1p(8)]], dtype=np.float32),
        metadata=metadata.copy(),
        target=target.copy(),
        sample_weights=weights.copy(),
        channel_names=("a", "b", "c"),
        side_feature_names=("flight_cycle", "log1p_flight_cycle"),
        lookback=8,
        scaled=True,
    )
    aligned = HeterogeneousDataset(tabular=tabular, sequence=sequence)
    assert len(aligned) == 2

    recent, recent_mask = build_resolution_view(
        values,
        mask,
        history_mode="recent_only",
        recent_lookback=4,
        history_bins=2,
    )
    assert recent.shape == (2, 4, 4)
    assert recent_mask.shape == (2, 4)
    multi, multi_mask = build_resolution_view(
        values,
        mask,
        history_mode="multiresolution",
        recent_lookback=4,
        history_bins=2,
    )
    assert multi.shape == (2, 6, 4)
    assert multi_mask.shape == (2, 6)
    assert np.all(multi[multi_mask] == 0.0)

    tensors = (
        torch.as_tensor(multi),
        torch.as_tensor(multi_mask),
        torch.zeros((2, 2)),
        torch.zeros((2, 5)),
    )
    cnn = HybridCNNRegressor(
        input_channels=4,
        side_features=2,
        tabular_features=5,
        branch_channels=4,
        kernel_sizes=[3, 5],
        tabular_hidden_units=8,
        fusion_hidden_units=8,
        dropout=0.0,
    )
    gru = HybridGRURegressor(
        input_channels=4,
        side_features=2,
        tabular_features=5,
        layers=1,
        hidden_units=6,
        tabular_hidden_units=8,
        fusion_hidden_units=8,
        dropout=0.0,
    )
    assert cnn(*tensors).shape == (2,)
    assert gru(*tensors).shape == (2,)
    print("Hybrid model verification passed")


if __name__ == "__main__":
    main()
