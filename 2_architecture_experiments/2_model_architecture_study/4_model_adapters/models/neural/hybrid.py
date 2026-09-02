"""Joint temporal and engineered-feature neural architectures."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import RobustScaler
import torch
from torch import Tensor, nn
from torch.nn.utils import rnn

from base import ModelAdapterError, tabular_values
from models.neural.multiscale_cnn import MultiScaleBranch
from models.neural.neural_base import NeuralInputs, NeuralModelAdapter


HISTORY_MODES = {"recent_only", "multiresolution"}


def build_resolution_view(
    sequences: np.ndarray,
    padding_mask: np.ndarray,
    *,
    history_mode: str,
    recent_lookback: int,
    history_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return recent raw cycles, optionally preceded by pooled older history."""

    if history_mode not in HISTORY_MODES:
        raise ModelAdapterError(f"Unknown hybrid history mode {history_mode!r}")
    if recent_lookback <= 0 or recent_lookback > sequences.shape[1]:
        raise ModelAdapterError("recent_lookback must fit inside the source lookback")
    recent = sequences[:, -recent_lookback:, :]
    recent_mask = padding_mask[:, -recent_lookback:]
    recent_flag = np.ones((*recent.shape[:2], 1), dtype=np.float32)
    recent_flag[recent_mask] = 0.0
    recent = np.concatenate([recent, recent_flag], axis=2)
    if history_mode == "recent_only":
        return recent.astype(np.float32), recent_mask.copy()

    older_length = sequences.shape[1] - recent_lookback
    if older_length <= 0 or history_bins <= 0 or history_bins > older_length:
        raise ModelAdapterError(
            "Multiresolution input needs positive history bins within older history"
        )
    older = sequences[:, :older_length, :]
    older_mask = padding_mask[:, :older_length]
    boundaries = np.linspace(0, older_length, history_bins + 1, dtype=int)
    pooled = np.zeros(
        (len(sequences), history_bins, sequences.shape[2] + 1),
        dtype=np.float32,
    )
    pooled_mask = np.ones((len(sequences), history_bins), dtype=np.bool_)
    for bin_index, (start, stop) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        valid = ~older_mask[:, start:stop]
        counts = valid.sum(axis=1)
        available = counts > 0
        if not np.any(available):
            continue
        weighted = older[:, start:stop, :] * valid[:, :, None]
        pooled[available, bin_index, :-1] = (
            weighted[available].sum(axis=1) / counts[available, None]
        )
        pooled_mask[available, bin_index] = False
    combined = np.concatenate([pooled, recent], axis=1)
    combined_mask = np.concatenate([pooled_mask, recent_mask], axis=1)
    combined[combined_mask] = 0.0
    return combined, combined_mask


class HybridNeuralAdapter(NeuralModelAdapter):
    """Prepare fold-scaled sequences and fold-scaled engineered features."""

    representation = "heterogeneous"

    def _prepare_inputs(self, data: Any, *, fit: bool) -> NeuralInputs:
        if not hasattr(data, "tabular") or not hasattr(data, "sequence"):
            raise ModelAdapterError(
                "Hybrid models require aligned tabular and sequence datasets"
            )
        sequence = data.sequence
        if not bool(getattr(sequence, "scaled", False)):
            raise ModelAdapterError("Hybrid models require fold-scaled telemetry")
        source_sequences = np.asarray(sequence.sequences, dtype=np.float32)
        source_mask = np.asarray(sequence.padding_mask, dtype=np.bool_)
        side = np.asarray(sequence.side_features, dtype=np.float64)
        tabular, feature_names = tabular_values(data.tabular)
        channel_names = tuple(sequence.channel_names)
        side_names = tuple(sequence.side_feature_names)

        transformed_sequences, transformed_mask = build_resolution_view(
            source_sequences,
            source_mask,
            history_mode=str(self.hyperparameters["history_mode"]),
            recent_lookback=int(self.hyperparameters["recent_lookback"]),
            history_bins=int(self.hyperparameters["history_bins"]),
        )
        if fit:
            self.feature_names = feature_names
            self.channel_names = channel_names
            self.side_feature_names = side_names
            self.source_lookback = int(sequence.lookback)
            self.tabular_scaler = RobustScaler(
                quantile_range=(25.0, 75.0),
                unit_variance=True,
            )
            self.side_scaler = RobustScaler(
                quantile_range=(25.0, 75.0),
                unit_variance=True,
            )
            scaled_tabular = self.tabular_scaler.fit_transform(tabular)
            scaled_side = self.side_scaler.fit_transform(side)
        else:
            if feature_names != self.feature_names:
                raise ModelAdapterError("Hybrid tabular feature order differs")
            if channel_names != self.channel_names:
                raise ModelAdapterError("Hybrid telemetry channel order differs")
            if side_names != self.side_feature_names:
                raise ModelAdapterError("Hybrid side-feature order differs")
            if int(sequence.lookback) != self.source_lookback:
                raise ModelAdapterError("Hybrid source lookback differs")
            scaled_tabular = self.tabular_scaler.transform(tabular)
            scaled_side = self.side_scaler.transform(side)

        arrays = (transformed_sequences, scaled_side, scaled_tabular)
        if any(not np.isfinite(values).all() for values in arrays):
            raise ModelAdapterError("Hybrid input transformation produced non-finite values")
        return NeuralInputs(
            tensors=(
                torch.as_tensor(transformed_sequences, dtype=torch.float32),
                torch.as_tensor(transformed_mask, dtype=torch.bool),
                torch.as_tensor(scaled_side, dtype=torch.float32),
                torch.as_tensor(scaled_tabular, dtype=torch.float32),
            ),
            rows=len(data),
        )


class TabularEncoder(nn.Module):
    def __init__(self, input_features: int, hidden_units: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, hidden_units),
            nn.GELU(),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class HybridCNNRegressor(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        side_features: int,
        tabular_features: int,
        branch_channels: int,
        kernel_sizes: list[int],
        tabular_hidden_units: int,
        fusion_hidden_units: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            MultiScaleBranch(input_channels, branch_channels, size, dropout)
            for size in kernel_sizes
        )
        self.tabular = TabularEncoder(tabular_features, tabular_hidden_units, dropout)
        temporal_width = len(kernel_sizes) * branch_channels * 2
        self.output = nn.Sequential(
            nn.Linear(
                temporal_width + side_features + tabular_hidden_units,
                fusion_hidden_units,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_units, 1),
        )

    def forward(
        self,
        sequences: Tensor,
        padding_mask: Tensor,
        side_features: Tensor,
        tabular_features: Tensor,
    ) -> Tensor:
        hidden = sequences.transpose(1, 2)
        valid_mask = (~padding_mask).unsqueeze(1).to(hidden.dtype)
        pooled = []
        for branch in self.branches:
            branch_values = branch(hidden * valid_mask, valid_mask)
            counts = valid_mask.sum(dim=2).clamp_min(1.0)
            mean = (branch_values * valid_mask).sum(dim=2) / counts
            maximum = branch_values.masked_fill(valid_mask == 0, -torch.inf).amax(dim=2)
            pooled.extend([mean, maximum])
        features = torch.cat(
            [*pooled, side_features, self.tabular(tabular_features)],
            dim=1,
        )
        return self.output(features).squeeze(-1)


class HybridGRURegressor(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        side_features: int,
        tabular_features: int,
        layers: int,
        hidden_units: int,
        tabular_hidden_units: int,
        fusion_hidden_units: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_channels,
            hidden_units,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.tabular = TabularEncoder(tabular_features, tabular_hidden_units, dropout)
        self.output = nn.Sequential(
            nn.Linear(
                hidden_units + side_features + tabular_hidden_units,
                fusion_hidden_units,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_units, 1),
        )

    def forward(
        self,
        sequences: Tensor,
        padding_mask: Tensor,
        side_features: Tensor,
        tabular_features: Tensor,
    ) -> Tensor:
        lengths = (~padding_mask).sum(dim=1)
        if torch.any(lengths <= 0):
            raise ModelAdapterError("Hybrid GRU received an empty history")
        right_padded = torch.zeros_like(sequences)
        for row_index, length_value in enumerate(lengths.tolist()):
            length = int(length_value)
            right_padded[row_index, :length] = sequences[row_index, -length:]
        packed = rnn.pack_padded_sequence(
            right_padded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        self.gru.flatten_parameters()
        _, hidden = self.gru(packed)
        features = torch.cat(
            [
                self.dropout(hidden[-1]),
                side_features,
                self.tabular(tabular_features),
            ],
            dim=1,
        )
        return self.output(features).squeeze(-1)


class HybridCNNAdapter(HybridNeuralAdapter):
    family = "hybrid_cnn"

    def _build_network(self, training_inputs: NeuralInputs) -> nn.Module:
        sequences, _, side, tabular = training_inputs.tensors
        return HybridCNNRegressor(
            input_channels=int(sequences.shape[2]),
            side_features=int(side.shape[1]),
            tabular_features=int(tabular.shape[1]),
            branch_channels=int(self.hyperparameters["branch_channels"]),
            kernel_sizes=[int(value) for value in self.hyperparameters["kernel_sizes"]],
            tabular_hidden_units=int(self.hyperparameters["tabular_hidden_units"]),
            fusion_hidden_units=int(self.hyperparameters["fusion_hidden_units"]),
            dropout=float(self.hyperparameters["dropout"]),
        )


class HybridGRUAdapter(HybridNeuralAdapter):
    family = "hybrid_gru"

    def _build_network(self, training_inputs: NeuralInputs) -> nn.Module:
        if self.hyperparameters["direction"] != "unidirectional":
            raise ModelAdapterError("Only causal unidirectional hybrid GRU is supported")
        sequences, _, side, tabular = training_inputs.tensors
        return HybridGRURegressor(
            input_channels=int(sequences.shape[2]),
            side_features=int(side.shape[1]),
            tabular_features=int(tabular.shape[1]),
            layers=int(self.hyperparameters["layers"]),
            hidden_units=int(self.hyperparameters["hidden_units"]),
            tabular_hidden_units=int(self.hyperparameters["tabular_hidden_units"]),
            fusion_hidden_units=int(self.hyperparameters["fusion_hidden_units"]),
            dropout=float(self.hyperparameters["dropout"]),
        )
