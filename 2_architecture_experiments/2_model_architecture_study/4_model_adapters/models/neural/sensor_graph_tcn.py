"""Implement a sensor-correlation graph followed by a causal TCN."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from base import ModelAdapterError
from models.neural.neural_base import NeuralInputs
from models.neural.sequence_base import SequenceNeuralAdapter
from models.neural.tcn import TCNResidualBlock


class SensorGraphLayer(nn.Module):
    """Mix each sensor with correlated neighbours at every time step."""

    def __init__(self, input_width: int, output_width: int, dropout: float) -> None:
        super().__init__()
        self.self_projection = nn.Linear(input_width, output_width)
        self.neighbour_projection = nn.Linear(
            input_width,
            output_width,
            bias=False,
        )
        self.normalization = nn.LayerNorm(output_width)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, adjacency: Tensor) -> Tensor:
        """Apply one residual-free graph message-passing layer."""

        neighbours = torch.einsum("ij,btjf->btif", adjacency, values)
        hidden = self.self_projection(values) + self.neighbour_projection(neighbours)
        return self.dropout(self.activation(self.normalization(hidden)))


class SensorGraphTCNRegressor(nn.Module):
    """Model cross-sensor structure before causal temporal dependencies."""

    def __init__(
        self,
        *,
        sensor_count: int,
        side_features: int,
        adjacency: Tensor,
        graph_hidden: int,
        graph_layers: int,
        temporal_blocks: int,
        temporal_channels: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency, persistent=True)
        graph_modules: list[SensorGraphLayer] = []
        input_width = 1
        for _ in range(graph_layers):
            graph_modules.append(
                SensorGraphLayer(input_width, graph_hidden, dropout)
            )
            input_width = graph_hidden
        self.graph_layers = nn.ModuleList(graph_modules)

        temporal_modules: list[TCNResidualBlock] = []
        temporal_input = sensor_count * graph_hidden
        for block_index in range(temporal_blocks):
            temporal_modules.append(
                TCNResidualBlock(
                    temporal_input,
                    temporal_channels,
                    kernel_size,
                    dilation=2**block_index,
                    dropout=dropout,
                )
            )
            temporal_input = temporal_channels
        self.temporal_blocks = nn.ModuleList(temporal_modules)
        self.output = nn.Linear(temporal_channels + side_features, 1)

    def forward(
        self,
        sequences: Tensor,
        padding_mask: Tensor,
        side_features: Tensor,
    ) -> Tensor:
        """Encode graph-aware telemetry and use the causal final state."""

        valid_time = (~padding_mask).unsqueeze(-1).unsqueeze(-1)
        hidden = sequences.unsqueeze(-1) * valid_time.to(sequences.dtype)
        for layer in self.graph_layers:
            hidden = layer(hidden, self.adjacency)
            hidden = hidden * valid_time.to(hidden.dtype)

        batch_size, time_steps, sensors, graph_width = hidden.shape
        hidden = hidden.reshape(batch_size, time_steps, sensors * graph_width)
        hidden = hidden.transpose(1, 2)
        temporal_mask = (~padding_mask).unsqueeze(1).to(hidden.dtype)
        for block in self.temporal_blocks:
            hidden = block(hidden, temporal_mask)
        final_state = hidden[:, :, -1]
        return self.output(torch.cat([final_state, side_features], dim=1)).squeeze(-1)


class SensorGraphTCNAdapter(SequenceNeuralAdapter):
    """Fit a fold-specific sensor graph and graph-temporal regressor."""

    family = "sensor_graph_tcn"

    def _prepare_inputs(self, data: Any, *, fit: bool) -> NeuralInputs:
        """Prepare shared sequence tensors and fit the graph on training rows."""

        inputs = super()._prepare_inputs(data, fit=fit)
        if fit:
            sequences, padding_mask, _ = inputs.tensors
            valid_values = sequences.numpy()[~padding_mask.numpy()]
            self.graph_adjacency = self._fit_adjacency(valid_values)
        return inputs

    def _fit_adjacency(self, values: np.ndarray) -> Tensor:
        """Build a symmetric top-k absolute-correlation sensor graph."""

        if values.ndim != 2 or values.shape[0] < 2:
            raise ModelAdapterError(
                "Sensor graph needs at least two valid telemetry rows"
            )
        correlations = np.corrcoef(values, rowvar=False)
        correlations = np.nan_to_num(
            np.abs(correlations),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        np.fill_diagonal(correlations, 0.0)

        sensor_count = correlations.shape[0]
        neighbours = min(
            int(self.hyperparameters["graph_neighbors"]),
            sensor_count - 1,
        )
        if neighbours <= 0:
            raise ModelAdapterError("Sensor graph needs at least two channels")

        selected = np.zeros_like(correlations, dtype=bool)
        for sensor_index in range(sensor_count):
            indices = np.argpartition(
                correlations[sensor_index],
                -neighbours,
            )[-neighbours:]
            selected[sensor_index, indices] = True
        adjacency = np.where(selected | selected.T, correlations, 0.0)
        adjacency = np.maximum(adjacency, adjacency.T)
        degrees = adjacency.sum(axis=1)
        if np.any(degrees <= 0.0):
            raise ModelAdapterError(
                "Fold-fitted sensor graph contains an isolated channel"
            )
        inverse_sqrt_degree = 1.0 / np.sqrt(degrees)
        normalized = (
            inverse_sqrt_degree[:, None]
            * adjacency
            * inverse_sqrt_degree[None, :]
        )
        return torch.as_tensor(normalized, dtype=torch.float32)

    def _build_network(self, training_inputs: NeuralInputs) -> nn.Module:
        """Build graph layers and the causal temporal residual stack."""

        sequence_tensor, _, side_tensor = training_inputs.tensors
        return SensorGraphTCNRegressor(
            sensor_count=int(sequence_tensor.shape[2]),
            side_features=int(side_tensor.shape[1]),
            adjacency=self.graph_adjacency,
            graph_hidden=int(self.hyperparameters["graph_hidden"]),
            graph_layers=int(self.hyperparameters["graph_layers"]),
            temporal_blocks=int(self.hyperparameters["temporal_blocks"]),
            temporal_channels=int(self.hyperparameters["temporal_channels"]),
            kernel_size=int(self.hyperparameters["kernel_size"]),
            dropout=float(self.hyperparameters["dropout"]),
        )
