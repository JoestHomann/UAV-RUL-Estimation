"""Implement a compact multi-scale temporal CNN for RUL regression."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.neural.neural_base import NeuralInputs
from models.neural.sequence_base import SequenceNeuralAdapter
from models.neural.tcn import CausalConv1d


class MultiScaleBranch(nn.Module):
    """Extract one temporal scale with two causal convolutions."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv_1 = CausalConv1d(
            input_channels,
            output_channels,
            kernel_size,
            dilation=1,
        )
        self.conv_2 = CausalConv1d(
            output_channels,
            output_channels,
            kernel_size,
            dilation=1,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, valid_mask: Tensor) -> Tensor:
        """Keep padded positions inert throughout the branch."""

        hidden = self.dropout(self.activation(self.conv_1(values)))
        hidden = hidden * valid_mask
        hidden = self.dropout(self.activation(self.conv_2(hidden)))
        return hidden * valid_mask


class MultiScaleCNNRegressor(nn.Module):
    """Combine short-, medium-, and long-range causal convolution features."""

    def __init__(
        self,
        input_channels: int,
        side_features: int,
        branch_channels: int,
        kernel_sizes: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        if not kernel_sizes or any(size <= 0 for size in kernel_sizes):
            raise ValueError("Multi-scale CNN kernel sizes must be positive")
        self.branches = nn.ModuleList(
            MultiScaleBranch(
                input_channels,
                branch_channels,
                kernel_size,
                dropout,
            )
            for kernel_size in kernel_sizes
        )
        pooled_width = len(kernel_sizes) * branch_channels * 2
        self.output = nn.Sequential(
            nn.Linear(pooled_width + side_features, branch_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_channels, 1),
        )

    @staticmethod
    def _masked_pool(hidden: Tensor, valid_mask: Tensor) -> Tensor:
        """Return masked mean and maximum summaries for one branch."""

        valid_counts = valid_mask.sum(dim=2).clamp_min(1.0)
        mean = (hidden * valid_mask).sum(dim=2) / valid_counts
        maximum = hidden.masked_fill(valid_mask == 0, -torch.inf).amax(dim=2)
        return torch.cat([mean, maximum], dim=1)

    def forward(
        self,
        sequences: Tensor,
        padding_mask: Tensor,
        side_features: Tensor,
    ) -> Tensor:
        """Encode every configured temporal scale and predict one RUL value."""

        hidden = sequences.transpose(1, 2)
        valid_mask = (~padding_mask).unsqueeze(1).to(hidden.dtype)
        hidden = hidden * valid_mask
        pooled = [
            self._masked_pool(branch(hidden, valid_mask), valid_mask)
            for branch in self.branches
        ]
        features = torch.cat([*pooled, side_features], dim=1)
        return self.output(features).squeeze(-1)


class MultiScaleCNNAdapter(SequenceNeuralAdapter):
    """Fit the multi-scale temporal CNN to fold-scaled sequence windows."""

    family = "multiscale_cnn"

    def _build_network(self, training_inputs: NeuralInputs) -> nn.Module:
        """Build parallel causal branches from the resolved candidate."""

        sequence_tensor, _, side_tensor = training_inputs.tensors
        return MultiScaleCNNRegressor(
            input_channels=int(sequence_tensor.shape[2]),
            side_features=int(side_tensor.shape[1]),
            branch_channels=int(self.hyperparameters["branch_channels"]),
            kernel_sizes=[
                int(value) for value in self.hyperparameters["kernel_sizes"]
            ],
            dropout=float(self.hyperparameters["dropout"]),
        )
