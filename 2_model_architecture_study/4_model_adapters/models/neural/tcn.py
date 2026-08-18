"""Implement the temporal convolutional network architecture."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.neural.neural_base import NeuralInputs
from models.neural.sequence_base import SequenceNeuralAdapter


class CausalConv1d(nn.Conv1d):
    """Preserve length while preventing convolution from using future steps."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        self.right_trim = (kernel_size - 1) * dilation
        super().__init__(
            input_channels,
            output_channels,
            kernel_size,
            padding=self.right_trim,
            dilation=dilation,
        )

    def forward(self, values: Tensor) -> Tensor:
        """Trim padded future positions after the causal convolution."""

        convolved = super().forward(values)
        if self.right_trim == 0:
            return convolved
        return convolved[:, :, : -self.right_trim]


class TCNResidualBlock(nn.Module):
    """Apply two causal convolutions and a shape-safe residual connection."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # The canonical block also weight-normalizes both convolutions. That is
        # deliberately omitted: torch implements weight normalization as a
        # module parametrization, and parametrized modules cannot be pickled,
        # while Steps 5 and 6 persist fitted adapters with joblib. Controlled
        # runs showed no training benefit from it here, so the persistence
        # contract wins.
        self.conv_1 = CausalConv1d(
            input_channels,
            output_channels,
            kernel_size,
            dilation,
        )
        self.conv_2 = CausalConv1d(
            output_channels,
            output_channels,
            kernel_size,
            dilation,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )

    def forward(self, values: Tensor, valid_mask: Tensor) -> Tensor:
        """Keep padded positions zero through convolutions and the residual.

        The residual is added after the second activation and the sum is not
        passed through another one. Activating the sum would clip the identity
        path to non-negative values, so a block could only ever add magnitude
        to what it received, and any unit whose sum fell below zero would send
        no gradient to either the convolutions or the skip connection. Both
        effects compound with depth; the canonical temporal convolutional
        network adds the residual after the nonlinearity for exactly this
        reason.
        """

        hidden = self.dropout(self.activation(self.conv_1(values)))
        hidden = hidden * valid_mask
        hidden = self.dropout(self.activation(self.conv_2(hidden)))
        hidden = hidden * valid_mask
        return (hidden + self.residual(values)) * valid_mask


class TCNRegressor(nn.Module):
    """Encode telemetry with dilated causal convolutions and residual blocks."""

    def __init__(
        self,
        input_channels: int,
        side_features: int,
        residual_blocks: int,
        channels: int,
        kernel_size: int,
        dilation_base: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[TCNResidualBlock] = []
        previous_channels = input_channels
        for block_index in range(residual_blocks):
            blocks.append(
                TCNResidualBlock(
                    previous_channels,
                    channels,
                    kernel_size,
                    dilation=dilation_base**block_index,
                    dropout=dropout,
                )
            )
            previous_channels = channels
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Linear(channels + side_features, 1)

    def forward(
        self,
        sequences: Tensor,
        padding_mask: Tensor,
        side_features: Tensor,
    ) -> Tensor:
        """Encode the causal history and combine it with age side features."""

        hidden = sequences.transpose(1, 2)
        valid_mask = (~padding_mask).unsqueeze(1).to(hidden.dtype)
        hidden = hidden * valid_mask
        for block in self.blocks:
            hidden = block(hidden, valid_mask)
        final_state = hidden[:, :, -1]
        return self.output(torch.cat([final_state, side_features], dim=1)).squeeze(-1)


class TCNAdapter(SequenceNeuralAdapter):
    """Fit a temporal convolutional network to scaled telemetry windows."""

    family = "tcn"

    def _build_network(self, training_inputs: NeuralInputs) -> nn.Module:
        """Build the configured causal residual stack."""

        sequence_tensor, _, side_tensor = training_inputs.tensors
        return TCNRegressor(
            input_channels=int(sequence_tensor.shape[2]),
            side_features=int(side_tensor.shape[1]),
            residual_blocks=int(self.hyperparameters["residual_blocks"]),
            channels=int(self.hyperparameters["channels"]),
            kernel_size=int(self.hyperparameters["kernel_size"]),
            dilation_base=int(self.hyperparameters["dilation_base"]),
            dropout=float(self.hyperparameters["dropout"]),
        )
