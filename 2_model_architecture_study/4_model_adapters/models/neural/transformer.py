"""Implement the optional masked Transformer encoder architecture."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from base import ModelAdapterError
from models.neural.neural_base import NeuralInputs
from models.neural.sequence_base import SequenceNeuralAdapter


class SinusoidalPositionEncoding(nn.Module):
    """Add deterministic sine/cosine positions to projected telemetry."""

    def __init__(self, model_width: int, maximum_length: int) -> None:
        super().__init__()
        positions = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, model_width, 2, dtype=torch.float32)
            * (-np.log(10_000.0) / model_width)
        )
        encoding = torch.zeros(maximum_length, model_width, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        odd_width = encoding[:, 1::2].shape[1]
        encoding[:, 1::2] = torch.cos(positions * frequencies[:odd_width])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, values: Tensor) -> Tensor:
        """Add the stored positions matching the active sequence length."""

        return values + self.encoding[:, : values.shape[1]]


class TransformerRegressor(nn.Module):
    """Encode telemetry with masked attention and a regression head."""

    def __init__(
        self,
        input_channels: int,
        side_features: int,
        maximum_length: int,
        encoder_layers: int,
        model_width: int,
        attention_heads: int,
        feed_forward_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if model_width % attention_heads != 0:
            raise ModelAdapterError(
                "Transformer model width must be divisible by attention heads"
            )
        self.input_projection = nn.Linear(input_channels, model_width)
        self.position_encoding = SinusoidalPositionEncoding(
            model_width,
            maximum_length,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_width,
            nhead=attention_heads,
            dim_feedforward=model_width * feed_forward_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=encoder_layers,
            # Padding is handled by the explicit Boolean mask. Disabling nested
            # tensors avoids an irrelevant warning for pre-norm encoder layers.
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(model_width + side_features, 1)

    def forward(
        self,
        sequences: Tensor,
        padding_mask: Tensor,
        side_features: Tensor,
    ) -> Tensor:
        """Apply masked attention and combine the final state with UAV age."""

        hidden = self.position_encoding(self.input_projection(sequences))
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        final_state = hidden[:, -1, :]
        return self.output(torch.cat([final_state, side_features], dim=1)).squeeze(-1)


class TransformerAdapter(SequenceNeuralAdapter):
    """Fit the conditional small masked Transformer encoder."""

    family = "transformer"

    def _build_network(self, training_inputs: NeuralInputs) -> nn.Module:
        """Build the configured encoder after validating position encoding."""

        if self.hyperparameters["position_encoding"] != "sinusoidal":
            raise ModelAdapterError("Only sinusoidal position encoding is supported")
        sequence_tensor, _, side_tensor = training_inputs.tensors
        return TransformerRegressor(
            input_channels=int(sequence_tensor.shape[2]),
            side_features=int(side_tensor.shape[1]),
            maximum_length=self.lookback,
            encoder_layers=int(self.hyperparameters["encoder_layers"]),
            model_width=int(self.hyperparameters["model_width"]),
            attention_heads=int(self.hyperparameters["attention_heads"]),
            feed_forward_ratio=int(self.hyperparameters["feed_forward_ratio"]),
            dropout=float(self.hyperparameters["dropout"]),
        )
