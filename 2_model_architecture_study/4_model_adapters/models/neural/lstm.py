"""Implement the unidirectional LSTM architecture."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils import rnn

from base import ModelAdapterError
from models.neural.neural_base import NeuralInputs
from models.neural.sequence_base import SequenceNeuralAdapter


class LSTMRegressor(nn.Module):
    """Encode variable-length histories with a packed unidirectional LSTM."""

    def __init__(
        self,
        input_channels: int,
        side_features: int,
        layers: int,
        hidden_units: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_units,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_units + side_features, 1)

    @staticmethod
    def _right_padded(
        left_padded: Tensor,
        padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Move real readings to the front for packed-sequence processing."""

        lengths = (~padding_mask).sum(dim=1)
        right_padded = torch.zeros_like(left_padded)
        for row_index, length_value in enumerate(lengths.tolist()):
            length = int(length_value)
            right_padded[row_index, :length] = left_padded[row_index, -length:]
        return right_padded, lengths

    def forward(
        self,
        sequences: Tensor,
        padding_mask: Tensor,
        side_features: Tensor,
    ) -> Tensor:
        """Encode valid readings and combine the final state with UAV age."""

        right_padded, lengths = self._right_padded(sequences, padding_mask)
        packed = rnn.pack_padded_sequence(
            right_padded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden_state, _) = self.lstm(packed)
        final_state = self.dropout(hidden_state[-1])
        return self.output(torch.cat([final_state, side_features], dim=1)).squeeze(-1)


class LSTMAdapter(SequenceNeuralAdapter):
    """Fit a packed unidirectional LSTM to scaled telemetry windows."""

    family = "lstm"

    def _build_network(self, training_inputs: NeuralInputs) -> nn.Module:
        """Build the contract's causal recurrent architecture."""

        if self.hyperparameters["direction"] != "unidirectional":
            raise ModelAdapterError("Only the causal unidirectional LSTM is supported")
        sequence_tensor, _, side_tensor = training_inputs.tensors
        return LSTMRegressor(
            input_channels=int(sequence_tensor.shape[2]),
            side_features=int(side_tensor.shape[1]),
            layers=int(self.hyperparameters["layers"]),
            hidden_units=int(self.hyperparameters["hidden_units"]),
            dropout=float(self.hyperparameters["dropout"]),
        )
