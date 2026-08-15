"""Implement neural tabular and temporal model adapters with PyTorch.

MLP, TCN, LSTM, and the conditional Transformer share one weighted training
loop, AdamW optimization, gradient clipping, deterministic seeding, early
stopping, persistence, and prediction behavior. Their network modules differ,
but the experiment runner sees the same ModelAdapter interface.
"""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import RobustScaler
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_, rnn
from torch.utils.data import DataLoader, TensorDataset

from base import (
    ModelAdapter,
    ModelAdapterError,
    TrainingSummary,
    root_mean_squared_error,
    sample_weight_values,
    tabular_values,
    target_values,
)


@dataclass(frozen=True)
class NeuralTrainingConfig:
    """Hold the optimization policy shared by all neural architectures."""

    batch_size: int
    maximum_epochs: int
    early_stopping_patience: int
    gradient_clip_global_norm: float

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ModelAdapterError("Neural batch size must be positive")
        if self.maximum_epochs <= 0:
            raise ModelAdapterError("Maximum neural epochs must be positive")
        if self.early_stopping_patience <= 0:
            raise ModelAdapterError("Early-stopping patience must be positive")
        if self.gradient_clip_global_norm <= 0:
            raise ModelAdapterError("Gradient clipping norm must be positive")


@dataclass(frozen=True)
class NeuralInputs:
    """Store the tensors passed positionally into one PyTorch network."""

    tensors: tuple[Tensor, ...]
    rows: int


class MLPRegressor(nn.Module):
    """Apply fully connected layers to engineered tabular features."""

    def __init__(
        self,
        input_features: int,
        hidden_layers: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        previous_width = input_features
        for width in hidden_layers:
            modules.extend(
                [
                    nn.Linear(previous_width, width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_width = width
        modules.append(nn.Linear(previous_width, 1))
        self.network = nn.Sequential(*modules)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


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
        hidden = self.dropout(self.activation(self.conv_1(values)))
        hidden = hidden * valid_mask
        hidden = self.dropout(self.activation(self.conv_2(hidden)))
        hidden = hidden * valid_mask
        return self.activation(hidden + self.residual(values)) * valid_mask


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
        hidden = sequences.transpose(1, 2)
        valid_mask = (~padding_mask).unsqueeze(1).to(hidden.dtype)
        hidden = hidden * valid_mask
        for block in self.blocks:
            hidden = block(hidden, valid_mask)
        final_state = hidden[:, :, -1]
        return self.output(torch.cat([final_state, side_features], dim=1)).squeeze(-1)


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
        return values + self.encoding[:, : values.shape[1]]


class TransformerRegressor(nn.Module):
    """Encode telemetry with masked self-attention and a final regression head."""

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
            # Padding is already handled by the explicit Boolean mask. Turning
            # nested tensors off avoids an irrelevant warning for pre-norm
            # encoder layers and keeps the tensor representation predictable.
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(model_width + side_features, 1)

    def forward(
        self,
        sequences: Tensor,
        padding_mask: Tensor,
        side_features: Tensor,
    ) -> Tensor:
        hidden = self.position_encoding(self.input_projection(sequences))
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        final_state = hidden[:, -1, :]
        return self.output(torch.cat([final_state, side_features], dim=1)).squeeze(-1)


class NeuralModelAdapter(ModelAdapter):
    """Share deterministic weighted training across all PyTorch networks."""

    stochastic = True

    def __init__(
        self,
        *,
        hyperparameters: dict[str, Any],
        seed: int,
        training_config: NeuralTrainingConfig,
        prediction_minimum: float = 0.0,
        training_epochs: int | None = None,
    ) -> None:
        super().__init__(
            hyperparameters=hyperparameters,
            seed=seed,
            prediction_minimum=prediction_minimum,
        )
        if training_epochs is not None and training_epochs <= 0:
            raise ModelAdapterError("Fixed training epochs must be positive")
        self.training_config = training_config
        self.training_epochs = training_epochs
        self.device = torch.device("cpu")

    @abstractmethod
    def _prepare_inputs(self, data: Any, *, fit: bool) -> NeuralInputs:
        """Convert an adapter dataset into ordered network tensors."""

    @abstractmethod
    def _build_network(self, training_inputs: NeuralInputs) -> nn.Module:
        """Create the architecture after its input dimensions are known."""

    def _set_reproducible_seed(self) -> None:
        """Seed Python, NumPy, and PyTorch and request deterministic kernels."""

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)

    def _prediction_loader(self, inputs: NeuralInputs) -> DataLoader:
        """Create a deterministic non-shuffled loader for inference."""

        return DataLoader(
            TensorDataset(*inputs.tensors),
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def _network_predictions(self, inputs: NeuralInputs) -> NDArray[np.float64]:
        """Run batched inference without gradient tracking."""

        self.network.eval()
        parts: list[NDArray[np.float64]] = []
        with torch.no_grad():
            for batch in self._prediction_loader(inputs):
                model_inputs = tuple(value.to(self.device) for value in batch)
                predictions = self.network(*model_inputs)
                parts.append(predictions.cpu().numpy().astype(np.float64))
        return np.concatenate(parts)

    def _validation_rmse(self, data: Any, inputs: NeuralInputs) -> float:
        """Evaluate clipped validation predictions with the tuning metric."""

        predictions = np.maximum(
            self._network_predictions(inputs),
            self.prediction_minimum,
        )
        return root_mean_squared_error(target_values(data), predictions)

    def fit(self, training_data: Any, validation_data: Any | None) -> TrainingSummary:
        started_at = self.start_timer()
        self._set_reproducible_seed()
        training_inputs = self._prepare_inputs(training_data, fit=True)
        validation_inputs = (
            self._prepare_inputs(validation_data, fit=False)
            if validation_data is not None
            else None
        )
        # Pandas may expose read-only NumPy views. torch.tensor deliberately
        # copies them into writable tensors owned by the training loop.
        targets = torch.tensor(target_values(training_data), dtype=torch.float32)
        weights = torch.tensor(
            sample_weight_values(training_data),
            dtype=torch.float32,
        )

        self.network = self._build_network(training_inputs).to(self.device)
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=float(self.hyperparameters["learning_rate"]),
            weight_decay=float(self.hyperparameters["weight_decay"]),
        )
        generator = torch.Generator().manual_seed(self.seed)
        training_loader = DataLoader(
            TensorDataset(*training_inputs.tensors, targets, weights),
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

        maximum_epochs = self.training_epochs or self.training_config.maximum_epochs
        use_early_stopping = (
            validation_data is not None and self.training_epochs is None
        )
        best_state: dict[str, Tensor] | None = None
        best_rmse = np.inf
        best_epoch: int | None = None
        epochs_without_improvement = 0
        epochs_completed = 0

        for epoch in range(1, maximum_epochs + 1):
            self.network.train()
            for batch in training_loader:
                *model_inputs, batch_targets, batch_weights = batch
                model_inputs = [value.to(self.device) for value in model_inputs]
                batch_targets = batch_targets.to(self.device)
                batch_weights = batch_weights.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                predictions = self.network(*model_inputs)
                squared_errors = torch.square(predictions - batch_targets)
                loss = torch.sum(batch_weights * squared_errors) / torch.sum(
                    batch_weights
                )
                loss.backward()
                clip_grad_norm_(
                    self.network.parameters(),
                    self.training_config.gradient_clip_global_norm,
                )
                optimizer.step()
            epochs_completed = epoch

            if use_early_stopping and validation_inputs is not None:
                validation_rmse = self._validation_rmse(
                    validation_data,
                    validation_inputs,
                )
                if validation_rmse < best_rmse:
                    best_rmse = validation_rmse
                    best_epoch = epoch
                    best_state = deepcopy(self.network.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if (
                        epochs_without_improvement
                        >= self.training_config.early_stopping_patience
                    ):
                        break

        if use_early_stopping and best_state is not None:
            self.network.load_state_dict(best_state)
        self._is_fitted = True

        final_validation_rmse = None
        validation_rows = 0
        if validation_data is not None and validation_inputs is not None:
            validation_rows = len(validation_data)
            final_validation_rmse = self._validation_rmse(
                validation_data,
                validation_inputs,
            )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.network.parameters()
            if parameter.requires_grad
        )
        self.training_summary = TrainingSummary(
            model_family=self.family,
            seed=self.seed,
            training_rows=len(training_data),
            validation_rows=validation_rows,
            training_seconds=self.elapsed_seconds(started_at),
            epochs_or_iterations=epochs_completed,
            best_epoch_or_iteration=best_epoch,
            best_validation_rmse=final_validation_rmse,
            trainable_parameters=int(trainable_parameters),
        )
        return self.training_summary

    def _predict_raw(self, data: Any) -> NDArray[np.float64]:
        inputs = self._prepare_inputs(data, fit=False)
        return self._network_predictions(inputs)


class MLPAdapter(NeuralModelAdapter):
    """Fit a robust-scaled multilayer perceptron on engineered features."""

    family = "mlp"
    representation = "tabular"

    def _prepare_inputs(self, data: Any, *, fit: bool) -> NeuralInputs:
        expected = None if fit else self.feature_names
        values, names = tabular_values(data, expected)
        if fit:
            self.feature_names = names
            self.feature_scaler = RobustScaler(
                with_centering=True,
                with_scaling=True,
                quantile_range=(25.0, 75.0),
                unit_variance=True,
            )
            transformed = self.feature_scaler.fit_transform(values)
        else:
            transformed = self.feature_scaler.transform(values)
        tensor = torch.as_tensor(transformed, dtype=torch.float32)
        return NeuralInputs((tensor,), len(data))

    def _build_network(self, training_inputs: NeuralInputs) -> nn.Module:
        input_features = int(training_inputs.tensors[0].shape[1])
        hidden_layers = [int(width) for width in self.hyperparameters["hidden_layers"]]
        if not hidden_layers or any(width <= 0 for width in hidden_layers):
            raise ModelAdapterError("MLP hidden layers must be positive")
        return MLPRegressor(
            input_features=input_features,
            hidden_layers=hidden_layers,
            dropout=float(self.hyperparameters["dropout"]),
        )


class SequenceNeuralAdapter(NeuralModelAdapter):
    """Share validation and side-feature scaling for temporal networks."""

    representation = "sequence"

    def _prepare_inputs(self, data: Any, *, fit: bool) -> NeuralInputs:
        sequences = np.asarray(getattr(data, "sequences", None), dtype=np.float32)
        padding_mask = np.asarray(
            getattr(data, "padding_mask", None),
            dtype=np.bool_,
        )
        side_features = np.asarray(
            getattr(data, "side_features", None),
            dtype=np.float64,
        )
        channel_names = tuple(getattr(data, "channel_names", ()))
        side_feature_names = tuple(getattr(data, "side_feature_names", ()))
        lookback = getattr(data, "lookback", None)
        scaled = getattr(data, "scaled", False)

        if not scaled:
            raise ModelAdapterError(
                "Sequence neural models require Step 3 fold-scaled telemetry"
            )
        if sequences.ndim != 3 or padding_mask.shape != sequences.shape[:2]:
            raise ModelAdapterError("Sequence or padding-mask shape is invalid")
        if side_features.ndim != 2 or side_features.shape[0] != len(data):
            raise ModelAdapterError("Sequence side-feature shape is invalid")
        if not np.isfinite(sequences).all() or not np.isfinite(side_features).all():
            raise ModelAdapterError("Sequence inputs contain non-finite values")

        if fit:
            self.channel_names = channel_names
            self.side_feature_names = side_feature_names
            self.lookback = int(lookback)
            self.side_feature_scaler = RobustScaler(
                with_centering=True,
                with_scaling=True,
                quantile_range=(25.0, 75.0),
                unit_variance=True,
            )
            scaled_side_features = self.side_feature_scaler.fit_transform(side_features)
        else:
            if channel_names != self.channel_names:
                raise ModelAdapterError(
                    "Sequence channel order differs from fitted order"
                )
            if side_feature_names != self.side_feature_names:
                raise ModelAdapterError(
                    "Sequence side-feature order differs from fitted order"
                )
            if int(lookback) != self.lookback:
                raise ModelAdapterError(
                    "Sequence lookback differs from fitted lookback"
                )
            scaled_side_features = self.side_feature_scaler.transform(side_features)

        return NeuralInputs(
            (
                torch.as_tensor(sequences, dtype=torch.float32),
                torch.as_tensor(padding_mask, dtype=torch.bool),
                torch.as_tensor(scaled_side_features, dtype=torch.float32),
            ),
            len(data),
        )


class TCNAdapter(SequenceNeuralAdapter):
    """Fit a temporal convolutional network to scaled telemetry windows."""

    family = "tcn"

    def _build_network(self, training_inputs: NeuralInputs) -> nn.Module:
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


class LSTMAdapter(SequenceNeuralAdapter):
    """Fit a packed unidirectional LSTM to scaled telemetry windows."""

    family = "lstm"

    def _build_network(self, training_inputs: NeuralInputs) -> nn.Module:
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


class TransformerAdapter(SequenceNeuralAdapter):
    """Fit the conditional small masked Transformer encoder."""

    family = "transformer"

    def _build_network(self, training_inputs: NeuralInputs) -> nn.Module:
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
