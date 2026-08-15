"""Implement the multilayer-perceptron architecture."""

from __future__ import annotations

from typing import Any

from sklearn.preprocessing import RobustScaler
import torch
from torch import Tensor, nn

from base import ModelAdapterError, tabular_values
from models.neural.neural_base import NeuralInputs, NeuralModelAdapter


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
        """Return one unconstrained RUL value per engineered feature row."""

        return self.network(features).squeeze(-1)


class MLPAdapter(NeuralModelAdapter):
    """Fit a robust-scaled multilayer perceptron on engineered features."""

    family = "mlp"
    representation = "tabular"

    def _prepare_inputs(self, data: Any, *, fit: bool) -> NeuralInputs:
        """Fit training-fold robust scaling and create one feature tensor."""

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
        """Build the configured hidden-layer stack for the observed width."""

        input_features = int(training_inputs.tensors[0].shape[1])
        hidden_layers = [int(width) for width in self.hyperparameters["hidden_layers"]]
        if not hidden_layers or any(width <= 0 for width in hidden_layers):
            raise ModelAdapterError("MLP hidden layers must be positive")
        return MLPRegressor(
            input_features=input_features,
            hidden_layers=hidden_layers,
            dropout=float(self.hyperparameters["dropout"]),
        )
