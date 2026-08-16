"""Provide the shared weighted training loop for all neural architectures."""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import random
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from base import (
    ModelAdapter,
    ModelAdapterError,
    TrainingSummary,
    root_mean_squared_error,
    sample_weight_values,
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
        """Reject invalid training values before any network is constructed."""

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
    """Store tensors passed positionally into one PyTorch network."""

    tensors: tuple[Tensor, ...]
    rows: int


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
        training_monitor: Any | None = None,
    ) -> None:
        """Store the shared optimizer policy and optional fixed epoch count."""

        super().__init__(
            hyperparameters=hyperparameters,
            seed=seed,
            prediction_minimum=prediction_minimum,
            training_monitor=training_monitor,
        )
        if training_epochs is not None and training_epochs <= 0:
            raise ModelAdapterError("Fixed training epochs must be positive")
        self.training_config = training_config
        self.training_epochs = training_epochs
        # Every tensor and module in this class already flows through
        # self.device (see _prepare_inputs callers, _network_predictions,
        # and the training loop below), so auto-detecting CUDA here is
        # enough to move the whole training loop onto a GPU when one is
        # available; no other method needs to change. CUBLAS_WORKSPACE_CONFIG
        # must already be set (by the entry-point script, before torch's
        # CUDA context is created) for torch.use_deterministic_algorithms(True)
        # to work correctly on CUDA; see _set_reproducible_seed.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        """Fit one network with weighted MSE and optional early stopping."""

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
            epoch_started_at = perf_counter()
            epoch_weighted_squared_error = 0.0
            epoch_weight_total = 0.0
            self.network.train()
            for batch in training_loader:
                *model_inputs, batch_targets, batch_weights = batch
                model_inputs = [value.to(self.device) for value in model_inputs]
                batch_targets = batch_targets.to(self.device)
                batch_weights = batch_weights.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                predictions = self.network(*model_inputs)
                squared_errors = torch.square(predictions - batch_targets)
                weighted_squared_error = torch.sum(batch_weights * squared_errors)
                batch_weight_total = torch.sum(batch_weights)
                loss = weighted_squared_error / batch_weight_total
                loss.backward()
                clip_grad_norm_(
                    self.network.parameters(),
                    self.training_config.gradient_clip_global_norm,
                )
                optimizer.step()
                # Accumulate detached values so TensorBoard receives the exact
                # sample-weighted epoch MSE without retaining computation graphs.
                epoch_weighted_squared_error += float(
                    weighted_squared_error.detach().cpu().item()
                )
                epoch_weight_total += float(batch_weight_total.detach().cpu().item())
            epochs_completed = epoch

            validation_rmse: float | None = None
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

            progress_scalars: dict[str, Any] = {
                "optimization/training_weighted_mse": (
                    epoch_weighted_squared_error / epoch_weight_total
                ),
                "optimization/learning_rate": optimizer.param_groups[0]["lr"],
                "timing/seconds_per_epoch": perf_counter() - epoch_started_at,
            }
            if validation_rmse is not None:
                progress_scalars.update(
                    {
                        "optimization/validation_rmse": validation_rmse,
                        "optimization/best_validation_rmse": best_rmse,
                        "early_stopping/patience_used": epochs_without_improvement,
                        "early_stopping/patience_remaining": (
                            self.training_config.early_stopping_patience
                            - epochs_without_improvement
                        ),
                    }
                )
            self.log_training_step(step=epoch, scalars=progress_scalars)

            if (
                use_early_stopping
                and epochs_without_improvement
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
        """Convert data with the fitted adapter state and run the network."""

        inputs = self._prepare_inputs(data, fit=False)
        return self._network_predictions(inputs)
