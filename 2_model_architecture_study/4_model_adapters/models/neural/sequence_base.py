"""Provide shared input validation and side-feature scaling for sequence models."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import RobustScaler
import torch

from base import ModelAdapterError
from models.neural.neural_base import NeuralInputs, NeuralModelAdapter


class SequenceNeuralAdapter(NeuralModelAdapter):
    """Prepare Step 3 sequence tensors consistently for temporal networks."""

    representation = "sequence"

    def _prepare_inputs(self, data: Any, *, fit: bool) -> NeuralInputs:
        """Validate tensors and fit only the two age side-feature scales."""

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
