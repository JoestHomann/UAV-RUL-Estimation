"""Focused checks for PE_34's Phase 3 fit-local adaptive weighting."""

from dataclasses import dataclass
from pathlib import Path
from types import MethodType
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = (
    ROOT
    / "2_architecture_experiments"
    / "2_model_architecture_study"
    / "4_model_adapters"
)
sys.path.insert(0, str(ADAPTERS))

from models.tabular.residual_corrected_tree_ensemble import (  # noqa: E402
    ResidualCorrectedTreeEnsembleAdapter,
)


@dataclass
class Dataset:
    features: pd.DataFrame
    metadata: pd.DataFrame
    target: pd.Series | None
    sample_weights: pd.Series | None
    fitting_target: pd.Series | None

    def __len__(self) -> int:
        return len(self.features)


class ZeroDifficultyModel:
    def fit(self, training: Dataset, validation: Dataset | None) -> None:
        del training, validation

    def predict(self, held: Dataset) -> np.ndarray:
        return np.zeros(len(held), dtype=float)


class Phase3AdaptiveUAVDeploymentTests(unittest.TestCase):
    def test_fit_local_difficulty_preserves_mass_and_relative_multiplier(self) -> None:
        uavs = [f"UAV_{index:02d}" for index in range(8)]
        target = pd.Series([12.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        data = Dataset(
            features=pd.DataFrame({"signal": np.arange(8, dtype=float)}),
            metadata=pd.DataFrame(
                {
                    "sample_id": [f"sample_{uav}" for uav in uavs],
                    "scenario": ["development_01"] * 8,
                    "uav_id": uavs,
                    "cutoff": [20] * 8,
                }
            ),
            target=target,
            sample_weights=pd.Series(np.ones(8)),
            fitting_target=target.copy(),
        )
        model = object.__new__(ResidualCorrectedTreeEnsembleAdapter)
        model.seed = 13
        model.adaptive_uav_weighting = {
            "method": "adaptive_uav_1_5",
            "multiplier": 1.5,
            "hard_fraction": 0.25,
            "difficulty_folds": 3,
            "difficulty_seed": 20270910,
        }
        model._calibration_data = MethodType(lambda self, training: training, model)
        model._difficulty_model = MethodType(
            lambda self: ZeroDifficultyModel(),
            model,
        )

        weighted = model._adaptive_training_data(data)

        self.assertAlmostEqual(weighted.sample_weights.sum(), 8.0)
        self.assertEqual(model.last_adaptive_weighting["hard_uavs"], uavs[:2])
        hard = data.metadata.uav_id.isin(uavs[:2]).to_numpy()
        ratio = weighted.sample_weights.to_numpy() / data.sample_weights.to_numpy()
        self.assertAlmostEqual(ratio[hard][0] / ratio[~hard][0], 1.5)


if __name__ == "__main__":
    unittest.main()
