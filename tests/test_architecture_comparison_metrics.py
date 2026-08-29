"""Regression tests for Phase 2 architecture-comparison metric metadata."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DIR = (
    REPOSITORY_ROOT
    / "2_model_architecture_study"
    / "7_architecture_comparison"
)
if str(COMPARISON_DIR) not in sys.path:
    sys.path.insert(0, str(COMPARISON_DIR))

from architecture_comparison import (  # noqa: E402
    METRIC_DIFFERENCE_INTERPRETATIONS,
    METRICS,
)
from plot_architecture_comparison import (  # noqa: E402
    _offset_diagnostics,
    _overprediction_tail_diagnostics,
    _seed_averaged_predictions,
)


class ArchitectureComparisonMetricTests(unittest.TestCase):
    def test_every_reported_metric_has_a_paired_difference_interpretation(self) -> None:
        self.assertEqual(
            set(METRIC_DIFFERENCE_INTERPRETATIONS),
            set(METRICS),
        )

    def test_offset_and_tail_diagnostics_use_declared_residual_sign(self) -> None:
        predictions = pd.DataFrame(
            {
                "model_family": ["xgboost"] * 4,
                "seed": [13, 37, 13, 37],
                "outer_fold": [0, 0, 1, 1],
                "scenario": ["locked_01"] * 4,
                "sample_id": ["a", "a", "b", "b"],
                "uav_id": ["u1", "u1", "u2", "u2"],
                "cutoff": [50, 50, 75, 75],
                "y_true": [10.0, 10.0, 20.0, 20.0],
                "y_pred": [14.0, 16.0, 18.0, 20.0],
                "residual": [4.0, 6.0, -2.0, 0.0],
            }
        )
        averaged = _seed_averaged_predictions(predictions)
        self.assertEqual(averaged["y_pred"].tolist(), [15.0, 19.0])
        self.assertEqual(averaged["residual"].tolist(), [5.0, -1.0])

        offsets = _offset_diagnostics(averaged, prediction_minimum=0.0)
        unadjusted = offsets.loc[offsets["offset_cycles"] == 0.0].iloc[0]
        minus_three = offsets.loc[offsets["offset_cycles"] == 3.0].iloc[0]
        self.assertAlmostEqual(float(unadjusted["bias"]), 2.0)
        self.assertAlmostEqual(float(unadjusted["overprediction_rate"]), 0.5)
        self.assertAlmostEqual(float(minus_three["bias"]), -1.0)

        tails = _overprediction_tail_diagnostics(predictions)
        overall = tails.loc[tails["group_value"] == "overall"].iloc[0]
        self.assertEqual(int(overall["positive_count"]), 2)
        self.assertAlmostEqual(float(overall["positive_p90"]), 5.8)
        self.assertAlmostEqual(float(overall["positive_p95"]), 5.9)
        self.assertAlmostEqual(float(overall["positive_maximum"]), 6.0)


if __name__ == "__main__":
    unittest.main()
