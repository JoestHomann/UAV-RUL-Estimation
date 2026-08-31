from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPORTER_PATH = (
    REPOSITORY_ROOT
    / "2_architecture_experiments"
    / "1_pipeline_experiments"
    / "report_signal_family_ablation.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "signal_family_ablation_report",
    REPORTER_PATH,
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Cannot load {REPORTER_PATH}")
reporter = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(reporter)


class SignalFamilyAblationReportTests(unittest.TestCase):
    def test_paired_improvements_use_matching_model_and_outer_fold(self) -> None:
        records = []
        for experiment, feature_set, offset in (
            ("control", "signal_control", 0.0),
            ("treatment", "signal_family_19_21", 1.0),
        ):
            for fold in (0, 1):
                records.append(
                    {
                        "experiment": experiment,
                        "feature_set": feature_set,
                        "model_family": "xgboost",
                        "outer_fold": fold,
                        "r2": 0.70 + 0.01 * offset,
                        "rmse": 30.0 - offset,
                        "bias": 4.0 - offset,
                        "overprediction_rate": 0.60 - 0.02 * offset,
                        "rms_overprediction": 20.0 - offset,
                    }
                )
        paired = reporter.pair_with_control(pd.DataFrame(records), "control")
        treatment = paired.loc[paired["experiment"] == "treatment"]
        self.assertTrue((treatment["r2_improvement"] > 0.0).all())
        self.assertTrue((treatment["rmse_improvement"] > 0.0).all())
        self.assertTrue((treatment["overprediction_rate_improvement"] > 0.0).all())

        summary = reporter.summarize_pairs(paired)
        row = summary.loc[summary["experiment"] == "treatment"].iloc[0]
        self.assertAlmostEqual(row["mean_r2_improvement"], 0.01)
        self.assertAlmostEqual(row["mean_rmse_improvement"], 1.0)
        self.assertEqual(row["r2_fold_wins"], 2)


if __name__ == "__main__":
    unittest.main()
