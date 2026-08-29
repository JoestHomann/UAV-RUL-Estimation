"""Tests for development-only pipeline experiment comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PATH = (
    REPOSITORY_ROOT / "pipeline_experiments" / "compare_experiments.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "pipeline_experiment_comparison",
    COMPARISON_PATH,
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Cannot load {COMPARISON_PATH}")
compare_experiments = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(compare_experiments)


class PipelineExperimentComparisonTests(unittest.TestCase):
    def test_selection_scope_reads_step5_without_locked_artifacts(self) -> None:
        manager_dir = Path("C:/repository/pipeline_experiments")
        experiment = {
            "phase_2_run_number": 17,
            "architectures": ["extra_trees", "xgboost"],
            "scenario_profile": "early_and_middle",
            "target_profile": "capped_125",
            "prediction_profile": "symmetric",
        }
        rows = []
        for family, base_rmse in (("extra_trees", 11.0), ("xgboost", 10.0)):
            for outer_fold in (0, 1):
                rows.append(
                    {
                        "model_family": family,
                        "outer_fold": outer_fold,
                        "selected_within_family": True,
                        "mean_inner_rmse": base_rmse + outer_fold,
                        "mean_inner_r2": 0.9 - outer_fold * 0.01,
                        "mean_inner_bias": -0.5,
                        "mean_inner_overprediction_rate": 0.4,
                        "mean_inner_root_mean_squared_overprediction": 8.0,
                        "mean_inner_underprediction_rate": 0.6,
                    }
                )
        candidates = pd.DataFrame(rows)

        with patch.object(
            compare_experiments,
            "MANAGER_DIR",
            manager_dir,
        ), patch.object(
            compare_experiments,
            "_read_config",
            return_value={"experiments": {"test_cell": experiment}},
        ), patch.object(
            Path,
            "is_file",
            return_value=True,
        ), patch.object(
            compare_experiments.pd,
            "read_csv",
            return_value=candidates,
        ) as read_csv, patch.object(
            pd.DataFrame,
            "to_csv",
        ):
            result = compare_experiments.compare(
                Path("catalog.toml"),
                ["test_cell"],
                scope="selection",
                output_path=Path("selection.csv"),
            )

        selected_path = str(read_csv.call_args.args[0]).replace("\\", "/")
        self.assertIn("/5_inner_model_selection/candidate_results.csv", selected_path)
        self.assertNotIn("6_locked_outer_evaluation", selected_path)
        self.assertEqual(set(result["model_family"]), {"extra_trees", "xgboost"})
        self.assertTrue((result["outer_fold_studies"] == 2).all())
        xgboost = result.loc[result["model_family"] == "xgboost"].iloc[0]
        self.assertAlmostEqual(float(xgboost["inner_rmse_mean"]), 10.5)
        self.assertEqual(xgboost["comparison_scope"], "selection")


if __name__ == "__main__":
    unittest.main()
