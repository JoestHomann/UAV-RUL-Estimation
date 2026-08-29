"""Smoke tests for the declarative experiment catalog."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANAGER_PATH = REPOSITORY_ROOT / "pipeline_experiments" / "run_experiments.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "pipeline_experiment_manager",
    MANAGER_PATH,
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MANAGER_PATH}")
run_experiments = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(run_experiments)


class PipelineExperimentCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = run_experiments._read_config(
            REPOSITORY_ROOT / "pipeline_experiments" / "pipeline_experiments.toml"
        )

    def test_catalog_has_phase1_profiles_and_named_experiments(self) -> None:
        experiments = run_experiments._experiments(self.config)
        self.assertIn("PE_run_1", experiments)
        self.assertIn("drift_ablation", self.config["profiles"])
        self.assertIn("current", self.config["scenario_profiles"])
        self.assertIn("dense_stride_5", self.config["prefix_variants"])

    def test_ready_experiment_uses_distinct_run_identities(self) -> None:
        experiment = run_experiments._experiment(self.config, "PE_run_1")
        self.assertEqual(experiment["phase_1_run_name"], "PE_run_1")
        self.assertEqual(experiment["phase_2_run_number"], 6)
        self.assertFalse(experiment["phase_3_enabled"])

    def test_stage_order_is_explicit(self) -> None:
        self.assertEqual(run_experiments.STAGES, ("phase1", "phase2", "phase3"))


if __name__ == "__main__":
    unittest.main()
