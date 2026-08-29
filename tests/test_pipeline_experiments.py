"""Smoke tests for the declarative experiment catalog."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tomllib
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
        self.assertEqual(run_experiments._configured_max_workers(self.config), 6)

    def test_ready_experiment_uses_distinct_run_identities(self) -> None:
        experiment = run_experiments._experiment(self.config, "PE_run_1")
        self.assertEqual(experiment["phase_1_run_name"], "PE_run_1")
        self.assertEqual(experiment["phase_2_run_number"], 6)
        self.assertEqual(
            experiment["architectures"],
            ["mean_baseline", "random_forest", "extra_trees", "xgboost"],
        )
        self.assertNotIn("max_workers", experiment)
        self.assertFalse(experiment["phase_3_enabled"])

    def test_future_pipeline_example_uses_focused_tree_controls(self) -> None:
        experiment = run_experiments._experiment(
            self.config,
            "PE_conservative_cap125",
        )
        self.assertEqual(experiment["architectures"], ["extra_trees", "xgboost"])

    def test_catboost_search_excludes_the_slow_tail(self) -> None:
        settings_path = (
            REPOSITORY_ROOT
            / "pipeline_experiments"
            / "phase_2_settings.toml"
        )
        with settings_path.open("rb") as settings_file:
            settings = tomllib.load(settings_file)
        search = settings["architectures"]["catboost"]["search"]
        self.assertEqual(search["maximum_trees"]["value"], 1000)
        self.assertEqual(search["depth"]["values"], [4, 6, 8])

    def test_pipeline_phase2_settings_are_independent(self) -> None:
        paths = run_experiments._paths(self.config)
        pipeline_settings = paths["phase_2_settings"]
        standalone_settings = (
            REPOSITORY_ROOT
            / "2_model_architecture_study"
            / "1_architecture_study_settings"
            / "architecture_study_settings.toml"
        )
        self.assertEqual(
            pipeline_settings,
            REPOSITORY_ROOT / "pipeline_experiments" / "phase_2_settings.toml",
        )
        self.assertNotEqual(pipeline_settings, standalone_settings)

    def test_pipeline_rejects_the_standalone_phase2_toml(self) -> None:
        config = copy.deepcopy(self.config)
        config["paths"]["phase_2_settings"] = (
            "2_model_architecture_study/1_architecture_study_settings/"
            "architecture_study_settings.toml"
        )
        experiment = run_experiments._experiment(config, "PE_run_1")
        interface_path, interface = run_experiments._load_interface(experiment)
        with self.assertRaisesRegex(
            run_experiments.ExperimentManagerError,
            "must point inside pipeline_experiments",
        ):
            run_experiments._phase2_settings(
                config,
                experiment,
                interface,
                interface_path,
            )

    def test_stage_order_is_explicit(self) -> None:
        self.assertEqual(run_experiments.STAGES, ("phase1", "phase2", "phase3"))

    def test_experiment_declares_default_stage_range(self) -> None:
        experiment = run_experiments._experiment(self.config, "PE_run_1")
        self.assertEqual(experiment["from_stage"], "phase2")
        self.assertEqual(experiment["through_stage"], "phase2")
        self.assertEqual(
            run_experiments._resolve_stage(experiment.get("from_stage"), key="from_stage", default="phase1"),
            "phase2",
        )


if __name__ == "__main__":
    unittest.main()
