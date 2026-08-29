"""Smoke tests for the declarative experiment catalog."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tomllib
import unittest
from unittest.mock import patch


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

    def test_target_scenario_matrix_is_explicit_and_unconfounded(self) -> None:
        cells = {
            "PE_run_1": ("current", "raw"),
            "PE_2x2_current_cap125": ("current", "capped_125"),
            "PE_2x2_early_raw": ("early_and_middle", "raw"),
            "PE_2x2_early_cap125": ("early_and_middle", "capped_125"),
        }
        for name, (scenario, target) in cells.items():
            experiment = run_experiments._experiment(self.config, name)
            self.assertTrue(experiment["enabled"])
            self.assertEqual(experiment["scenario_profile"], scenario)
            self.assertEqual(experiment["target_profile"], target)
            self.assertEqual(experiment["prediction_profile"], "symmetric")
            self.assertEqual(experiment["prefix_variant"], "current20")
            self.assertEqual(experiment["feature_set"], "screened_drift_pruned")
            self.assertEqual(experiment["candidate_budget"], 50)

        for name in cells.keys() - {"PE_run_1"}:
            experiment = run_experiments._experiment(self.config, name)
            self.assertEqual(experiment["architectures"], ["extra_trees", "xgboost"])
            self.assertEqual(experiment["phase_2_scope"], "selection_only")
        self.assertEqual(
            run_experiments._experiment(self.config, "PE_run_1")["phase_2_scope"],
            "complete",
        )

    def test_early_middle_matrix_cells_share_phase1_artifacts(self) -> None:
        raw = run_experiments._experiment(self.config, "PE_2x2_early_raw")
        capped = run_experiments._experiment(
            self.config,
            "PE_2x2_early_cap125",
        )
        self.assertEqual(raw["phase_1_run_name"], capped["phase_1_run_name"])
        self.assertEqual(raw["phase_1_mode"], "rebuild")
        self.assertEqual(capped["phase_1_mode"], "reuse")

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

    def test_pipeline_phase2_artifacts_are_experiment_owned(self) -> None:
        paths = run_experiments._phase2_paths("PE_run_1")
        expected = (
            REPOSITORY_ROOT
            / "pipeline_experiments"
            / "runs"
            / "PE_run_1"
            / "phase2"
        )
        self.assertEqual(paths["root"], expected)
        self.assertEqual(
            paths["specification"],
            expected
            / "1_architecture_study_settings"
            / "artifacts"
            / "experiment_specification.json",
        )

    def test_pipeline_phase3_settings_reference_experiment_phase2_root(self) -> None:
        experiment = copy.deepcopy(
            run_experiments._experiment(self.config, "PE_run_1")
        )
        experiment["phase_3_selected_model_family"] = "xgboost"
        with patch.object(run_experiments, "_write_json"):
            _, payload = run_experiments._phase3_settings(
                "PE_run_1",
                self.config,
                experiment,
            )
        self.assertEqual(
            payload["phase_2_run_root"],
            "pipeline_experiments/runs/PE_run_1/phase2",
        )

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

    def test_phase2_scope_rejects_unknown_values(self) -> None:
        with self.assertRaisesRegex(
            run_experiments.ExperimentManagerError,
            "phase_2_scope must be one of",
        ):
            run_experiments._phase2_scope({"phase_2_scope": "locked_almost"})

    def test_selection_only_scope_never_dispatches_locked_steps(self) -> None:
        experiment = copy.deepcopy(
            run_experiments._experiment(
                self.config,
                "PE_2x2_current_cap125",
            )
        )
        commands: list[list[str]] = []

        def record(command: list[str], *, label: str) -> None:
            del label
            commands.append(command)

        with patch.object(
            run_experiments,
            "_load_interface",
            return_value=(Path("interface.json"), {}),
        ), patch.object(
            run_experiments,
            "_phase2_settings",
            return_value={},
        ), patch.object(
            run_experiments,
            "_write_json",
        ), patch.object(
            run_experiments,
            "_run_command",
            side_effect=record,
        ):
            run_experiments._run_phase2(
                "PE_2x2_current_cap125",
                self.config,
                experiment,
            )

        orchestrator = commands[-1]
        self.assertIn("--through-step", orchestrator)
        self.assertEqual(
            orchestrator[orchestrator.index("--through-step") + 1],
            "5",
        )
        self.assertFalse(
            any("run_architecture_comparison.py" in part for command in commands for part in command)
        )

    def test_promoted_scope_dispatches_locked_evaluation_and_comparison(self) -> None:
        experiment = copy.deepcopy(
            run_experiments._experiment(
                self.config,
                "PE_2x2_current_cap125",
            )
        )
        experiment["phase_2_scope"] = "complete"
        commands: list[list[str]] = []

        def record(command: list[str], *, label: str) -> None:
            del label
            commands.append(command)

        with patch.object(
            run_experiments,
            "_load_interface",
            return_value=(Path("interface.json"), {}),
        ), patch.object(
            run_experiments,
            "_phase2_settings",
            return_value={},
        ), patch.object(
            run_experiments,
            "_write_json",
        ), patch.object(
            run_experiments,
            "_run_command",
            side_effect=record,
        ):
            run_experiments._run_phase2(
                "PE_2x2_current_cap125",
                self.config,
                experiment,
            )

        orchestrator = commands[-2]
        self.assertEqual(
            orchestrator[orchestrator.index("--through-step") + 1],
            "6",
        )
        self.assertTrue(
            any("run_architecture_comparison.py" in part for part in commands[-1])
        )

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
