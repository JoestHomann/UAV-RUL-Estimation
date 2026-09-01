"""Smoke tests for the declarative experiment catalog."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import unittest
from unittest.mock import patch

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_EXPERIMENTS_ROOT = (
    REPOSITORY_ROOT / "2_architecture_experiments" / "1_pipeline_experiments"
)
MODEL_ARCHITECTURE_ROOT = (
    REPOSITORY_ROOT / "2_architecture_experiments" / "2_model_architecture_study"
)
MANAGER_PATH = PIPELINE_EXPERIMENTS_ROOT / "run_experiments.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "pipeline_experiment_manager",
    MANAGER_PATH,
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MANAGER_PATH}")
run_experiments = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(run_experiments)

import report_ensemble_calibration
import promote_calibrated_ensemble
import conditional_safety_calibration


class PipelineExperimentCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = run_experiments._read_config(
            PIPELINE_EXPERIMENTS_ROOT / "pipeline_experiments.toml"
        )

    def test_catalog_has_phase1_profiles_and_named_experiments(self) -> None:
        experiments = run_experiments._experiments(self.config)
        self.assertIn("PE_1", experiments)
        self.assertIn("drift_ablation", self.config["profiles"])
        self.assertIn("signal_family_ablation", self.config["profiles"])
        self.assertIn("current", self.config["scenario_profiles"])
        self.assertIn("dense_stride_5", self.config["prefix_variants"])
        self.assertEqual(run_experiments._configured_max_workers(self.config), 6)

    def test_each_pipeline_run_has_one_settings_file_and_entry_point(self) -> None:
        experiments_root = PIPELINE_EXPERIMENTS_ROOT / "experiments"
        expected_steps = {1: 1, 2: 7, 3: 1, 4: 1, 5: 1}
        for run_number, step_count in expected_steps.items():
            experiment_name = f"PE_{run_number}"
            experiment_dir = experiments_root / experiment_name
            toml_files = list(experiment_dir.glob("*.toml"))
            launchers = list(experiment_dir.glob("*.py"))
            self.assertEqual(toml_files, [experiment_dir / "settings.toml"])
            self.assertEqual(launchers, [experiment_dir / "run.py"])
            config = run_experiments._read_config(toml_files[0])
            definition = config["run_definitions"][experiment_name]
            self.assertEqual(definition["pipeline_experiment"], experiment_name)
            self.assertEqual(definition["pipeline_run"], "run_1")
            self.assertTrue((experiment_dir / "runs" / "run_1").is_dir())
            self.assertEqual(len(definition["steps"]), step_count)
            for step in definition["steps"]:
                self.assertEqual(
                    step["script"],
                    "2_architecture_experiments/1_pipeline_experiments/run_experiments.py",
                )
                self.assertTrue(
                    (REPOSITORY_ROOT / step["script"]).is_file(),
                    step["script"],
                )
                chain = config["script_chains"][step["script_chain"]]
                self.assertGreater(len(chain["scripts"]), 0)
                for script in chain["scripts"]:
                    self.assertTrue((REPOSITORY_ROOT / script).is_file(), script)

    def test_each_pipeline_run_launcher_lists_its_reviewable_plan(self) -> None:
        experiments_root = PIPELINE_EXPERIMENTS_ROOT / "experiments"
        for run_number in range(1, 6):
            experiment_name = f"PE_{run_number}"
            launcher = experiments_root / experiment_name / "run.py"
            completed = subprocess.run(
                [sys.executable, str(launcher), "--list"],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(f"{experiment_name}:", completed.stdout)
            self.assertIn("settings.toml", completed.stdout)
            self.assertIn(f"PE_{run_number}\\runs\\run_1", completed.stdout)
            self.assertIn("Execution plan:", completed.stdout)

    def test_legacy_pipeline_run_path_resolves_after_restructure(self) -> None:
        legacy = (
            "pipeline_experiments/runs/PE_run_3/PE3_features_drift/phase2/"
            "2_tabular_data_adapter/artifacts/tabular_dataset_manifest.json"
        )
        resolved = run_experiments.repository_path(REPOSITORY_ROOT, legacy)
        expected = (
            PIPELINE_EXPERIMENTS_ROOT
            / "experiments"
            / "PE_3"
            / "runs"
            / "run_1"
            / "PE3_features_drift"
            / "phase2"
            / "2_tabular_data_adapter"
            / "artifacts"
            / "tabular_dataset_manifest.json"
        )
        self.assertEqual(resolved, expected.resolve())
        self.assertTrue(resolved.is_file())

    def test_pe5_defines_exactly_four_target_submission_variants(self) -> None:
        settings = (
            PIPELINE_EXPERIMENTS_ROOT
            / "experiments"
            / "PE_5"
            / "settings.toml"
        )
        config = run_experiments._read_config(settings)
        workflows = run_experiments._target_submission_workflows(config)
        self.assertEqual(set(workflows), {"PE_5"})
        self.assertEqual(
            set(workflows["PE_5"]["variants"]),
            {"hard_cap_125", "raw", "weighted_raw", "soft_tail"},
        )

    def test_pipeline_run_structure_has_no_legacy_definition_surface(self) -> None:
        self.assertFalse((PIPELINE_EXPERIMENTS_ROOT / "definitions").exists())
        self.assertFalse((PIPELINE_EXPERIMENTS_ROOT / "phase_2_settings.toml").exists())
        self.assertTrue(
            (PIPELINE_EXPERIMENTS_ROOT / "_internal" / "shared_settings.toml").is_file()
        )
        self.assertTrue(
            (
                PIPELINE_EXPERIMENTS_ROOT
                / "_internal"
                / "phase_2_base_settings.toml"
            ).is_file()
        )

    def test_each_settings_file_declares_its_run_identity_once(self) -> None:
        experiments_root = PIPELINE_EXPERIMENTS_ROOT / "experiments"
        for number in range(1, 5):
            experiment_name = f"PE_{number}"
            settings_path = experiments_root / experiment_name / "settings.toml"
            with settings_path.open("rb") as stream:
                raw = tomllib.load(stream)
            self.assertEqual(
                raw["pipeline"],
                {"experiment": experiment_name, "run": "run_1"},
            )
            for table_name in (
                "run_definitions",
                "experiments",
                "experiment_groups",
                "experiment_workflows",
                "conditional_calibration_workflows",
            ):
                for specification in raw.get(table_name, {}).values():
                    self.assertNotIn("pipeline_experiment", specification)
                    self.assertNotIn("pipeline_run", specification)
            config = run_experiments._read_config(settings_path)
            definition = config["run_definitions"][experiment_name]
            self.assertEqual(definition["pipeline_experiment"], experiment_name)
            self.assertEqual(definition["pipeline_run"], "run_1")

        pe2 = run_experiments._read_config(
            experiments_root / "PE_2" / "settings.toml"
        )
        self.assertEqual(pe2["experiments"]["PE_1"]["pipeline_experiment"], "PE_1")
        self.assertEqual(pe2["experiments"]["PE_1"]["pipeline_run"], "run_1")

    def test_compatibility_catalog_composes_all_run_definitions(self) -> None:
        self.assertEqual(
            set(self.config["run_definitions"]),
            {"PE_1", "PE_2", "PE_3", "PE_4"},
        )
        self.assertEqual(len(run_experiments._experiments(self.config)), 36)
        self.assertIn(
            "PE_target_scenario_2x2",
            run_experiments._experiment_groups(self.config),
        )
        target_group = run_experiments._experiment_group(
            self.config,
            "PE_target_scenario_2x2",
        )
        self.assertEqual(target_group["model_families"], ["extra_trees", "xgboost"])

    def test_legacy_phase2_prefixes_resolve_after_repository_move(self) -> None:
        self.assertEqual(
            run_experiments._repo_path(
                "pipeline_experiments/run_experiments.py",
                description="legacy pipeline manager",
            ),
            PIPELINE_EXPERIMENTS_ROOT / "run_experiments.py",
        )
        self.assertEqual(
            run_experiments._repo_path(
                "2_model_architecture_study/run_phase_2.py",
                description="legacy architecture runner",
            ),
            MODEL_ARCHITECTURE_ROOT / "run_phase_2.py",
        )

    def test_signal_family_ablation_is_an_explicit_development_group(self) -> None:
        group = run_experiments._experiment_group(
            self.config,
            "PE_signal_family_ablation",
        )
        self.assertEqual(group["control"], "PE_signal_control")
        self.assertEqual(len(group["experiments"]), 6)
        observed_sets = []
        for name in group["experiments"]:
            experiment = run_experiments._experiment(self.config, name)
            self.assertEqual(experiment["phase_2_scope"], "selection_only")
            self.assertEqual(experiment["architectures"], ["extra_trees", "xgboost"])
            self.assertEqual(experiment["candidate_budget"], 25)
            self.assertFalse(experiment["phase_3_enabled"])
            observed_sets.append(experiment["feature_set"])
        self.assertEqual(
            observed_sets,
            self.config["profiles"]["signal_family_ablation"]["feature_sets"],
        )

    def test_recommended_experiment_groups_are_development_only(self) -> None:
        group_names = {
            "PE_failure_cycle_target",
            "PE_baseline_normalization",
            "PE_fault_mode",
            "PE_signal_compression",
            "PE_dense_prefix_training",
        }
        groups = run_experiments._experiment_groups(self.config)
        self.assertTrue(group_names.issubset(groups))
        for group_name in group_names:
            group = groups[group_name]
            for name in group["experiments"]:
                experiment = run_experiments._experiment(self.config, name)
                self.assertEqual(experiment["phase_2_scope"], "selection_only")
                self.assertEqual(
                    experiment["architectures"],
                    ["extra_trees", "xgboost"],
                )
                self.assertEqual(experiment["candidate_budget"], 25)
                self.assertFalse(experiment["phase_3_enabled"])

    def test_pe_run_3_groups_use_one_parent_artifact_directory(self) -> None:
        groups = run_experiments._experiment_groups(self.config)
        expected = {
            "PE3_feature_union",
            "PE3_cap_sensitivity",
            "PE3_ensemble_calibration",
            "PE3_severity_loss",
        }
        self.assertTrue(expected.issubset(groups))
        for group_name in expected:
            self.assertEqual(groups[group_name]["pipeline_experiment"], "PE_3")
            self.assertEqual(groups[group_name]["pipeline_run"], "run_1")
            for name in groups[group_name]["experiments"]:
                experiment = run_experiments._experiment(self.config, name)
                self.assertEqual(experiment["pipeline_experiment"], "PE_3")
                self.assertEqual(experiment["pipeline_run"], "run_1")
                self.assertEqual(
                    run_experiments._run_dir(name, experiment),
                    PIPELINE_EXPERIMENTS_ROOT
                    / "experiments"
                    / "PE_3"
                    / "runs"
                    / "run_1"
                    / name,
                )

    def test_pe_run_3_is_declared_as_an_automatic_workflow(self) -> None:
        workflow = run_experiments._experiment_workflows(self.config)["PE_3"]
        self.assertEqual(workflow["feature_group"], "PE3_feature_union")
        self.assertEqual(workflow["cap_group"], "PE3_cap_sensitivity")
        self.assertEqual(workflow["ensemble_group"], "PE3_ensemble_calibration")
        self.assertEqual(workflow["safety_group"], "PE3_severity_loss")
        self.assertEqual(workflow["safety_r2_tolerance"], 0.005)
        self.assertEqual(workflow["promotion"], "PE3_final_ensemble")
        promotion = run_experiments._promotions(self.config)["PE3_final_ensemble"]
        self.assertEqual(promotion["workflow"], "PE_3")
        self.assertEqual(promotion["ensemble_group"], "PE3_ensemble_calibration")

    def test_pe_run_4_declares_all_conditional_calibration_quantiles(self) -> None:
        workflow = run_experiments._conditional_calibration_workflows(self.config)[
            "PE_4"
        ]
        self.assertEqual(workflow["source_phase_3_run"], 5)
        self.assertEqual(workflow["quantiles"], [0.50, 0.55, 0.60, 0.65, 0.70])
        self.assertEqual(workflow["r2_tolerance"], 0.005)
        self.assertEqual(
            workflow["prediction_bin_edges"],
            [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0],
        )

    def test_conditional_calibration_is_cross_fitted_and_subtraction_only(self) -> None:
        rows = []
        for fold in range(3):
            for index, prediction in enumerate(range(10, 130, 10)):
                rows.append(
                    {
                        "outer_fold": fold,
                        "predicted_rul": float(prediction),
                        "observed_rul": float(prediction - 2 - fold),
                    }
                )
        table = pd.DataFrame.from_records(rows)
        original = table["predicted_rul"].to_numpy(dtype=float)
        calibrated, curves = conditional_safety_calibration.cross_fit_policy(
            table,
            quantile=0.70,
            edges=pd.Series([0, 25, 50, 75, 100, 125, 150]).to_numpy(dtype=float),
            minimum_rows=2,
        )
        self.assertTrue((calibrated <= original).all())
        self.assertTrue((calibrated >= 0.0).all())
        self.assertEqual({record["outer_fold"] for record in curves}, {0, 1, 2})
        self.assertTrue(all(record["correction"] >= 0.0 for record in curves))

    def test_pe_run_3_severity_cells_vary_only_prediction_profile(self) -> None:
        group = run_experiments._experiment_group(self.config, "PE3_severity_loss")
        profiles = []
        for name in group["experiments"]:
            experiment = run_experiments._experiment(self.config, name)
            self.assertEqual(experiment["architectures"], ["xgboost"])
            self.assertEqual(experiment["feature_set"], "screened_signal_union")
            self.assertEqual(experiment["target_profile"], "capped_125")
            profiles.append(experiment["prediction_profile"])
        self.assertEqual(
            profiles,
            ["symmetric", "severity_1_5", "severity_2_0", "severity_3_0"],
        )

    def test_target_and_adapter_strategy_cells_are_explicit(self) -> None:
        self.assertEqual(
            run_experiments._experiment(self.config, "PE_failure_cycle")[
                "target_profile"
            ],
            "failure_cycle",
        )
        self.assertEqual(
            run_experiments._experiment(self.config, "PE_fault_mode_experts")[
                "fault_mode_strategy"
            ],
            "experts",
        )
        self.assertEqual(
            run_experiments._experiment(self.config, "PE_compression_pca")[
                "signal_compression_strategy"
            ],
            "pca_only",
        )

    def test_feature_catalog_contract_uses_experiment_feature_sets(self) -> None:
        phase1 = {
            "artifacts": {
                "feature_catalog": {
                    "required_columns": ["age_only", "screened_drift_pruned"]
                }
            }
        }
        expected_sets = {
            "normalization_raw": 310,
            "normalization_robust": 310,
            "normalization_combined": 618,
        }
        run_experiments._update_feature_catalog_contract(phase1, expected_sets)
        self.assertEqual(
            phase1["artifacts"]["feature_catalog"]["required_columns"],
            [
                "feature_name",
                "channel",
                "channel_role",
                "statistic",
                "window",
                *expected_sets,
            ],
        )

    def test_ready_experiment_uses_distinct_run_identities(self) -> None:
        experiment = run_experiments._experiment(self.config, "PE_1")
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
            "PE_1": ("current", "raw"),
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

        for name in cells.keys() - {"PE_1"}:
            experiment = run_experiments._experiment(self.config, name)
            self.assertEqual(experiment["architectures"], ["extra_trees", "xgboost"])
        for name in ("PE_2x2_current_cap125", "PE_2x2_early_raw"):
            experiment = run_experiments._experiment(self.config, name)
            self.assertEqual(experiment["phase_2_scope"], "selection_only")
        self.assertEqual(
            run_experiments._experiment(
                self.config,
                "PE_2x2_early_cap125",
            )["phase_2_scope"],
            "complete",
        )
        self.assertTrue(
            run_experiments._experiment(
                self.config,
                "PE_2x2_early_cap125",
            )["phase_3_enabled"]
        )
        self.assertEqual(
            run_experiments._experiment(
                self.config,
                "PE_2x2_early_cap125",
            )["phase_3_run_number"],
            4,
        )
        self.assertEqual(
            run_experiments._experiment(self.config, "PE_1")["phase_2_scope"],
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
            PIPELINE_EXPERIMENTS_ROOT / "_internal" / "phase_2_base_settings.toml"
        )
        with settings_path.open("rb") as settings_file:
            settings = tomllib.load(settings_file)
        search = settings["architectures"]["catboost"]["search"]
        self.assertEqual(search["maximum_trees"]["value"], 1000)
        self.assertEqual(search["depth"]["values"], [4, 6, 8])

    def test_hist_gradient_boosting_search_is_available_but_disabled(self) -> None:
        settings_path = (
            PIPELINE_EXPERIMENTS_ROOT / "_internal" / "phase_2_base_settings.toml"
        )
        with settings_path.open("rb") as settings_file:
            settings = tomllib.load(settings_file)
        self.assertFalse(settings["study"]["enabled"]["hist_gradient_boosting"])
        search = settings["architectures"]["hist_gradient_boosting"]["search"]
        self.assertEqual(
            set(search),
            {
                "max_iter",
                "learning_rate",
                "max_leaf_nodes",
                "max_depth",
                "min_samples_leaf",
                "l2_regularization",
            },
        )

    def test_pipeline_phase2_settings_are_independent(self) -> None:
        paths = run_experiments._paths(self.config)
        pipeline_settings = paths["phase_2_settings"]
        standalone_settings = (
            MODEL_ARCHITECTURE_ROOT
            / "1_architecture_study_settings"
            / "architecture_study_settings.toml"
        )
        self.assertEqual(
            pipeline_settings,
            PIPELINE_EXPERIMENTS_ROOT
            / "_internal"
            / "phase_2_base_settings.toml",
        )
        self.assertNotEqual(pipeline_settings, standalone_settings)

    def test_pipeline_phase2_artifacts_are_experiment_owned(self) -> None:
        experiment = run_experiments._experiment(self.config, "PE_1")
        paths = run_experiments._phase2_paths("PE_1", experiment)
        expected = (
            PIPELINE_EXPERIMENTS_ROOT
            / "experiments"
            / "PE_1"
            / "runs"
            / "run_1"
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

    def test_figures_are_collected_into_a_flat_run_gallery(self) -> None:
        test_root = REPOSITORY_ROOT / "tmp" / "pipeline_figure_collection_test"
        test_root.mkdir(parents=True, exist_ok=True)
        try:
            runs_dir = test_root / "runs"
            run_dir = runs_dir / "PE_test"
            phase2_figure = (
                run_dir
                / "phase2"
                / "7_architecture_comparison"
                / "figures"
                / "performance.png"
            )
            report_figure = run_dir / "reporting" / "paired_comparison.png"
            phase2_figure.parent.mkdir(parents=True)
            report_figure.parent.mkdir(parents=True)
            phase2_figure.write_bytes(b"phase2")
            report_figure.write_bytes(b"report")

            with patch.object(run_experiments, "RUNS_DIR", runs_dir):
                manifest = run_experiments._collect_figures("PE_test")

            gallery = run_dir / "figures"
            self.assertEqual(len(manifest["figures"]), 2)
            self.assertEqual((gallery / "performance.png").read_bytes(), b"phase2")
            self.assertEqual(
                (gallery / "paired_comparison.png").read_bytes(),
                b"report",
            )
            self.assertTrue((gallery / "figure_manifest.json").is_file())
        finally:
            shutil.rmtree(test_root, ignore_errors=True)

    def test_subexperiment_figures_are_collected_in_parent_gallery(self) -> None:
        test_root = REPOSITORY_ROOT / "tmp" / "pipeline_parent_gallery_test"
        test_root.mkdir(parents=True, exist_ok=True)
        try:
            runs_dir = test_root / "runs"
            experiment = {"pipeline_run": "PE_parent"}
            source = (
                runs_dir
                / "PE_parent"
                / "PE_cell"
                / "reporting"
                / "comparison.png"
            )
            source.parent.mkdir(parents=True)
            source.write_bytes(b"comparison")
            phase3_root = test_root / "phase3"
            phase3_figure = (
                phase3_root
                / "runs"
                / "run_4"
                / "7_post_run_reporting"
                / "figures"
                / "submission.png"
            )
            phase3_figure.parent.mkdir(parents=True)
            phase3_figure.write_bytes(b"submission")
            related = [
                {
                    "pipeline_run": "PE_parent",
                    "phase_3_enabled": True,
                    "phase_3_run_number": 4,
                }
            ]
            with (
                patch.object(run_experiments, "RUNS_DIR", runs_dir),
                patch.object(run_experiments, "PHASE_3_ROOT", phase3_root),
            ):
                manifest = run_experiments._collect_figures(
                    "PE_cell",
                    experiment,
                    related_experiments=related,
                )
                refreshed = run_experiments._collect_figures(
                    "PE_parent",
                    related_experiments=related,
                )
            gallery = runs_dir / "PE_parent" / "figures"
            self.assertEqual(manifest["pipeline_run"], "PE_parent")
            self.assertEqual(len(refreshed["figures"]), 2)
            self.assertEqual((gallery / "comparison.png").read_bytes(), b"comparison")
            self.assertEqual((gallery / "submission.png").read_bytes(), b"submission")
            self.assertFalse((runs_dir / "PE_parent" / "PE_cell" / "figures").exists())
        finally:
            shutil.rmtree(test_root, ignore_errors=True)

    def test_ensemble_report_uses_selected_cross_fitted_predictions(self) -> None:
        test_root = REPOSITORY_ROOT / "tmp" / "pipeline_ensemble_report_test"
        test_root.mkdir(parents=True, exist_ok=True)
        try:
            source_dir = (
                test_root
                / "runs"
                / "PE_parent"
                / "PE_source"
                / "phase2"
                / "5_inner_model_selection"
            )
            source_dir.mkdir(parents=True)
            records = []
            for outer_fold in range(2):
                for inner_fold in range(2):
                    for row in range(4):
                        observed = float(20 + 5 * row + outer_fold)
                        shared = {
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "validation_row": row,
                            "uav_id": f"UAV_{outer_fold}_{inner_fold}_{row}",
                            "scenario": f"development_{inner_fold:02d}",
                            "cutoff": float(10 + row),
                            "observed_rul": observed,
                        }
                        records.append(
                            {
                                **shared,
                                "model_family": "xgboost",
                                "predicted_rul": observed + 2.0,
                            }
                        )
                        records.append(
                            {
                                **shared,
                                "model_family": "extra_trees",
                                "predicted_rul": observed - 2.0,
                            }
                        )
            pd.DataFrame.from_records(records).to_csv(
                source_dir / "selected_inner_predictions.csv.gz",
                index=False,
                compression="gzip",
            )
            config = {
                "experiments": {
                    "PE_source": {"pipeline_run": "PE_parent"},
                },
                "experiment_groups": {
                    "PE_ensemble": {
                        "control": "PE_source",
                        "experiments": ["PE_source"],
                        "blend_weights": [0.5],
                        "calibration_degree": 1,
                        "calibration_ridge_alpha": 1.0,
                    }
                },
            }
            output_dir = test_root / "runs" / "PE_parent" / "reporting"
            with patch.object(
                report_ensemble_calibration,
                "EXPERIMENTS_DIR",
                test_root / "runs",
            ):
                manifest = report_ensemble_calibration.write_report(
                    config,
                    "PE_ensemble",
                    output_dir,
                )
            summary = pd.read_csv(output_dir / "summary.csv")
            best = summary.iloc[0]
            self.assertEqual(manifest["best_method"], "blend_xgb_0.50")
            self.assertEqual(best["method"], "blend_xgb_0.50")
            self.assertAlmostEqual(float(best["mean_r2"]), 1.0)
            self.assertTrue((output_dir / "ensemble_calibration_comparison.png").is_file())
        finally:
            shutil.rmtree(test_root, ignore_errors=True)

    def test_pe_run_3_workflow_propagates_winners_and_selects_safety(self) -> None:
        test_root = REPOSITORY_ROOT / "tmp" / "pipeline_workflow_test"
        test_root.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(self.config)

        def write_group_report(
            name: str,
            config_path: Path,
            active_config: dict,
            *,
            force: bool,
            report_config_path: Path | None = None,
        ) -> None:
            del config_path, force
            self.assertIsNotNone(report_config_path)
            output = run_experiments._reporting_directory(active_config, name)
            output.mkdir(parents=True, exist_ok=True)
            if name == "PE3_feature_union":
                rows = []
                scores = {
                    "PE3_features_drift": 0.86,
                    "PE3_features_signal": 0.87,
                    "PE3_features_union": 0.89,
                }
                for experiment, score in scores.items():
                    for family in ("extra_trees", "xgboost"):
                        rows.append(
                            {
                                "experiment": experiment,
                                "model_family": family,
                                "mean_r2": score,
                                "mean_rmse": 12.0 - score,
                            }
                        )
                pd.DataFrame(rows).to_csv(output / "paired_summary.csv", index=False)
            elif name == "PE3_cap_sensitivity":
                group = run_experiments._experiment_group(active_config, name)
                self.assertTrue(
                    all(
                        run_experiments._experiment(active_config, experiment)[
                            "feature_set"
                        ]
                        == "screened_signal_union"
                        for experiment in group["experiments"]
                    )
                )
                rows = []
                scores = {
                    "PE3_cap_110": 0.88,
                    "PE3_features_union": 0.89,
                    "PE3_cap_140": 0.91,
                    "PE3_cap_150": 0.90,
                }
                for experiment, score in scores.items():
                    for family in ("extra_trees", "xgboost"):
                        rows.append(
                            {
                                "experiment": experiment,
                                "model_family": family,
                                "mean_r2": score,
                                "mean_rmse": 12.0 - score,
                            }
                        )
                pd.DataFrame(rows).to_csv(output / "paired_summary.csv", index=False)
            elif name == "PE3_ensemble_calibration":
                group = run_experiments._experiment_group(active_config, name)
                self.assertEqual(group["control"], "PE3_cap_140")
                self.assertEqual(group["experiments"], ["PE3_cap_140"])
                pd.DataFrame(
                    [
                        {
                            "method": "xgboost",
                            "mean_r2": 0.900,
                            "mean_rmse": 10.0,
                            "mean_bias": 1.0,
                            "mean_overprediction_rate": 0.60,
                            "mean_rms_overprediction": 4.0,
                        },
                        {
                            "method": "xgboost__calibrated",
                            "mean_r2": 0.897,
                            "mean_rmse": 10.2,
                            "mean_bias": -0.5,
                            "mean_overprediction_rate": 0.40,
                            "mean_rms_overprediction": 1.5,
                        },
                    ]
                ).to_csv(output / "summary.csv", index=False)
            else:
                group = run_experiments._experiment_group(active_config, name)
                for experiment_name in group["experiments"]:
                    experiment = run_experiments._experiment(
                        active_config,
                        experiment_name,
                    )
                    self.assertEqual(experiment["feature_set"], "screened_signal_union")
                    self.assertEqual(experiment["target_profile"], "capped_140")
                pd.DataFrame(
                    [
                        {
                            "experiment": "PE3_safety_symmetric",
                            "mean_r2": 0.900,
                            "mean_rmse": 10.0,
                            "mean_bias": 1.0,
                            "mean_overprediction_rate": 0.60,
                            "mean_rms_overprediction": 4.0,
                        },
                        {
                            "experiment": "PE3_safety_severity_2_0",
                            "mean_r2": 0.898,
                            "mean_rmse": 10.1,
                            "mean_bias": -0.4,
                            "mean_overprediction_rate": 0.35,
                            "mean_rms_overprediction": 1.0,
                        },
                    ]
                ).to_csv(output / "paired_summary.csv", index=False)

        def selected_prediction_summary(
            active_config: dict,
            experiment_name: str,
            *,
            model_family: str,
        ) -> dict:
            del active_config
            self.assertEqual(model_family, "xgboost")
            values = {
                "PE3_cap_140": (0.900, 10.0, 1.0, 0.60, 4.0),
                "PE3_safety_severity_1_5": (0.899, 10.1, 0.0, 0.45, 2.0),
                "PE3_safety_severity_2_0": (0.898, 10.1, -0.4, 0.35, 1.0),
                "PE3_safety_severity_3_0": (0.890, 10.5, -1.0, 0.25, 0.5),
            }
            r2, rmse, bias, rate, rms = values[experiment_name]
            return {
                "experiment": experiment_name,
                "model_family": "xgboost",
                "outer_folds": 5,
                "mean_r2": r2,
                "mean_rmse": rmse,
                "mean_bias": bias,
                "mean_overprediction_rate": rate,
                "mean_rms_overprediction": rms,
                "predictions": f"{experiment_name}.csv.gz",
            }

        try:
            with (
                patch.object(run_experiments, "RUNS_DIR", test_root / "runs"),
                patch.object(
                    run_experiments,
                    "run_experiment_group",
                    side_effect=write_group_report,
                ),
                patch.object(
                    run_experiments,
                    "_selected_prediction_summary",
                    side_effect=selected_prediction_summary,
                ),
                patch.object(run_experiments, "_collect_existing_figures"),
                patch.object(run_experiments, "run_promotion") as promotion,
            ):
                manifest = run_experiments.run_experiment_workflow(
                    "PE_3",
                    PIPELINE_EXPERIMENTS_ROOT / "pipeline_experiments.toml",
                    config,
                    force=False,
                )
            selections = manifest["selections"]
            self.assertEqual(selections["feature"]["experiment"], "PE3_features_union")
            self.assertEqual(selections["target_cap"]["experiment"], "PE3_cap_140")
            self.assertEqual(
                selections["final"]["candidate"],
                "loss:PE3_safety_severity_2_0",
            )
            promotion.assert_called_once()
            promotion_args, promotion_kwargs = promotion.call_args
            self.assertEqual(promotion_args[0], "PE3_final_ensemble")
            self.assertEqual(
                promotion_args[1]["experiments"]["PE3_cap_140"]["feature_set"],
                "screened_signal_union",
            )
            self.assertEqual(promotion_kwargs, {"force": False})
            self.assertTrue(
                (
                    test_root
                    / "runs"
                    / "PE_3"
                    / "runs"
                    / "run_1"
                    / "workflow"
                    / "selection_manifest.json"
                ).is_file()
            )
        finally:
            shutil.rmtree(test_root, ignore_errors=True)

    def test_promoted_ensemble_contract_is_frozen_from_workflow_selection(self) -> None:
        test_root = REPOSITORY_ROOT / "tmp" / "pipeline_promotion_contract_test"
        workflow_dir = test_root / "runs" / "PE_3" / "workflow"
        workflow_dir.mkdir(parents=True, exist_ok=True)
        selection = {
            "status": "complete",
            "uses_locked_evaluation": False,
            "selections": {
                "final": {"candidate": "ensemble:blend_xgb_0.50__calibrated"},
                "ensemble_accuracy": {
                    "method": "blend_xgb_0.50__calibrated",
                    "source_experiment": "PE_source",
                    "mean_r2": 0.88,
                    "mean_rmse": 11.5,
                },
                "feature": {"feature_set": "screened_drift_pruned"},
                "target_cap": {"target_profile": "capped_125"},
            },
        }
        resolved = {
            "experiments": {
                "PE_source": {
                    "pipeline_run": "PE_3",
                    "feature_set": "screened_drift_pruned",
                    "target_profile": "capped_125",
                    "phase_2_scope": "selection_only",
                }
            },
            "experiment_groups": {
                "PE3_ensemble_calibration": {
                    "calibration_degree": 2,
                    "calibration_ridge_alpha": 10.0,
                }
            },
        }
        (workflow_dir / "selection_manifest.json").write_text(
            json.dumps(selection),
            encoding="utf-8",
        )
        (workflow_dir / "resolved_catalog.json").write_text(
            json.dumps(resolved),
            encoding="utf-8",
        )
        config = {
            "experiment_workflows": {
                "PE_3": {"pipeline_run": "PE_3"},
            },
            "promotions": {
                "PE3_final_ensemble": {
                    "workflow": "PE_3",
                    "ensemble_group": "PE3_ensemble_calibration",
                }
            }
        }
        try:
            with patch.object(
                promote_calibrated_ensemble,
                "EXPERIMENTS_DIR",
                test_root / "runs",
            ):
                contract, source, _ = promote_calibrated_ensemble._selected_contract(
                    config,
                    "PE3_final_ensemble",
                )
            self.assertEqual(
                contract["selected_candidate"],
                "ensemble:blend_xgb_0.50__calibrated",
            )
            self.assertEqual(
                contract["component_families"],
                ["extra_trees", "xgboost"],
            )
            self.assertEqual(contract["xgboost_weight"], 0.5)
            self.assertEqual(contract["feature_set"], "screened_drift_pruned")
            self.assertEqual(contract["target_profile"], "capped_125")
            self.assertFalse(contract["locked_results_used_for_selection"])
            self.assertEqual(source["phase_2_scope"], "selection_only")
        finally:
            shutil.rmtree(test_root, ignore_errors=True)

    def test_promoted_calibrator_combines_aligned_locked_components(self) -> None:
        test_root = REPOSITORY_ROOT / "tmp" / "pipeline_promotion_combine_test"
        test_root.mkdir(parents=True, exist_ok=True)
        try:
            development = pd.DataFrame(
                {
                    "raw_blend": [18.0, 29.0, 41.0, 52.0, 64.0, 73.0],
                    "cutoff": [8.0, 12.0, 16.0, 20.0, 24.0, 28.0],
                    "observed_rul": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
                }
            )
            contract = {
                "calibration": {"degree": 1, "ridge_alpha": 10.0},
            }
            model_path = test_root / "calibrator.joblib"
            promote_calibrated_ensemble._fit_calibrator(
                development,
                contract,
                model_path,
            )
            records = []
            for row, target in enumerate((25.0, 45.0, 65.0)):
                common = {
                    "seed": 13,
                    "outer_fold": 0,
                    "scenario": f"locked_{row + 1:02d}",
                    "sample_id": f"sample_{row}",
                    "uav_id": f"UAV_{row}",
                    "cutoff": float(10 + row * 5),
                    "terminal_lifetime": 100.0,
                    "lifetime_quantile": 0.5,
                    "y_true": target,
                }
                records.append(
                    {**common, "model_family": "xgboost", "y_pred": target + 4.0}
                )
                records.append(
                    {**common, "model_family": "extra_trees", "y_pred": target - 2.0}
                )
            component_path = test_root / "components.csv.gz"
            pd.DataFrame.from_records(records).to_csv(
                component_path,
                index=False,
                compression="gzip",
            )
            combined = promote_calibrated_ensemble._combine_locked_predictions(
                component_path,
                model_path,
                0.5,
            )
            self.assertEqual(len(combined), 3)
            self.assertTrue(combined["y_pred"].notna().all())
            self.assertTrue((combined["y_pred"] >= 0.0).all())
            self.assertIn("calibration_correction", combined)
        finally:
            shutil.rmtree(test_root, ignore_errors=True)

    def test_pipeline_phase3_settings_reference_experiment_phase2_root(self) -> None:
        experiment = copy.deepcopy(
            run_experiments._experiment(self.config, "PE_1")
        )
        experiment["phase_3_selected_model_family"] = "xgboost"
        with patch.object(run_experiments, "_write_json"):
            _, payload = run_experiments._phase3_settings(
                "PE_1",
                self.config,
                experiment,
            )
        self.assertEqual(
            payload["phase_2_run_root"],
            "2_architecture_experiments/1_pipeline_experiments/"
            "experiments/PE_1/runs/run_1/phase2",
        )

    def test_pipeline_rejects_the_standalone_phase2_toml(self) -> None:
        config = copy.deepcopy(self.config)
        config["paths"]["phase_2_settings"] = (
            "2_architecture_experiments/2_model_architecture_study/"
            "1_architecture_study_settings/"
            "architecture_study_settings.toml"
        )
        experiment = run_experiments._experiment(config, "PE_1")
        interface_path, interface = run_experiments._load_interface(experiment)
        with self.assertRaisesRegex(
            run_experiments.ExperimentManagerError,
            "must point inside 1_pipeline_experiments",
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
        experiment = run_experiments._experiment(self.config, "PE_1")
        self.assertEqual(experiment["from_stage"], "phase2")
        self.assertEqual(experiment["through_stage"], "phase2")
        self.assertEqual(
            run_experiments._resolve_stage(experiment.get("from_stage"), key="from_stage", default="phase1"),
            "phase2",
        )


if __name__ == "__main__":
    unittest.main()
