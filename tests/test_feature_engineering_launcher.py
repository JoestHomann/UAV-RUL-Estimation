from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_ROOT = (
    REPOSITORY_ROOT / "0_data_analysis" / "model_guided_feature_analysis"
)
LAUNCHER_ROOT = DIAGNOSTIC_ROOT
for path in (LAUNCHER_ROOT, DIAGNOSTIC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_experiments import (
    DEFAULT_SETTINGS,
    diagnostic_command,
    load_settings,
)
from run_feature_diagnostics import (
    effective_model_parameters,
    parse_parameter_overrides,
)


class FeatureEngineeringLauncherTests(unittest.TestCase):
    def test_default_registry_defines_both_extended_prefix_variants(self) -> None:
        settings = load_settings(DEFAULT_SETTINGS)
        experiment = settings.experiments["FE_run_1"]

        self.assertTrue(experiment.enabled)
        self.assertEqual(experiment.profile, "extended_features")
        self.assertEqual(experiment.phase_1_run_name, "FE_run_1")
        self.assertEqual(
            experiment.prefix_variants,
            ("current20", "prefix40_stratified"),
        )
        self.assertEqual(
            experiment.diagnostics.models,
            ("xgboost", "extra_trees"),
        )

    def test_diagnostic_command_contains_editable_run_settings(self) -> None:
        settings = load_settings(DEFAULT_SETTINGS)
        experiment = settings.experiments["FE_run_1"]
        interface = {
            "artifacts": {
                "training_features": "training.csv.gz",
                "development_features": "development.csv.gz",
                "test_features": "test.csv.gz",
                "feature_catalog": "catalog.csv",
                "outer_folds": "outer_folds.csv",
            }
        }
        command = diagnostic_command(
            settings,
            experiment,
            "current20",
            interface,
        )

        self.assertIn("--xgboost-parameters-json", command)
        self.assertIn("--extra-trees-parameters-json", command)
        self.assertIn("screened_robust", command)
        self.assertIn(str(experiment.output_root / "current20"), command)

    def test_drift_ablation_run_is_one_paired_current20_experiment(self) -> None:
        settings = load_settings(DEFAULT_SETTINGS)
        experiment = settings.experiments["FE_run_2"]

        self.assertEqual(experiment.profile, "drift_ablation_features")
        self.assertEqual(experiment.phase_1_run_name, "FE_run_2")
        self.assertEqual(experiment.prefix_variants, ("current20",))
        self.assertEqual(
            experiment.diagnostics.feature_sets,
            (
                "screened_v1",
                "screened_drift_pruned",
                "screened_drift_replaced",
            ),
        )
        self.assertEqual(
            experiment.diagnostics.models,
            ("xgboost", "extra_trees"),
        )

    def test_model_overrides_merge_and_random_state_stays_seed_owned(self) -> None:
        overrides = parse_parameter_overrides(
            '{"n_estimators": 12, "max_depth": 3}',
            "--xgboost-parameters-json",
        )
        parameters = effective_model_parameters("xgboost", overrides)
        self.assertEqual(parameters["n_estimators"], 12)
        self.assertEqual(parameters["max_depth"], 3)
        self.assertEqual(parameters["objective"], "reg:squarederror")

        with self.assertRaises(ValueError):
            parse_parameter_overrides(
                '{"random_state": 99}',
                "--xgboost-parameters-json",
            )


if __name__ == "__main__":
    unittest.main()
