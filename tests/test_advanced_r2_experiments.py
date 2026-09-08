"""Leakage and configuration tests for PE_14 through PE_24."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPOSITORY_ROOT / "2_architecture_experiments" / "1_pipeline_experiments"
PHASE2_ROOT = REPOSITORY_ROOT / "2_architecture_experiments" / "2_model_architecture_study"
for directory in (
    PIPELINE_ROOT,
    REPOSITORY_ROOT / "1_dataset_construction" / "5_prefix_feature_engineering",
    PHASE2_ROOT / "2_tabular_data_adapter",
    PHASE2_ROOT / "3_sequence_data_adapter",
    PHASE2_ROOT / "4_model_adapters",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from advanced_r2_utils import (  # noqa: E402
    causal_filter_features,
    cross_fit_residual,
    method_report,
    prediction_history_features,
    subset_dataset,
)
from confirmation_utils import (  # noqa: E402
    evaluation_jobs,
    fixed_partitions,
    generated_partitions,
    validate_partitions,
)
from experiment_config import read_experiment_config  # noqa: E402
from history_corrected_tree_system import (  # noqa: E402
    build_lagged_dataset,
    fixed_lag_history_features,
)
from model_registry import ModelAdapterFactory  # noqa: E402
from models.tabular.residual_corrected_tree_ensemble import (  # noqa: E402
    ResidualCorrectedTreeEnsembleAdapter,
    ScaledRidgeResidualRegressor,
    calibration_sample_weights,
)
from no_op_training_monitor import NoOpTrainingMonitor  # noqa: E402
from policies import PredictionPolicy  # noqa: E402
from run_marginal_ensemble import cross_fit_blend  # noqa: E402
from run_population_degradation import degradation_features  # noqa: E402
from run_regime_tabpfn_confirmation import cross_fit_regime_blend  # noqa: E402
from run_residual_refinement import distinct_calibration_manifest  # noqa: E402
from run_tabular_prior import cross_fit_outer_blend  # noqa: E402
from tabular_data_adapter import TabularDataAdapter, TabularDataset  # noqa: E402


class AdvancedR2ExperimentTests(unittest.TestCase):
    def test_catalog_registers_all_six_workflows(self) -> None:
        config = read_experiment_config(PIPELINE_ROOT / "pipeline_experiments.toml")
        self.assertTrue({f"PE_{number}" for number in range(14, 20)}.issubset(
            config["run_definitions"]
        ))
        expected = {
            "validation_audit_workflows": "PE_14",
            "causal_filtering_workflows": "PE_15",
            "residual_refinement_workflows": "PE_16",
            "population_degradation_workflows": "PE_17",
            "tabular_prior_workflows": "PE_18",
            "marginal_ensemble_workflows": "PE_19",
        }
        for table, workflow in expected.items():
            self.assertIn(workflow, config[table])
            self.assertEqual(config[table][workflow]["pipeline_experiment"], workflow)
            self.assertEqual(config[table][workflow]["pipeline_run"], "run_1")

    def test_catalog_registers_the_confirmation_and_final_workflows(self) -> None:
        config = read_experiment_config(PIPELINE_ROOT / "pipeline_experiments.toml")
        expected = {
            "history_confirmation_workflows": "PE_20",
            "tabular_confirmation_workflows": "PE_21",
            "combined_confirmation_workflows": "PE_22",
            "final_candidate_workflows": "PE_23",
            "regime_tabpfn_workflows": "PE_24",
        }
        for table, workflow in expected.items():
            self.assertIn(workflow, config[table])
            self.assertEqual(config[table][workflow]["pipeline_experiment"], workflow)
            self.assertEqual(config[table][workflow]["pipeline_run"], "run_1")

    def test_pe16_declares_bounded_eight_cell_ablation(self) -> None:
        config = read_experiment_config(
            PIPELINE_ROOT / "experiments" / "PE_16" / "settings.toml"
        )
        candidates = config["residual_refinement_workflows"]["PE_16"]["candidates"]
        self.assertEqual(len(candidates), 8)
        self.assertIn(0.0, {float(row["correction_strength"]) for row in candidates})
        self.assertIn(0.5, {float(row["correction_strength"]) for row in candidates})
        self.assertEqual(
            {str(row["residual_family"]) for row in candidates},
            {"ridge", "hist_gradient_boosting"},
        )
        wide = [
            row for row in candidates
            if str(row["calibration_features_path"]) == "generated_distinct_20"
        ]
        self.assertEqual(len(wide), 2)
        self.assertEqual(
            sum(
                str(row["calibration_weighting"])
                == "equal_uav_unique_endpoints"
                for row in candidates
            ),
            1,
        )

    def test_pe16_generates_twenty_distinct_nominal_endpoints_per_uav(self) -> None:
        raw = pd.DataFrame(
            {
                "uav_id": ["A"] * 130 + ["B"] * 130,
                "flight_cycle": list(range(1, 131)) * 2,
                "RUL": list(range(130, 0, -1)) * 2,
            }
        )
        manifest = distinct_calibration_manifest(
            raw,
            endpoint_count=20,
            minimum_rul=1.0,
            maximum_rul=125.0,
        )
        self.assertTrue(manifest["RUL"].between(1.0, 125.0).all())
        self.assertTrue(manifest.groupby("uav_id")["cutoff"].nunique().eq(20).all())
        self.assertFalse(manifest["sample_id"].duplicated().any())

    def test_pe16_unique_endpoint_weighting_removes_scenario_duplicates(self) -> None:
        metadata = pd.DataFrame(
            {
                "uav_id": ["A", "A", "A", "B", "B"],
                "cutoff": [10, 10, 20, 30, 40],
            }
        )
        weights = calibration_sample_weights(
            metadata,
            "equal_uav_unique_endpoints",
        )
        totals = pd.DataFrame(
            {
                "uav_id": metadata["uav_id"],
                "cutoff": metadata["cutoff"],
                "weight": weights,
            }
        )
        uav_totals = totals.groupby("uav_id")["weight"].sum()
        endpoint_totals = totals.groupby(["uav_id", "cutoff"])["weight"].sum()
        self.assertAlmostEqual(float(uav_totals["A"]), float(uav_totals["B"]))
        self.assertAlmostEqual(
            float(endpoint_totals[("A", 10)]),
            float(endpoint_totals[("A", 20)]),
        )

    def test_each_new_experiment_has_one_launcher_and_artifact_root(self) -> None:
        for number in range(14, 25):
            root = PIPELINE_ROOT / "experiments" / f"PE_{number}"
            self.assertTrue((root / "run.py").is_file())
            self.assertTrue((root / "settings.toml").is_file())
            self.assertTrue((root / "runs" / "run_1").is_dir())
            config = read_experiment_config(root / "settings.toml")
            definition = config["run_definitions"][f"PE_{number}"]
            self.assertEqual(definition["pipeline_experiment"], f"PE_{number}")
            self.assertEqual(len(definition["steps"]), 1)

    def test_pe20_uses_complete_fixed_outer_and_inner_partitions(self) -> None:
        workflow = read_experiment_config(
            PIPELINE_ROOT / "experiments" / "PE_20" / "settings.toml"
        )["history_confirmation_workflows"]["PE_20"]
        outer, inner = fixed_partitions(
            outer_path=workflow["outer_folds"],
            inner_path=workflow["inner_folds"],
            split_seed=int(workflow["split_seed"]),
        )
        validate_partitions(
            outer,
            inner,
            expected_outer_folds=5,
            expected_inner_folds=4,
        )
        jobs = evaluation_jobs(outer, inner, include_inner=True)
        self.assertEqual(len(jobs), 25)
        self.assertEqual(sum(job.evaluation_level == "outer" for job in jobs), 5)
        self.assertTrue(
            all(not (job.training_uavs & job.validation_uavs) for job in jobs)
        )

    def test_pe21_freezes_two_new_seeds_and_the_original_weight_grid(self) -> None:
        workflow = read_experiment_config(
            PIPELINE_ROOT / "experiments" / "PE_21" / "settings.toml"
        )["tabular_confirmation_workflows"]["PE_21"]
        self.assertEqual(workflow["split_seeds"], [20260917, 20260927])
        self.assertEqual(workflow["challenger_weights"], [0.0, 0.05, 0.1, 0.15, 0.25])
        self.assertEqual(workflow["minimum_fold_wins"], 8)
        self.assertEqual(workflow["feature_set"], "screened_drift_pruned")

    def test_pe24_freezes_fresh_seeds_and_one_strict_primary_gate(self) -> None:
        workflow = read_experiment_config(
            PIPELINE_ROOT / "experiments" / "PE_24" / "settings.toml"
        )["regime_tabpfn_workflows"]["PE_24"]
        self.assertEqual(
            workflow["split_seeds"],
            [20261007, 20261017, 20261027],
        )
        self.assertTrue(
            set(workflow["split_seeds"]).isdisjoint({20260814, 20260917, 20260927})
        )
        self.assertEqual(workflow["gate_maximum_depth"], 2)
        self.assertEqual(workflow["gate_minimum_samples_leaf"], 40)
        self.assertEqual(workflow["gate_maximum_tabpfn_weight"], 0.5)
        self.assertEqual(
            workflow["gate_features"],
            [
                "control_prediction",
                "uncertainty_std",
                "uncertainty_range",
                "family_disagreement",
                "challenger_gap",
            ],
        )
        self.assertEqual(workflow["minimum_fold_wins"], 12)
        self.assertEqual(workflow["minimum_pooled_r2"], 0.9)
        self.assertTrue(workflow["require_bootstrap_improvement"])
        outer, inner = generated_partitions(
            history_summary_path=workflow["history_summary"],
            split_seeds=workflow["split_seeds"],
            outer_fold_count=workflow["outer_fold_count"],
            inner_fold_count=workflow["inner_fold_count"],
        )
        validate_partitions(
            outer,
            inner,
            expected_outer_folds=5,
            expected_inner_folds=4,
        )
        jobs = evaluation_jobs(outer, inner, include_inner=True)
        self.assertEqual(len(jobs), 75)
        self.assertEqual(sum(job.evaluation_level == "outer" for job in jobs), 15)
        self.assertTrue(
            all(not (job.training_uavs & job.validation_uavs) for job in jobs)
        )

    def test_fixed_lag_history_is_strictly_past_and_excludes_cap(self) -> None:
        metadata = pd.DataFrame(
            {
                "query_index": [0, 0, 0, 1, 1],
                "history_lag": [0, 2, 5, 0, 2],
                "cutoff": [20, 18, 15, 40, 38],
            }
        )
        predictions = np.array([45.0, 46.0, 125.0, 30.0, 31.0])
        observed = fixed_lag_history_features(
            metadata,
            predictions,
            query_count=2,
            near_cap_threshold=124.5,
        )
        self.assertEqual(observed.loc[0, "history_count"], 1.0)
        self.assertEqual(observed.loc[0, "history_near_cap_fraction"], 0.5)
        changed = predictions.copy()
        changed[3:] = -999.0
        unaffected = fixed_lag_history_features(
            metadata,
            changed,
            query_count=2,
            near_cap_threshold=124.5,
        )
        pd.testing.assert_series_equal(observed.loc[0], unaffected.loc[0])

    def test_lagged_features_ignore_raw_rows_after_the_query_cutoff(self) -> None:
        pe20 = read_experiment_config(
            PIPELINE_ROOT / "experiments" / "PE_20" / "settings.toml"
        )["history_confirmation_workflows"]["PE_20"]
        adapter = TabularDataAdapter(REPOSITORY_ROOT / pe20["tabular_manifest"])
        development = adapter.load_development(str(pe20["feature_set"]))
        query = subset_dataset(development, np.arange(len(development)) == 0)
        raw = pd.read_csv(REPOSITORY_ROOT / pe20["train_csv"])
        expected = build_lagged_dataset(
            raw,
            query,
            feature_names=list(query.features.columns),
            lags=pe20["history_lags"],
            feature_profile=str(pe20["feature_profile"]),
        )
        changed = raw.copy()
        uav = str(query.metadata.iloc[0].uav_id)
        cutoff = float(query.metadata.iloc[0].cutoff)
        future = changed.uav_id.astype(str).eq(uav) & changed.flight_cycle.gt(cutoff)
        telemetry = [column for column in changed if column.startswith("telemetry_")]
        changed.loc[future, telemetry] = 1e12
        observed = build_lagged_dataset(
            changed,
            query,
            feature_names=list(query.features.columns),
            lags=pe20["history_lags"],
            feature_profile=str(pe20["feature_profile"]),
        )
        pd.testing.assert_frame_equal(expected.features, observed.features)

    def test_pe18_pins_local_tabpfn_and_separate_blend_gate(self) -> None:
        workflow = read_experiment_config(
            PIPELINE_ROOT / "experiments" / "PE_18" / "settings.toml"
        )["tabular_prior_workflows"]["PE_18"]
        self.assertEqual(workflow["tabpfn"]["version"], "8.5.0")
        self.assertTrue(workflow["tabpfn"]["required"])
        self.assertTrue(str(workflow["tabpfn"]["checkpoint"]).endswith(".ckpt"))
        self.assertEqual(len(workflow["tabpfn"]["sha256"]), 64)
        self.assertLess(
            workflow["blend_minimum_relative_rmse_improvement"],
            workflow["minimum_relative_rmse_improvement"],
        )
        self.assertIn(0.0, workflow["challenger_weights"])

    def test_pe14_and_pe19_declare_nested_selection_sources(self) -> None:
        pe14 = read_experiment_config(
            PIPELINE_ROOT / "experiments" / "PE_14" / "settings.toml"
        )["validation_audit_workflows"]["PE_14"]
        self.assertEqual(len(pe14["split_seeds"]), 3)
        self.assertEqual(pe14["outer_fold_count"], 5)
        self.assertGreaterEqual(pe14["minimum_fold_wins"], 12)
        pe19 = read_experiment_config(
            PIPELINE_ROOT / "experiments" / "PE_19" / "settings.toml"
        )["marginal_ensemble_workflows"]["PE_19"]
        self.assertIn("tree_inner_predictions", pe19)
        self.assertIn("temporal_inner_predictions", pe19)
        self.assertIn(0.0, pe19["challenger_weights"])

    def test_causal_filter_ignores_every_future_observation(self) -> None:
        raw = pd.DataFrame(
            {
                "uav_id": ["A"] * 6,
                "flight_cycle": np.arange(1, 7),
                "telemetry_13": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            }
        )
        endpoints = pd.DataFrame({"uav_id": ["A"], "cutoff": [3]})
        expected = causal_filter_features(
            raw, endpoints, channels=["telemetry_13"], half_life=3.0
        )
        changed = raw.copy()
        changed.loc[changed.flight_cycle.gt(3), "telemetry_13"] = 1e9
        observed = causal_filter_features(
            changed, endpoints, channels=["telemetry_13"], half_life=3.0
        )
        pd.testing.assert_frame_equal(expected, observed)

    def test_prediction_history_is_strictly_past_and_excludes_cap(self) -> None:
        rows = pd.DataFrame(
            {
                "outer_fold": [0, 0, 0, 0],
                "inner_fold": [0, 0, 0, 0],
                "uav_id": ["A"] * 4,
                "scenario": ["s1", "s2", "s3", "s4"],
                "cutoff": [10.0, 15.0, 20.0, 30.0],
                "predicted_rul": [125.0, 50.0, 45.0, 999.0],
            }
        )
        expected = prediction_history_features(rows)
        self.assertEqual(expected.loc[2, "history_count"], 1.0)
        self.assertEqual(expected.loc[2, "history_near_cap_fraction"], 0.5)
        changed = rows.copy()
        changed.loc[3, "predicted_rul"] = -999.0
        observed = prediction_history_features(changed)
        pd.testing.assert_series_equal(expected.loc[2], observed.loc[2])

    def test_residual_cross_fit_keeps_uavs_disjoint(self) -> None:
        rows = pd.DataFrame(
            {
                "outer_fold": [0, 0, 0, 0],
                "inner_fold": [0, 0, 1, 1],
                "uav_id": ["A", "A", "B", "B"],
                "observed_rul": [8.0, 9.0, 18.0, 19.0],
                "base": [10.0, 11.0, 20.0, 21.0],
                "feature": [1.0, 2.0, 3.0, 4.0],
            }
        )
        prediction, provenance = cross_fit_residual(
            rows,
            base_column="base",
            feature_columns=["base", "feature"],
            model_family="ridge",
            strength=0.5,
        )
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(len(provenance), 2)
        self.assertTrue(all(row["uav_overlap"] == 0 for row in provenance))

    def test_population_features_use_query_prefix_only(self) -> None:
        histories = {
            "Q": pd.DataFrame(
                {
                    "flight_cycle": np.arange(1, 7),
                    "telemetry_13": [0.0, 1.0, 2.0, 3.0, 20.0, 30.0],
                }
            )
        }
        rows = pd.DataFrame({"uav_id": ["Q"], "cutoff": [4]})
        reference = {"telemetry_13": (0.0, 10.0, 1.0)}
        expected = degradation_features(rows, histories, reference, recent_window=4)
        changed = {"Q": histories["Q"].copy()}
        changed["Q"].loc[changed["Q"].flight_cycle.gt(4), "telemetry_13"] = -1e9
        observed = degradation_features(rows, changed, reference, recent_window=4)
        pd.testing.assert_frame_equal(expected, observed)

    def test_outer_blend_allows_zero_to_win(self) -> None:
        rows = pd.DataFrame(
            {
                "outer_fold": list(range(5)),
                "uav_id": [f"U{index}" for index in range(5)],
                "observed_rul": [10.0, 20.0, 30.0, 40.0, 50.0],
                "tree_prediction": [10.0, 20.0, 30.0, 40.0, 50.0],
                "challenger_prediction": [30.0, 40.0, 50.0, 60.0, 70.0],
            }
        )
        selection_rows = rows.copy()
        selection_rows["outer_fold"] = 0
        selection_rows["uav_id"] = [f"T{index}" for index in range(5)]
        rows["outer_fold"] = 0
        prediction, provenance = cross_fit_blend(
            rows, [0.0, 0.05, 0.10], selection_rows
        )
        np.testing.assert_allclose(prediction, rows.tree_prediction)
        self.assertTrue(
            all(row["selected_challenger_weight"] == 0.0 for row in provenance)
        )

    def test_pe18_blend_selection_uses_only_outer_training_uavs(self) -> None:
        held = pd.DataFrame(
            {
                "outer_fold": [0, 1],
                "uav_id": ["H0", "H1"],
                "observed_rul": [10.0, 20.0],
                "control_prediction": [12.0, 22.0],
                "challenger_prediction": [10.0, 20.0],
            }
        )
        selection = pd.DataFrame(
            {
                "outer_fold": [0, 0, 1, 1],
                "uav_id": ["T00", "T01", "T10", "T11"],
                "observed_rul": [10.0, 20.0, 30.0, 40.0],
                "control_prediction": [12.0, 22.0, 32.0, 42.0],
                "challenger_prediction": [10.0, 20.0, 30.0, 40.0],
            }
        )
        prediction, provenance = cross_fit_outer_blend(
            held,
            selection,
            weights=[0.0, 0.5, 1.0],
        )
        np.testing.assert_allclose(prediction, held["challenger_prediction"])
        self.assertTrue(all(row["uav_overlap"] == 0 for row in provenance))
        self.assertTrue(
            all(row["selected_challenger_weight"] == 1.0 for row in provenance)
        )

    def test_pe24_regime_gate_is_training_only_and_bounded(self) -> None:
        features = [
            "control_prediction",
            "uncertainty_std",
            "uncertainty_range",
            "family_disagreement",
            "challenger_gap",
        ]
        selection = pd.DataFrame(
            {
                "outer_fold": [0] * 4,
                "uav_id": ["T0", "T1", "T2", "T3"],
                "observed_rul": [10.0, 10.0, 80.0, 80.0],
                "control_prediction": [20.0, 20.0, 80.0, 80.0],
                "challenger_prediction": [10.0, 10.0, 90.0, 90.0],
                "uncertainty_std": [2.0, 3.0, 2.0, 3.0],
                "uncertainty_range": [4.0, 6.0, 4.0, 6.0],
                "family_disagreement": [1.0, 1.0, 1.0, 1.0],
                "challenger_gap": [-10.0, -10.0, 10.0, 10.0],
            }
        )
        held = pd.DataFrame(
            {
                "outer_fold": [0, 0],
                "uav_id": ["H0", "H1"],
                "observed_rul": [-999.0, -999.0],
                "control_prediction": [20.0, 80.0],
                "challenger_prediction": [10.0, 90.0],
                "uncertainty_std": [2.5, 2.5],
                "uncertainty_range": [5.0, 5.0],
                "family_disagreement": [1.0, 1.0],
                "challenger_gap": [-10.0, 10.0],
            }
        )
        prediction, weights, provenance = cross_fit_regime_blend(
            held,
            selection,
            feature_columns=features,
            maximum_depth=1,
            minimum_samples_leaf=1,
            maximum_tabpfn_weight=0.5,
            minimum_challenger_gap=1.0,
            random_state=7,
        )
        changed = held.copy()
        changed["observed_rul"] = [1e9, 1e9]
        changed_prediction, changed_weights, _ = cross_fit_regime_blend(
            changed,
            selection,
            feature_columns=features,
            maximum_depth=1,
            minimum_samples_leaf=1,
            maximum_tabpfn_weight=0.5,
            minimum_challenger_gap=1.0,
            random_state=7,
        )
        np.testing.assert_allclose(prediction, changed_prediction)
        np.testing.assert_allclose(weights, changed_weights)
        np.testing.assert_allclose(weights, [0.5, 0.0])
        np.testing.assert_allclose(prediction, [15.0, 80.0])
        self.assertEqual(provenance[0]["uav_overlap"], 0)

    def test_method_report_can_restrict_promotion_to_primary_method(self) -> None:
        rows = pd.DataFrame(
            {
                "outer_fold": [0, 0, 1, 1],
                "uav_id": ["A", "B", "C", "D"],
                "observed_rul": [10.0, 20.0, 30.0, 40.0],
                "cutoff": [100.0, 100.0, 100.0, 100.0],
            }
        )
        methods = {
            "control": np.array([12.0, 22.0, 32.0, 42.0]),
            "diagnostic": np.array([10.0, 20.0, 30.0, 40.0]),
            "primary": np.array([12.0, 22.0, 32.0, 42.0]),
        }
        root = REPOSITORY_ROOT / ".tmp" / "method_report_gate_test"
        manifest = method_report(
            rows,
            methods,
            root=root,
            control="control",
            minimum_fold_wins=2,
            minimum_relative_rmse_improvement=0.01,
            promotion_eligible_methods={"primary"},
            require_bootstrap_improvement=True,
            minimum_pooled_r2=0.9,
        )
        decisions = pd.read_csv(root / "reporting" / "promotion_decisions.csv")
        self.assertEqual(manifest["winner"], "control")
        diagnostic = decisions.loc[decisions.method.eq("diagnostic")].iloc[0]
        self.assertFalse(bool(diagnostic.promotion_eligible))
        self.assertFalse(bool(diagnostic.passes_gate))

    def test_scaled_ridge_residual_head_preserves_feature_contract(self) -> None:
        x = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [2.0, 1.0, 0.0]})
        model = ScaledRidgeResidualRegressor(alpha=1.0).fit(
            x, np.array([0.0, 1.0, 2.0]), sample_weight=np.ones(3)
        )
        self.assertEqual(model.predict(x).shape, (3,))
        with self.assertRaisesRegex(ValueError, "feature order"):
            model.predict(x[["b", "a"]])

    def test_residual_ensemble_diagnostics_match_public_prediction(self) -> None:
        class FixedMember:
            def __init__(self, values: list[float]) -> None:
                self.values = np.asarray(values, dtype=float)

            def predict(self, data: object) -> np.ndarray:
                return self.values.copy()

        class ZeroResidual:
            @staticmethod
            def predict(features: pd.DataFrame) -> np.ndarray:
                return np.zeros(len(features), dtype=float)

        model = object.__new__(ResidualCorrectedTreeEnsembleAdapter)
        model._is_fitted = True
        model.member_seeds = (13, 37, 73)
        model.members = [
            FixedMember([10.0, 12.0]),
            FixedMember([11.0, 13.0]),
            FixedMember([12.0, 14.0]),
            FixedMember([20.0, 22.0]),
            FixedMember([21.0, 23.0]),
            FixedMember([22.0, 24.0]),
        ]
        model.xgboost_weight = 0.5
        model.prediction_minimum = 0.0
        model.correction_strength = 1.0
        model.residual_features = ()
        model.residual_model = ZeroResidual()
        model.prediction_policy = PredictionPolicy()
        data = TabularDataset(
            features=pd.DataFrame(index=range(2)),
            metadata=pd.DataFrame({"cutoff": [100.0, 120.0]}),
            target=None,
            sample_weights=None,
        )
        diagnostics = model.predict_with_diagnostics(data)
        np.testing.assert_allclose(
            model.predict(data),
            diagnostics["predicted_rul"],
        )
        np.testing.assert_allclose(diagnostics["predicted_rul"], [16.0, 18.0])
        self.assertTrue(
            {
                "base_prediction",
                "uncertainty_std",
                "uncertainty_range",
                "family_disagreement",
            }.issubset(diagnostics)
        )
        self.assertTrue((diagnostics["uncertainty_std"] > 0.0).all())

    def test_run6_factory_accepts_fold_local_calibration_contract(self) -> None:
        settings = read_experiment_config(
            PIPELINE_ROOT / "experiments" / "PE_14" / "settings.toml"
        )["validation_audit_workflows"]["PE_14"]
        system = settings["systems"]["run_6"]
        candidates = pd.read_csv(REPOSITORY_ROOT / system["selected_candidates"])
        selected = candidates.loc[candidates.selected]
        hyperparameters = json.loads(str(selected.iloc[0].hyperparameters_json))
        for key in (
            "calibration_features_path",
            "calibration_internal_folds",
            "calibration_degree",
            "calibration_ridge_alpha",
        ):
            hyperparameters[key] = system[key]
        factory = ModelAdapterFactory(REPOSITORY_ROOT / system["specification"])
        model = factory.create(
            "calibrated_tree_blend",
            hyperparameters,
            seed=13,
            allow_disabled=True,
            training_monitor=NoOpTrainingMonitor(),
        )
        self.assertIsNotNone(model.calibration_features_path)
        _, xgboost = model._new_components()
        self.assertIsInstance(xgboost.require_training_monitor(), NoOpTrainingMonitor)

    def test_run6_local_calibration_removes_held_outer_uavs(self) -> None:
        workflow = read_experiment_config(
            PIPELINE_ROOT / "experiments" / "PE_14" / "settings.toml"
        )["validation_audit_workflows"]["PE_14"]
        system = workflow["systems"]["run_6"]
        candidates = pd.read_csv(REPOSITORY_ROOT / system["selected_candidates"])
        hyperparameters = json.loads(
            str(candidates.loc[candidates.selected].iloc[0].hyperparameters_json)
        )
        for key in (
            "calibration_features_path",
            "calibration_internal_folds",
            "calibration_degree",
            "calibration_ridge_alpha",
        ):
            hyperparameters[key] = system[key]
        factory = ModelAdapterFactory(REPOSITORY_ROOT / system["specification"])
        model = factory.create(
            "calibrated_tree_blend",
            hyperparameters,
            seed=13,
            allow_disabled=True,
            training_monitor=NoOpTrainingMonitor(),
        )
        adapter = TabularDataAdapter(REPOSITORY_ROOT / workflow["tabular_manifest"])
        full_training = adapter.load_training(str(workflow["feature_set"]))
        folds = pd.read_csv(REPOSITORY_ROOT / workflow["outer_folds"])
        held_uavs = set(folds.loc[folds.outer_fold.eq(0), "uav_id"].astype(str))
        training = subset_dataset(
            full_training,
            ~full_training.metadata.uav_id.astype(str).isin(held_uavs).to_numpy(),
        )
        calibration = model._calibration_data(training)
        self.assertFalse(
            held_uavs & set(calibration.metadata.uav_id.astype(str))
        )
        self.assertEqual(calibration.metadata.uav_id.nunique(), 80)


if __name__ == "__main__":
    unittest.main()
