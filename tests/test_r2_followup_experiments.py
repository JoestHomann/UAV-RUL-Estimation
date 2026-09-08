"""Scientific isolation, resumability and routing checks for PE_25 and PE_26."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
import shutil
import unittest
from uuid import uuid4
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "2_architecture_experiments" / "1_pipeline_experiments"
PHASE2 = ROOT / "2_architecture_experiments" / "2_model_architecture_study"
for directory in (PIPELINE, PHASE2 / "4_model_adapters", PHASE2 / "2_tabular_data_adapter"):
    sys.path.insert(0, str(directory))

from confirmation_utils import evaluation_jobs, prediction_records  # noqa: E402
from experiment_config import read_experiment_config  # noqa: E402
from followup_experiment_utils import (  # noqa: E402
    aligned_methods, complete_cell, register_run, route_predictions,
    select_routed_blend, validate_saved_nested,
)
from models.tabular.residual_corrected_tree_ensemble import ResidualCorrectedTreeEnsembleAdapter  # noqa: E402
from policies import PredictionPolicy, TargetPolicy  # noqa: E402
from run_experiment_definition import _execution_plan  # noqa: E402
from run_short_history_specialist import routed_specialist_predictions  # noqa: E402
import run_short_history_specialist as specialist_runner  # noqa: E402
from short_history_tree_system import ShortHistoryTreeEnsemble, early_prefixes  # noqa: E402
from tabular_data_adapter import TabularDataset  # noqa: E402


@contextmanager
def test_directory():
    # mkdir with inherited Windows permissions; tempfile's mode 0700 is not
    # compatible with the restricted Windows test process on this host.
    base = (ROOT / ".tmp").resolve()
    path = base / ("r2_followup_" + uuid4().hex)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        path.resolve().relative_to(base)
        shutil.rmtree(path)


def dataset() -> TabularDataset:
    return TabularDataset(
        features=pd.DataFrame({"sensor": [1., 2., 99., 3., 99., 99.]}),
        metadata=pd.DataFrame({"uav_id": ["A", "A", "A", "B", "B", "B"],
                               "cutoff": [10., 100., 150., 10., 110., 150.]}),
        target=pd.Series([160., 70., 20., 155., 55., 15.]),
        fitting_target=pd.Series([125., 70., 20., 125., 55., 15.]),
        sample_weights=pd.Series([1/3] * 6),
    )


def routed_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    training = pd.DataFrame({"outer_fold": [0]*4, "inner_fold": [0, 0, 1, 1],
        "uav_id": ["A", "A", "B", "B"], "cutoff": [50., 150., 50., 150.],
        "observed_rul": [15., 110., 15., 110.],
        "control_prediction": [20., 110., 20., 110.],
        "challenger_prediction": [10., 140., 10., 140.]})
    held = pd.DataFrame({"outer_fold": [0, 0], "inner_fold": [-1, -1],
        "uav_id": ["C", "C"], "cutoff": [50., 150.], "observed_rul": [15., 110.],
        "control_prediction": [20., 110.], "challenger_prediction": [10., 140.]})
    return held, training


def nested_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer = pd.DataFrame({"split_seed": [7]*6, "uav_id": list("ABCDEF"), "outer_fold": [0,0,1,1,2,2]})
    rows = []
    for fold in range(3):
        for index, uav in enumerate(outer.loc[outer.outer_fold.ne(fold), "uav_id"]):
            rows.append({"split_seed": 7, "outer_fold": fold, "uav_id": uav, "inner_fold": index % 2})
    inner = pd.DataFrame(rows)
    records = []
    for job in evaluation_jobs(outer, inner, include_inner=True):
        uavs = sorted(job.validation_uavs)
        d = TabularDataset(pd.DataFrame({"sensor": [1.] * len(uavs)}),
            pd.DataFrame({"uav_id": uavs, "cutoff": [50.] * len(uavs), "scenario": ["s1"] * len(uavs)}),
            pd.Series([25.] * len(uavs)), None)
        records.extend(prediction_records(job, d, {"control": np.full(len(d), 30.), "challenger": np.full(len(d), 20.)}))
    return pd.DataFrame(records), outer, inner


class FollowupExperimentsTests(unittest.TestCase):
    def test_configs_load_independently_and_catalog_lists_both(self) -> None:
        catalog = read_experiment_config(PIPELINE / "pipeline_experiments.toml")
        for name, table in (("PE_25", "restricted_tabpfn_workflows"), ("PE_26", "short_history_workflows")):
            path = PIPELINE / "experiments" / name / "settings.toml"
            config = read_experiment_config(path)
            self.assertEqual(config[table][name]["pipeline_run"], "run_1")
            self.assertIn(name, catalog[table])
            _, _, steps = _execution_plan(path, name)
            self.assertEqual(len(steps), 2)
            self.assertEqual(steps[0]["name"], "validate_inputs")

    def test_outer_labels_never_change_selected_rule_or_predictions(self) -> None:
        held, training = routed_rows()
        kwargs = dict(weights=[0., .25, .5], thresholds=[50., 100.], route_column="control_prediction")
        baseline, provenance = select_routed_blend(held, training, **kwargs)
        changed = held.copy()
        changed["observed_rul"] = [999., -1000.]
        result, changed_provenance = select_routed_blend(changed, training, **kwargs)
        np.testing.assert_array_equal(result, baseline)
        pd.testing.assert_frame_equal(provenance, changed_provenance)
        np.testing.assert_array_equal(result, [15., 110.])
        self.assertEqual(provenance.iloc[0].selected_weight, .5)

    def test_outer_uav_overlap_is_rejected(self) -> None:
        held, training = routed_rows()
        training.loc[0, "uav_id"] = "C"
        with self.assertRaisesRegex(ValueError, "outer-held"):
            select_routed_blend(held, training, weights=[0., .5], thresholds=[50.], route_column="control_prediction")

    def test_zero_weight_control_wins_when_challenger_is_worse(self) -> None:
        held, training = routed_rows()
        training["observed_rul"] = training.control_prediction
        result, provenance = select_routed_blend(held, training, weights=[.5, 0.], thresholds=[50.], route_column="cutoff")
        self.assertEqual(provenance.iloc[0].selected_weight, 0.)
        np.testing.assert_array_equal(result, held.control_prediction)

    def test_label_free_routing_and_cutoff_boundary(self) -> None:
        held, _ = routed_rows()
        held = held.drop(columns="observed_rul")
        np.testing.assert_array_equal(route_predictions(held, weight=.5, threshold=50., route_column="cutoff"), [15., 110.])
        with self.assertRaises(ValueError):
            route_predictions(held, weight=.5, threshold=50., route_column="observed_rul")

    def test_invalid_grid_is_rejected(self) -> None:
        held, training = routed_rows()
        for weights, thresholds in (([.5], [50.]), ([0., 2.], [50.]), ([0., .5], [float("nan")])):
            with self.assertRaises(ValueError):
                select_routed_blend(held, training, weights=weights, thresholds=thresholds, route_column="cutoff")

    def test_exact_alignment_rejects_missing_duplicate_and_changed_label(self) -> None:
        table, _, _ = nested_fixture()
        rows = table.loc[table.evaluation_level.eq("outer")]
        self.assertEqual(len(aligned_methods(rows, "challenger")), 6)
        for corrupted in (rows.iloc[1:], pd.concat([rows, rows.iloc[:1]])):
            with self.assertRaises(ValueError):
                aligned_methods(corrupted, "challenger")
        corrupted = rows.copy()
        corrupted.loc[corrupted.method.eq("challenger"), "observed_rul"] += 1
        with self.assertRaises(ValueError):
            aligned_methods(corrupted, "challenger")

    def test_complete_nested_source_and_corruption_checks(self) -> None:
        table, outer, inner = nested_fixture()
        kwargs = dict(methods={"control", "challenger"}, outer_count=3, inner_count=2)
        validate_saved_nested(table, outer, inner, **kwargs)
        for corrupted in (table.iloc[1:], pd.concat([table, table.iloc[:1]])):
            with self.assertRaises(ValueError):
                validate_saved_nested(corrupted, outer, inner, **kwargs)
        corrupted = table.copy()
        corrupted.loc[corrupted.evaluation_level.eq("inner"), "outer_fold"] = 99
        with self.assertRaises(ValueError):
            validate_saved_nested(corrupted, outer, inner, **kwargs)

    def test_checkpoint_checks_endpoint_identity_not_only_row_count(self) -> None:
        table, outer, inner = nested_fixture()
        job = evaluation_jobs(outer, inner, include_inner=True)[0]
        expected = table.loc[table.outer_fold.eq(job.report_fold) & table.evaluation_level.eq("outer") & table.method.eq("control")]
        self.assertTrue(complete_cell(table, job, "control", expected))
        corrupt = table.copy()
        corrupt.loc[expected.index[0], "cutoff"] = 49.
        with self.assertRaises(ValueError):
            complete_cell(corrupt, job, "control", expected)

    def test_registration_rejects_changed_settings_and_source_content(self) -> None:
        with test_directory() as directory:
            root = Path(directory)
            source = root / "input.txt"
            source.write_text("original", encoding="utf-8")
            workflow = {"maximum_history": 100}
            register_run(root / "reporting", workflow, [source])
            register_run(root / "reporting", workflow, [source])
            with self.assertRaisesRegex(ValueError, "new pipeline.run"):
                register_run(root / "reporting", {"maximum_history": 90}, [source])
            source.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "new pipeline.run"):
                register_run(root / "reporting", workflow, [source])

    def test_early_prefixes_preserve_raw_and_capped_targets_and_unit_weights(self) -> None:
        short = early_prefixes(dataset(), 100.)
        np.testing.assert_array_equal(short.metadata.cutoff, [10., 100., 10.])
        np.testing.assert_array_equal(short.target, [160., 70., 155.])
        np.testing.assert_array_equal(short.fitting_target, [125., 70., 125.])
        np.testing.assert_allclose(short.sample_weights.groupby(short.metadata.uav_id).sum(), [1., 1.])
        self.assertEqual(list(short.features.columns), ["sensor"])

    def test_later_prefix_changes_do_not_change_specialist_training_inputs(self) -> None:
        original = dataset()
        changed = dataset()
        mask = changed.metadata.cutoff.gt(100)
        changed.features.loc[mask, "sensor"] = -9999.
        changed.target.loc[mask] = 9999.
        a, b = early_prefixes(original, 100), early_prefixes(changed, 100)
        pd.testing.assert_frame_equal(a.features, b.features)
        pd.testing.assert_series_equal(a.target, b.target)

    def test_missing_early_training_uav_rejected(self) -> None:
        data = dataset()
        data.metadata.loc[data.metadata.uav_id.eq("B"), "cutoff"] = 200.
        with self.assertRaisesRegex(ValueError, "every training UAV"):
            early_prefixes(data, 100.)

    def test_specialist_calibration_is_early_and_excludes_held_uavs(self) -> None:
        # Exercise the actual inherited CSV loader with a small isolated contract.
        with test_directory() as directory:
            path = Path(directory)
            rows = pd.DataFrame({"sample_id": list("abcdef"), "scenario": ["s1"] * 6,
                "uav_id": ["A", "A", "B", "B", "HELD", "HELD"],
                "cutoff": [50., 150.] * 3, "RUL": [80., 20.] * 3, "sensor": [1.] * 6})
            calibration = path / "calibration.csv"
            rows.to_csv(calibration, index=False)
            contract = path / "contract.json"
            contract.write_text(json.dumps({"contract_version": 1,
                "calibration_features_path": str(calibration.relative_to(ROOT)),
                "member_seeds": [13, 37], "internal_folds": 2,
                "residual_features": ["sensor"], "xgboost_weight_grid": [0., 1.]}), encoding="utf-8")
            model = ShortHistoryTreeEnsemble(maximum_cutoff=100.,
                hyperparameters={"ensemble_contract_path": str(contract.relative_to(ROOT))}, seed=13)
            result = model._calibration_data(early_prefixes(dataset(), 100.))
            self.assertEqual(set(result.metadata.uav_id), {"A", "B"})
            self.assertTrue(result.metadata.cutoff.le(100).all())
            rows.loc[rows.uav_id.eq("HELD"), ["RUL", "sensor"]] = 99999.
            rows.to_csv(calibration, index=False)
            changed = model._calibration_data(early_prefixes(dataset(), 100.))
            pd.testing.assert_frame_equal(result.features, changed.features)
            pd.testing.assert_series_equal(result.target, changed.target)

    def test_long_histories_never_reach_specialist_predict(self) -> None:
        class Model:
            def predict(self, data):
                assert data.metadata.cutoff.le(100.).all()
                return np.full(len(data), 7.)
        data = dataset()
        control = np.full(len(data), 20.)
        result = routed_specialist_predictions(Model(), data, control, 100.)
        np.testing.assert_array_equal(result, [7., 7., 20., 7., 20., 20.])

    def test_full_specialist_fit_receives_only_reweighted_early_prefixes(self) -> None:
        model = object.__new__(ShortHistoryTreeEnsemble)
        model.maximum_cutoff = 100.
        with patch.object(ResidualCorrectedTreeEnsembleAdapter, "fit", return_value="ok") as fitted:
            self.assertEqual(model.fit(dataset()), "ok")
        passed = fitted.call_args.args[0]
        self.assertTrue(passed.metadata.cutoff.le(100).all())
        np.testing.assert_allclose(passed.sample_weights.groupby(passed.metadata.uav_id).sum(), 1.)

    def test_real_tiny_specialist_fit_preserves_internal_uav_isolation(self) -> None:
        workflow = read_experiment_config(PIPELINE / "experiments/PE_26/settings.toml")["short_history_workflows"]["PE_26"]
        original = json.loads((ROOT / workflow["source_contract"]).read_text(encoding="utf-8"))
        contract = copy.deepcopy(original)
        contract["member_seeds"] = [13, 37]
        contract["internal_folds"] = 2
        contract["components"]["extra_trees"]["hyperparameters"]["n_estimators"] = 5
        contract["components"]["xgboost"]["training_iterations"] = 3
        contract["residual_model"]["maximum_iterations"] = 3
        contract["residual_model"]["minimum_samples_leaf"] = 2
        metadata = pd.DataFrame({"uav_id": np.repeat([f"U{i}" for i in range(8)], 3),
                                 "cutoff": np.tile([10., 50., 150.], 8), "scenario": ["s1"] * 24,
                                 "sample_id": [f"p{i}" for i in range(24)]})
        features = pd.DataFrame(np.random.default_rng(7).normal(size=(24, len(contract["residual_features"]))),
                                columns=contract["residual_features"])
        target = pd.Series(180. - metadata.cutoff - np.repeat(np.arange(8), 3))
        data = TabularDataset(features, metadata, target, pd.Series([1/3] * 24))
        with test_directory() as path:
            calibration = path / "calibration.csv"
            table = pd.concat([metadata, features], axis=1)
            table["RUL"] = target
            table.to_csv(calibration, index=False)
            contract["calibration_features_path"] = str(calibration.relative_to(ROOT))
            contract_path = path / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            model = ShortHistoryTreeEnsemble(maximum_cutoff=100.,
                hyperparameters={"ensemble_contract_path": str(contract_path.relative_to(ROOT))}, seed=13)
            model.configure_policies(TargetPolicy(mode="piecewise_cap", maximum_rul=125.), PredictionPolicy())
            original_fit_members = model._fit_members
            observed_internal_folds = []
            def verify_members(training, validation, *, retain):
                self.assertTrue(training.metadata.cutoff.le(100.).all())
                if not retain:
                    self.assertFalse(set(training.metadata.uav_id) & set(validation.metadata.uav_id))
                    observed_internal_folds.append(set(validation.metadata.uav_id))
                return original_fit_members(training, validation, retain=retain)
            with patch.object(model, "_fit_members", side_effect=verify_members), threadpool_limits(limits=1):
                summary = model.fit(data)
            self.assertEqual(summary.training_rows, 16)
            self.assertEqual(len(observed_internal_folds), 2)
            self.assertEqual(set.union(*observed_internal_folds), set(metadata.uav_id))
            with threadpool_limits(limits=1):
                prediction = model.predict(early_prefixes(data, 100.))
            self.assertTrue(np.isfinite(prediction).all())
            self.assertTrue((prediction >= 0).all())

    def test_complete_workflow_resumes_without_refitting_completed_cells(self) -> None:
        table, outer, inner = nested_fixture()
        uavs = list("ABCDEF")
        data = TabularDataset(pd.DataFrame({"sensor": np.arange(12, dtype=float)}),
            pd.DataFrame({"uav_id": np.repeat(uavs, 2), "cutoff": np.tile([50., 150.], 6),
                          "scenario": ["s1"] * 12}), pd.Series(np.tile([25., 15.], 6)), pd.Series([.5] * 12))
        workflow = read_experiment_config(PIPELINE / "experiments/PE_26/settings.toml")["short_history_workflows"]["PE_26"]
        workflow.update(split_seeds=[7], outer_fold_count=3, inner_fold_count=2, minimum_fold_wins=2)
        calls = []
        class FakeModel:
            def __init__(self, specialist):
                self.specialist = specialist
            def _calibration_data(self, training):
                return early_prefixes(training, 100.)
            def fit(self, training, validation):
                self.uavs = set(training.metadata.uav_id)
                calls.append((self.specialist, self.uavs))
            def predict(self, held):
                assert not self.uavs & set(held.metadata.uav_id)
                return np.full(len(held), 25. if self.specialist else 30.)
        class FakeAdapter:
            def __init__(self, path):
                pass
            def load_training(self, feature_set):
                return data
            def load_development(self, feature_set):
                return data
            def _copied_path(self, name):
                return ROOT / workflow["tabular_manifest"]
        with test_directory() as root, patch.object(specialist_runner, "TabularDataAdapter", FakeAdapter), \
                patch.object(specialist_runner, "generated_partitions", return_value=(outer, inner)), \
                patch.object(specialist_runner, "new_model", side_effect=lambda workflow, specialist: FakeModel(specialist)), \
                patch.object(specialist_runner, "method_report", return_value={"promoted": False}):
            first = specialist_runner.run(workflow, root)
            self.assertEqual(len(calls), 18)
            self.assertTrue(first["all_prediction_cells_completed"])
            second = specialist_runner.run(workflow, root)
            self.assertEqual(len(calls), 18)
            self.assertTrue(second["nested_inner_predictions_complete"])
            predictions = pd.read_csv(root / "reporting/fold_predictions.csv")
            paired = aligned_methods(predictions.loc[predictions.evaluation_level.eq("outer")], "short_history_specialist")
            long = paired.cutoff.gt(100.)
            np.testing.assert_array_equal(paired.loc[long, "control_prediction"], paired.loc[long, "challenger_prediction"])


if __name__ == "__main__":
    unittest.main()
