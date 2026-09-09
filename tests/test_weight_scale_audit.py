"""PE_29 weight semantics, conditional gates, checkpoint integrity and real workers."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "2_architecture_experiments/1_pipeline_experiments"
sys.path.insert(0, str(PIPELINE))
from test_feature_comparison import fixture, raw_fixture, temporary_directory, workflow as source_workflow
import run_weight_scale_audit as runner
from experiment_config import read_experiment_config
from run_experiment_definition import _execution_plan
from confirmation_utils import evaluation_jobs, select_uavs, prediction_records


def workflow():
    return read_experiment_config(PIPELINE / "experiments/PE_29/settings.toml")["weight_scale_workflows"]["PE_29"]


def payload(job, method, training, held, error):
    result = {"method": method, "training_uavs": sorted(job.training_uavs), "training_rows": len(training),
        "feature_names": list(training.features), "feature_sha256": runner.pe28.digest(list(training.features)),
        "training_sha256": runner.training_digest(training), "audit": {"calibration_uavs": sorted(job.training_uavs)},
        "predictions": prediction_records(job, held, {method: held.target.to_numpy() + error})}
    return result


class WeightScaleAuditTests(unittest.TestCase):
    def test_catalog_and_configuration(self):
        catalog = read_experiment_config(PIPELINE / "pipeline_experiments.toml")
        self.assertIn("PE_29", catalog["weight_scale_workflows"])
        _, _, steps = _execution_plan(PIPELINE / "experiments/PE_29/settings.toml", "PE_29")
        self.assertEqual([s["name"] for s in steps], ["validate_inputs", "evaluate_weight_scale_audit"])
        runner.validate_workflow(workflow())
        for change in ({"confirmation_split_seeds": [20270107, 5]}, {"max_workers": 1.5},
                       {"cpu_threads": 0}, {"audit_minimum_fold_wins": 2.5}, {"reference_prefix_count": 10}):
            with self.assertRaises(ValueError):
                runner.validate_workflow({**workflow(), **change})

    def test_only_dimensionful_xgboost_parameters_are_scaled(self):
        source = source_workflow(); original = deepcopy(source)
        adjusted = runner.adjusted_workflow(source, .05)
        self.assertEqual(source, original)
        self.assertEqual(adjusted["simple"]["catboost"], source["simple"]["catboost"])
        xgb = adjusted["simple"]["xgboost"]
        self.assertEqual({key: xgb[key] for key in ("reg_lambda", "reg_alpha", "gamma", "min_child_weight")},
                         {"reg_lambda": .05, "reg_alpha": 0., "gamma": 0., "min_child_weight": .05})
        for key, value in source["simple"]["xgboost"].items():
            if key not in {"reg_lambda", "reg_alpha", "gamma", "min_child_weight"}:
                self.assertEqual(xgb[key], value)
        for factor in (0, -1, float("nan")):
            with self.assertRaises(ValueError): runner.adjusted_workflow(source, factor)

    def test_real_xgboost_uniform_weight_scaling_equivalence(self):
        from xgboost import XGBRegressor
        rng = np.random.default_rng(29)
        x = rng.normal(size=(100, 4)); y = 30 + 8*x[:, 0] + rng.normal(size=100)
        params = dict(n_estimators=8, max_depth=3, tree_method="hist", device="cpu", n_jobs=1,
                      base_score=30., learning_rate=.1, reg_lambda=1., reg_alpha=.2, gamma=.1, min_child_weight=1.)
        adjusted = runner.adjusted_workflow({"simple": {"xgboost": params}}, .05)["simple"]["xgboost"]
        with threadpool_limits(limits=1):
            baseline = XGBRegressor(**params).fit(x, y, sample_weight=np.ones(len(y))).predict(x)
            actual = XGBRegressor(**adjusted).fit(x, y, sample_weight=np.full(len(y), .05)).predict(x)
        np.testing.assert_allclose(actual, baseline, rtol=0, atol=1e-5)

    def test_dense_weights_labels_and_causal_sparse_agreement(self):
        raw = raw_fixture(2, 8)
        raw = raw.loc[~(raw.uav_id.eq("U01") & raw.flight_cycle.gt(5))].reset_index(drop=True)
        raw["RUL"] = raw.groupby("uav_id").flight_cycle.transform("max") - raw.flight_cycle
        dense = runner.dense_view(raw)
        self.assertEqual(len(dense), 13)
        sums = dense.sample_weights.groupby(dense.metadata.uav_id).sum()
        np.testing.assert_allclose(sums, 1.)
        self.assertEqual(int(dense.target.eq(0).sum()), 2)
        self.assertEqual(dense.features.shape[1], 266)
        histories = runner.validate_raw(raw)
        sparse = runner.sparse_view(dense, histories)
        pd.testing.assert_frame_equal(sparse.features, dense.features)
        pd.testing.assert_series_equal(sparse.sample_weights, dense.sample_weights)
        changed = raw.copy()
        changed.loc[changed.flight_cycle.gt(3), [c for c in raw if c.startswith("telemetry_")]] = -999
        later = runner.dense_view(changed)
        pd.testing.assert_frame_equal(dense.features.loc[dense.metadata.cutoff.le(3)],
                                      later.features.loc[later.metadata.cutoff.le(3)])
        with self.assertRaisesRegex(ValueError, "labels differ"):
            runner.sparse_view(replace(dense, target=dense.target + 1), histories)

    def gate_run(self, statuses):
        calls = []
        def fit(jobs, method): calls.append((jobs, method)); return [{"method": method}]
        def report(stage, *args):
            status = statuses[stage]
            return {"status": status, "winner": "scaled_dense" if status == "promoted" else "control",
                    "promoted": status == "promoted"}
        with temporary_directory() as root:
            (root / "reporting").mkdir()
            result = runner.gated_stages(root, workflow(), ["audit"], ["confirm"],
                {"legacy_sparse": [], "run7": []}, fit, report)
            selection = root / "reporting/selected_configuration.json"
            if result["selected_configuration"]:
                self.assertEqual(json.loads(selection.read_text())["selected_configuration"], result["selected_configuration"])
            return result, calls

    def test_failed_audit_never_fits_density_or_confirmation(self):
        result, calls = self.gate_run({"audit": "no_promotion"})
        self.assertEqual(calls, [(["audit"], "scaled_sparse")])
        self.assertEqual(result["skipped_stages"], ["density", "reference", "confirmation"])
        self.assertFalse(result["promoted"])

    def test_failed_density_retains_sparse_and_confirms_only_sparse(self):
        result, calls = self.gate_run(dict(audit="screening_candidate", density="no_promotion",
                                          reference="screening_candidate", confirmation="no_promotion"))
        self.assertEqual(result["selected_configuration"], "scaled_sparse")
        self.assertEqual(calls[-2:], [(["confirm"], "run7"), (["confirm"], "scaled_sparse")])
        self.assertEqual(len(calls), 4)
        self.assertFalse(result["promoted"])

    def test_failed_reference_never_confirms(self):
        result, calls = self.gate_run(dict(audit="screening_candidate", density="screening_candidate", reference="no_promotion"))
        self.assertEqual(result["selected_configuration"], "scaled_dense")
        self.assertEqual(len(calls), 2)
        self.assertFalse(result["confirmation_completed"])

    def test_passing_gates_confirm_only_frozen_dense(self):
        result, calls = self.gate_run(dict(audit="screening_candidate", density="screening_candidate",
                                          reference="screening_candidate", confirmation="promoted"))
        self.assertTrue(result["promoted"])
        self.assertEqual(calls[-1], (["confirm"], "scaled_dense"))
        self.assertFalse(result["automatic_production_replacement"])

    def test_checkpoint_rejects_changed_weights_targets_and_features(self):
        _, data, outer, inner = fixture()
        job = evaluation_jobs(outer, inner, include_inner=False)[0]
        training, held = select_uavs(data, job.training_uavs), select_uavs(data, job.validation_uavs)
        saved = payload(job, "scaled_sparse", training, held, 1)
        with temporary_directory() as root:
            path = root / "cell.json"
            runner.atomic_json(path, {"payload": saved, "sha256": runner.pe28.digest(saved)})
            self.assertEqual(runner.load_cell(path, job, "scaled_sparse", training, held), saved)
            for changed in (replace(training, sample_weights=training.sample_weights * 2),
                            replace(training, target=training.target + 1), replace(training, features=training.features + 1)):
                with self.assertRaisesRegex(ValueError, "Training values"):
                    runner.load_cell(path, job, "scaled_sparse", changed, held)

    def test_full_gated_workflow_interruption_resume_and_contract(self):
        raw, data, outer, inner = fixture()
        config = workflow(); config.update(audit_split_seed=7, confirmation_split_seeds=[17,27], max_workers=1,
            audit_minimum_fold_wins=2, density_minimum_fold_wins=2, reference_minimum_fold_wins=2,
            confirmation_minimum_fold_wins=4)
        jobs = evaluation_jobs(outer, inner, include_inner=False)
        references = {"legacy_sparse": [], "run7": []}
        for job in jobs[:3]:
            for key, error in (("legacy_sparse", 10), ("run7", 8)):
                references[key].append(payload(job, key, select_uavs(data, job.training_uavs), select_uavs(data, job.validation_uavs), error))
        prepared = (source_workflow(), raw, (data, data), (data, data), jobs, references,
                    [PIPELINE / "experiments/PE_29/settings.toml"], outer, inner)
        calls, interrupted = [], []
        def fit(training, calibration, held, wf):
            if len(calls) == 4 and not interrupted:
                interrupted.append(True); raise RuntimeError("simulated interruption")
            self.assertEqual(set(training.metadata.uav_id), set(calibration.metadata.uav_id))
            self.assertFalse(set(training.metadata.uav_id) & set(held.metadata.uav_id))
            calls.append(len(training))
            error = 1 if len(training) > 100 else 2
            return held.target.to_numpy() + error, {"calibration_uavs": sorted(training.metadata.uav_id.unique())}
        def control_fit(training, calibration, held, wf):
            _, audit = fit(training, calibration, held, wf)
            return held.target.to_numpy() + 8, audit
        with temporary_directory() as root, patch.object(runner, "prepare", return_value=prepared), \
             patch.object(runner, "fit_simple", side_effect=fit), patch.object(runner, "fit_run7", side_effect=control_fit), \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                runner.run(config, root)
            self.assertEqual(len(list((root / "cells").glob("*.json"))), 4)
            result = runner.run(config, root)
            self.assertTrue(result["promoted"])
            self.assertEqual(result["completed_new_recipe_evaluations"], 18)
            self.assertEqual(result["selected_configuration"], "scaled_dense")
            again = runner.run(config, root)
            self.assertEqual(result, again)
            self.assertEqual(len(calls), 18)
            with self.assertRaisesRegex(ValueError, "Registered settings"):
                runner.run({**config, "cpu_threads": 2}, root)
            (root / "reporting/pre_registration.json").unlink()
            with self.assertRaisesRegex(ValueError, "without their pre-registration"):
                runner.run(config, root)

    def test_real_spawned_adjusted_cells_match_serial(self):
        _, data, outer, inner = fixture()
        jobs = evaluation_jobs(outer, inner, include_inner=False)[:2]
        source = source_workflow(); source["inner_fold_count"] = 2
        source["simple"]["early_stopping_rounds"] = 2
        source["simple"]["xgboost"].update(n_estimators=4, max_depth=2, device="cpu")
        source["simple"]["catboost"].update(iterations=4, depth=2, thread_count=1)
        adjusted = runner.adjusted_workflow(source, .05)
        with temporary_directory() as root:
            def tasks(prefix):
                return [(job, "scaled_sparse", select_uavs(data, job.training_uavs), select_uavs(data, job.training_uavs),
                         select_uavs(data, job.validation_uavs), adjusted, root / f"{prefix}_{i}.json", 1)
                        for i, job in enumerate(jobs)]
            actual = runner.execute_pending(tasks("parallel"), 2)
            expected = runner.execute_pending(tasks("serial"), 1)
            self.assertTrue(all(p["audit"]["worker_pid"] != os.getpid() for p in actual))
            for left, right in zip(actual, expected):
                self.assertEqual(left["predictions"], right["predictions"])
                self.assertEqual(left["training_sha256"], right["training_sha256"])
                self.assertEqual(left["audit"]["xgboost_parameters"]["min_child_weight"], .05)


if __name__ == "__main__":
    unittest.main()
