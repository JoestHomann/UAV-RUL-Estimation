"""PE_30 fixed recipe, independent density screen, checkpoints and workers."""

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

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "2_architecture_experiments/1_pipeline_experiments"
sys.path.insert(0, str(PIPELINE))
import run_density_comparison as runner
from test_feature_comparison import fixture, temporary_directory, workflow as source_workflow
from test_weight_scale_audit import payload
from experiment_config import read_experiment_config
from run_experiment_definition import _execution_plan
from confirmation_utils import evaluation_jobs, select_uavs


def workflow():
    return read_experiment_config(PIPELINE / "experiments/PE_30/settings.toml")["density_workflows"]["PE_30"]


class DensityComparisonTests(unittest.TestCase):
    def test_catalog_and_fixed_settings(self):
        catalog = read_experiment_config(PIPELINE / "pipeline_experiments.toml")
        self.assertIn("PE_30", catalog["density_workflows"])
        _, _, steps = _execution_plan(PIPELINE / "experiments/PE_30/settings.toml", "PE_30")
        self.assertEqual([s["name"] for s in steps], ["validate_inputs", "evaluate_density_comparison"])
        runner.validate_workflow(workflow())
        for change in ({"max_workers": 0}, {"cpu_threads": 1.5}, {"screen_split_seed": -1},
                       {"confirmation_split_seeds": [1,1]}, {"feature_variant": "A_current"},
                       {"density_minimum_fold_wins": 6}, {"reference_minimum_relative_rmse_improvement": float("nan")}):
            with self.assertRaises(ValueError): runner.validate_workflow({**workflow(), **change})

    def gate_run(self, density, reference, confirmation="no_promotion"):
        calls, reports = [], []
        def fit(jobs, method): calls.append((jobs, method)); return [{"method": method}]
        def report(stage, *args):
            reports.append(stage)
            status = {"density": density, "reference": reference, "confirmation": confirmation}[stage]
            return {"status": status, "promoted": status == "promoted", "winner": runner.CANDIDATE}
        with temporary_directory() as root:
            (root / "reporting").mkdir()
            result = runner.gated_stages(root, ["screen"], ["confirmation"],
                {"legacy_sparse": [], "run7": []}, fit, report)
            frozen = json.loads((root / "reporting/selected_configuration.json").read_text())
            self.assertEqual(frozen["confirmation_eligible"], result["confirmation_completed"])
            frozen["density_passed"] = not frozen["density_passed"]
            runner.atomic_json(root / "reporting/selected_configuration.json", frozen)
            with self.assertRaisesRegex(ValueError, "Frozen PE_30"):
                runner.gated_stages(root, ["screen"], ["confirmation"], {"legacy_sparse": [], "run7": []},
                    lambda *args: [], lambda *args: {"status": "screening_candidate" if args[0] == "density" and density == "screening_candidate" else "no_promotion"})
        return result, calls, reports

    def test_both_comparisons_reported_and_both_gates_required(self):
        for density, reference in (("no_promotion", "no_promotion"), ("no_promotion", "screening_candidate"),
                                   ("screening_candidate", "no_promotion")):
            result, calls, reports = self.gate_run(density, reference)
            self.assertEqual(calls, [(["screen"], runner.CANDIDATE)])
            self.assertEqual(reports, ["density", "reference"])
            self.assertFalse(result["confirmation_completed"])
            self.assertFalse(result["promoted"])

    def test_confirmation_uses_only_original_dense_and_no_fallback(self):
        for status in ("promoted", "no_promotion"):
            result, calls, _ = self.gate_run("screening_candidate", "screening_candidate", status)
            self.assertEqual(calls, [(["screen"], runner.CANDIDATE), (["confirmation"], "run7"),
                                     (["confirmation"], runner.CANDIDATE)])
            self.assertEqual(result["promoted"], status == "promoted")
            self.assertFalse(result["automatic_production_replacement"])

    def test_prepare_builds_dense_with_equal_uav_weights_and_original_source(self):
        raw, data, outer, inner = fixture()
        data = runner.source_tools.sparse_view(data, runner.source_tools.validate_raw(raw))
        source = source_workflow(); original = deepcopy(source)
        prepared = (source, raw, (data, data), (data, data), [], {}, [], outer, inner)
        with patch.object(runner.source_tools, "prepare", return_value=prepared) as inherited, redirect_stdout(io.StringIO()):
            result = runner.prepare(workflow())
        self.assertEqual(inherited.call_args.args[0]["audit_split_seed"], workflow()["screen_split_seed"])
        dense, dev = result[1]
        self.assertEqual(result[0], original)
        self.assertEqual(len(dense), len(raw))
        np.testing.assert_allclose(dense.sample_weights.groupby(dense.metadata.uav_id).sum(), 1.)
        self.assertIs(dev, data)

    def test_original_parameters_reach_fitter_and_checkpoint_rejects_changed_weights(self):
        raw, data, outer, inner = fixture()
        # This test uses the dense feature schema for both training and held inputs.
        dense = runner.source_tools.dense_view(raw)
        dense = replace(dense, metadata=dense.metadata.assign(scenario="synthetic"))
        job = evaluation_jobs(outer, inner, include_inner=False)[0]
        training, held = select_uavs(dense, job.training_uavs), select_uavs(dense, job.validation_uavs)
        source = source_workflow(); original = deepcopy(source)
        def fitter(train, calibration, validation, settings):
            self.assertEqual(settings, original)
            self.assertEqual(set(calibration.metadata.uav_id), set(train.metadata.uav_id))
            self.assertFalse(set(train.metadata.uav_id) & set(validation.metadata.uav_id))
            return validation.target.to_numpy(), {"calibration_uavs": sorted(train.metadata.uav_id.unique())}
        with temporary_directory() as root, patch.object(runner, "fit_simple", side_effect=fitter), redirect_stdout(io.StringIO()):
            path = root / "cell.json"
            saved = runner.execute_cell(job, runner.CANDIDATE, training, training, held, source, path, 1)
            self.assertEqual(saved["audit"]["sampling"], "all_cycles")
            self.assertFalse(saved["audit"]["regularization_adjusted"])
            self.assertEqual(saved["audit"]["xgboost_parameters"]["reg_lambda"], 1.)
            with self.assertRaisesRegex(ValueError, "weights changed"):
                runner.load_cell(path, job, runner.CANDIDATE, replace(training, sample_weights=training.sample_weights*2), held)
            path.write_text('{"payload": {}, "sha256": "bad"}')
            with self.assertRaisesRegex(ValueError, "checksum"):
                runner.load_cell(path, job, runner.CANDIDATE, training, held)

    def test_preflight_and_full_workflow_interruption_resume(self):
        raw, data, outer, inner = fixture()
        dense = runner.source_tools.dense_view(raw)
        config = workflow(); config.update(screen_split_seed=7, confirmation_split_seeds=[17,27], max_workers=1,
            density_minimum_fold_wins=2, reference_minimum_fold_wins=2, confirmation_minimum_fold_wins=4)
        jobs = evaluation_jobs(outer, inner, include_inner=False)
        references = {"legacy_sparse": [], "run7": []}
        for job in jobs[:3]:
            for key, error in (("legacy_sparse", 10), ("run7", 8)):
                references[key].append(payload(job, key, select_uavs(data, job.training_uavs), select_uavs(data, job.validation_uavs), error))
        source = source_workflow()
        prepared = (source, (dense, data), (data, data), jobs, references,
                    [PIPELINE / "experiments/PE_30/settings.toml"], outer, inner, len(data))
        calls, interrupted = [], []
        def fit(training, calibration, held, wf):
            if len(calls) == 4 and not interrupted:
                interrupted.append(True); raise RuntimeError("simulated interruption")
            self.assertEqual(wf, source)
            self.assertEqual(set(training.metadata.uav_id), set(calibration.metadata.uav_id))
            self.assertFalse(set(training.metadata.uav_id) & set(held.metadata.uav_id))
            calls.append(len(training))
            return held.target.to_numpy() + (1 if len(training) > 100 else 8), {
                "calibration_uavs": sorted(training.metadata.uav_id.unique())}
        with temporary_directory() as root, patch.object(runner, "prepare", return_value=prepared), \
             patch.object(runner, "fit_simple", side_effect=fit), patch.object(runner, "fit_run7", side_effect=fit), \
             redirect_stdout(io.StringIO()):
            ready = runner.run(config, root, check_only=True)
            self.assertTrue(ready["ready"])
            self.assertFalse(calls)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                runner.run(config, root)
            self.assertEqual(len(list((root / "cells").glob("*.json"))), 4)
            result = runner.run(config, root)
            self.assertTrue(result["promoted"])
            self.assertEqual(result["completed_new_recipe_evaluations"], 15)
            again = runner.run(config, root)
            self.assertEqual(result, again)
            self.assertEqual(len(calls), 15)
            with self.assertRaisesRegex(ValueError, "Registered settings"):
                runner.run({**config, "cpu_threads": 2}, root)
            (root / "reporting/pre_registration.json").unlink()
            with self.assertRaisesRegex(ValueError, "without their pre-registration"):
                runner.run(config, root)

    def test_real_spawned_dense_cells_match_serial(self):
        raw, data, outer, inner = fixture()
        dense = runner.source_tools.dense_view(raw)
        dev = runner.source_tools.sparse_view(data, runner.source_tools.validate_raw(raw))
        jobs = evaluation_jobs(outer, inner, include_inner=False)[:2]
        source = source_workflow(); source["inner_fold_count"] = 2
        source["simple"]["early_stopping_rounds"] = 2
        source["simple"]["xgboost"].update(n_estimators=4, max_depth=2, device="cpu")
        source["simple"]["catboost"].update(iterations=4, depth=2, thread_count=1)
        with temporary_directory() as root:
            def tasks(prefix):
                return [(job, runner.CANDIDATE, select_uavs(dense, job.training_uavs), select_uavs(dev, job.training_uavs),
                         select_uavs(dev, job.validation_uavs), source, root / f"{prefix}_{i}.json", 1)
                        for i, job in enumerate(jobs)]
            actual = runner.execute_pending(tasks("parallel"), 2)
            expected = runner.execute_pending(tasks("serial"), 1)
            self.assertTrue(all(p["audit"]["worker_pid"] != os.getpid() for p in actual))
            for left, right in zip(actual, expected):
                self.assertEqual(left["predictions"], right["predictions"])
                self.assertEqual(left["training_sha256"], right["training_sha256"])
                self.assertEqual(left["audit"]["sampling"], "all_cycles")
                self.assertEqual(left["audit"]["xgboost_parameters"]["reg_lambda"], 1.)


if __name__ == "__main__":
    unittest.main()
