"""PE_31 fit-local weighting, outer isolation, complete selection, reports and resume."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
import multiprocessing
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "2_architecture_experiments/1_pipeline_experiments"
sys.path.insert(0, str(PIPELINE))
import run_campaign as runner
import campaign_data as data_tools
import campaign_models as models
import campaign_reporting as reports
from test_feature_comparison import fixture, temporary_directory, workflow as original_workflow
from experiment_config import read_experiment_config
from confirmation_utils import evaluation_jobs, select_uavs
from run_experiment_definition import _execution_plan


def workflow():
    return read_experiment_config(PIPELINE / "experiments/PE_31/settings.toml")["campaign_workflows"]["PE_31"]


def prepared_fixture():
    raw, data, outer, inner = fixture()
    pieces = []
    for suite, endpoint_seed in (("historical", -1), ("nominal", 1), ("unrestricted", 2)):
        pieces.append(replace(data, metadata=data.metadata.assign(suite=suite, endpoint_seed=endpoint_seed, scenario=suite)))
    dev = replace(data, features=pd.concat([p.features for p in pieces], ignore_index=True),
        metadata=pd.concat([p.metadata for p in pieces], ignore_index=True),
        target=pd.concat([p.target for p in pieces], ignore_index=True), sample_weights=None)
    bundle = {"train": {"sparse_B": data, "dense_B": data, "sparse_E": data, "dense_E": data, "run7": data},
              "dev": {"B": dev, "E": dev, "run7": dev}, "raw": raw}
    jobs = evaluation_jobs(outer, inner, include_inner=False)
    source = original_workflow(); source["inner_fold_count"] = 2
    source["simple"]["early_stopping_rounds"] = 2
    source["simple"]["xgboost"].update(n_estimators=4, max_depth=2, device="cpu")
    source["simple"]["catboost"].update(iterations=4, depth=2, thread_count=1)
    source["campaign_extra_trees_estimators"] = 4
    return source, bundle, jobs, outer, inner


class CampaignTests(unittest.TestCase):
    def test_catalog_registry_and_budget(self):
        cfg = read_experiment_config(PIPELINE / "pipeline_experiments.toml")
        self.assertIn("PE_31", cfg["campaign_workflows"])
        _, _, steps = _execution_plan(PIPELINE / "experiments/PE_31/settings.toml", "PE_31")
        self.assertEqual([x["name"] for x in steps], ["reproduce_submission", "validate_inputs", "interaction_benchmark", "nested_joint_search"])
        runner.validate(workflow())
        self.assertEqual(len(data_tools.POLICIES), 6); self.assertEqual(len(data_tools.recipes()), 12)
        self.assertEqual(runner.budget(workflow(), range(10))["maximum_recipe_evaluations"], 1360)
        for change in ({"model_seeds": [0]}, {"endpoint_seeds": [1,1,2]}, {"max_workers": 0}, {"selection_folds": 2}):
            with self.assertRaises(ValueError): runner.validate({**workflow(), **change})

    def test_six_policy_weight_invariants_in_actual_subsets(self):
        _, data, _, _ = fixture()
        mask = ~(data.metadata.uav_id.eq("U00") & data.metadata.cutoff.eq(60))
        from advanced_r2_utils import subset_dataset
        data = subset_dataset(data, mask.to_numpy())
        for policy, (_, influence, mass) in data_tools.POLICIES.items():
            actual = data_tools.weight_data(data, policy)
            target = len(data) if mass == "row_count" else data.metadata.uav_id.nunique()
            self.assertAlmostEqual(actual.sample_weights.sum(), target)
            if influence == "uav":
                sums = actual.sample_weights.groupby(actual.metadata.uav_id).sum()
                np.testing.assert_allclose(sums, sums.iloc[0])
            else:
                np.testing.assert_allclose(actual.sample_weights, actual.sample_weights.iloc[0])
        for policy in ("sparse_uav_mass", "sparse_unit_mean"):
            balanced = select_uavs(data, {"U01", "U02"})
            values = data_tools.weight_data(balanced, policy).sample_weights
            np.testing.assert_allclose(values, .5 if policy.endswith("mass") else 1.)

    def test_endpoint_draws_do_not_depend_on_other_uav_labels(self):
        raw, hist, _, _ = fixture()
        cutoffs = np.array([10,20,30,40,50])
        first = data_tools.endpoint_suites(raw, hist, cutoffs, [1,2,3])
        altered = raw.copy(); altered.loc[altered.uav_id.eq("U00"), "RUL"] += 5
        second = data_tools.endpoint_suites(altered, hist, cutoffs, [1,2,3])
        left = first.loc[first.uav_id.ne("U00")].reset_index(drop=True)
        right = second.loc[second.uav_id.ne("U00")].reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)
        self.assertTrue(first.loc[first.suite.eq("nominal"), "RUL"].between(1,125).all())

    def test_real_candidate_families_and_outer_label_invariance(self):
        source, bundle, jobs, _, _ = prepared_fixture(); job = jobs[0]
        train = select_uavs(bundle["train"]["sparse_B"], job.training_uavs)
        calibration = select_uavs(bundle["dev"]["B"], job.training_uavs)
        held = select_uavs(bundle["dev"]["B"], job.validation_uavs)
        for family in ("blend", "catboost", "extra_trees"):
            recipe = {"view": "B", "family": family, "capacity": "original"}
            pred, audit = models.fit_candidate(train, calibration, held, source, "dense_unit_rows", recipe, 0)
            again, other = models.fit_candidate(train, calibration, replace(held, target=held.target+999), source, "dense_unit_rows", recipe, 0)
            np.testing.assert_allclose(pred, again, atol=0, rtol=0)
            self.assertEqual(audit["inner_selection"], other["inner_selection"])
            self.assertEqual(audit["weight_sum"], len(train))
            for partition in audit["inner_selection"]:
                fit, stop, oof = map(set, (partition["fit_uavs"], partition["stopping_uavs"], partition["oof_uavs"]))
                self.assertFalse(fit & stop or fit & oof or stop & oof)

    def test_engine_resume_and_integrity(self):
        source, data, jobs, _, _ = prepared_fixture(); job = jobs[0]
        calls = []
        def fit(train, calibration, held, *args):
            calls.append(True)
            return held.target.to_numpy()+1, {"base_estimator_fits": 1}
        with temporary_directory() as root, patch.object(models, "fit_candidate", side_effect=fit), redirect_stdout(io.StringIO()):
            engine = runner.Engine(root, job, 0, source, data)
            first = engine.fit(job.training_uavs, job.validation_uavs)
            second = engine.fit(job.training_uavs, job.validation_uavs)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(len(calls), 1)
            path = next(engine.directory.glob("*.json")); obj = json.loads(path.read_text())
            obj["payload"]["predictions"][0]["predicted_rul"] += 1
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, "checksum"):
                engine.fit(job.training_uavs, job.validation_uavs)

    def test_nested_selection_excludes_outer_labels_and_records_all_choices(self):
        source, data, jobs, _, _ = prepared_fixture(); job = jobs[0]
        fit_calls = []
        def fit(train, calibration, held, source, policy, recipe, seed):
            fit_calls.append((set(train.metadata.uav_id), set(held.metadata.uav_id)))
            self.assertFalse(set(train.metadata.uav_id) & set(held.metadata.uav_id))
            error = 1 if policy == "dense_unit_rows" and recipe["family"] == "catboost" else 2 if policy == "dense_unit_rows" else 4
            return held.target.to_numpy()+error, {"base_estimator_fits": 1}
        def control(train, calibration, held, source): return held.target.to_numpy()+8, {"base_estimator_fits": 1}
        with temporary_directory() as root, patch.object(models, "fit_candidate", side_effect=fit), \
             patch.object(models, "fit_run7", side_effect=control), redirect_stdout(io.StringIO()):
            engine = runner.Engine(root, job, 0, source, data)
            result = runner.nested_search(engine, workflow())
            selected = json.loads((engine.directory / "selection.json").read_text())
            self.assertEqual(selected["policy"], "dense_unit_rows")
            self.assertEqual(len(selected["candidate_scores"]), 60)
            self.assertEqual(set(result.method), {"run7", "selected_standalone", "selected_blend"})
            self.assertFalse(selected["selection_uses_outer_labels"])
            previous = len(fit_calls)
            runner.nested_search(engine, workflow())
            self.assertEqual(len(fit_calls), previous)
            # Mutate only outer labels, rebuild in a separate directory; selection must stay identical.
            changed = deepcopy(data)
            for view, dev in changed["dev"].items():
                dev.target.loc[dev.metadata.uav_id.isin(job.validation_uavs)] += 999
            other = runner.Engine(root / "changed", job, 0, source, changed)
            runner.nested_search(other, workflow())
            self.assertEqual(selected, json.loads((other.directory / "selection.json").read_text()))

    def test_complete_workflow_partial_resume_and_reports(self):
        source, data, jobs, outer, inner = prepared_fixture()
        jobs = jobs[:1]
        cfg = workflow(); cfg["max_workers"] = 1; cfg["bootstrap_repetitions"] = 20
        prepared = (source, data, jobs, [PIPELINE / "experiments/PE_31/settings.toml"], outer, inner)
        calls, interrupted = [], []
        def fit(train, calibration, held, *args):
            if len(calls) == 4 and not interrupted:
                interrupted.append(True); raise RuntimeError("simulated interruption")
            calls.append(True)
            return held.target.to_numpy()+1, {"base_estimator_fits": 1, "fit_seconds": .1}
        def control(train, calibration, held, *args):
            prediction, audit = fit(train, calibration, held)
            return prediction+7, audit
        def reference(raw, train_ids, held, *args): return fit(None, None, held)
        with temporary_directory() as root, patch.object(data_tools, "prepare", return_value=prepared), \
             patch.object(models, "fit_candidate", side_effect=fit), patch.object(models, "fit_run7", side_effect=control), \
             patch.object(models, "reference_protocol", side_effect=reference), redirect_stdout(io.StringIO()):
            check = runner.run(cfg, root, "check")
            self.assertTrue(check["ready"]); self.assertFalse(calls)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                runner.run(cfg, root, "all")
            result = runner.run(cfg, root, "all")
            self.assertTrue(result["full_campaign_completed"])
            self.assertTrue(result["results"]["search"]["promoted"])
            old_count = len(calls)
            runner.run(cfg, root, "all")
            self.assertEqual(len(calls), old_count)
            for name in ("variation.csv", "interaction_contrasts.csv", "paired_comparisons.csv"):
                self.assertTrue((root / "stages/benchmark/reporting" / name).is_file())
            with self.assertRaisesRegex(ValueError, "Registered settings"):
                runner.run({**cfg, "cpu_threads": 2}, root, "search")

    def test_real_reference_protocol_with_tiny_models(self):
        source, data, jobs, _, _ = prepared_fixture(); job = jobs[0]
        held = select_uavs(data["dev"]["B"], job.validation_uavs)
        original = models.importlib.util.spec_from_file_location
        def spec(name, path):
            value = original(name, path)
            execute = value.loader.exec_module
            def tiny(module):
                execute(module)
                module.XGB_PARAMS.update(n_estimators=4, max_depth=2)
                module.CATBOOST_PARAMS.update(iterations=4, depth=2, allow_writing_files=False)
                module.XGB_EARLY_STOPPING_ROUNDS = 2; module.CATBOOST_EARLY_STOPPING_ROUNDS = 2
            value.loader.exec_module = tiny
            return value
        with patch.object(models.importlib.util, "spec_from_file_location", side_effect=spec):
            pred, audit = models.reference_protocol(data["raw"], set(job.training_uavs), held, source, 0)
        self.assertTrue(np.isfinite(pred).all())
        self.assertEqual(audit["base_estimator_fits"], 12)
        self.assertFalse(set(audit["final_fit_uavs"]) & set(audit["final_stopping_uavs"]))

    def test_real_spawned_candidate_matches_serial(self):
        source, data, jobs, _, _ = prepared_fixture(); job = jobs[0]
        args = (select_uavs(data["train"]["sparse_B"], job.training_uavs), select_uavs(data["dev"]["B"], job.training_uavs),
                select_uavs(data["dev"]["B"], job.validation_uavs), source, "dense_unit_rows", data_tools.base_recipe(), 0)
        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(models.fit_candidate, *args) for _ in range(2)]
            actual = [f.result() for f in futures]
        expected, _ = models.fit_candidate(*args)
        for pred, audit in actual:
            np.testing.assert_allclose(pred, expected, atol=0, rtol=0)
            self.assertEqual(audit["weight_sum"], len(args[0]))


if __name__ == "__main__": unittest.main()
