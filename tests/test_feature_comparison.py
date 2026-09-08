"""Causal feature semantics, nested model selection and PE_28 checkpoint gates."""

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "2_architecture_experiments/1_pipeline_experiments"
sys.path.insert(0, str(PIPELINE))
import run_feature_comparison as runner
import feature_comparison_features as features
import feature_comparison_models as models
from tabular_data_adapter import TabularDataset
from experiment_config import read_experiment_config
from confirmation_utils import evaluation_jobs, select_uavs, prediction_records
from run_experiment_definition import _execution_plan


@contextmanager
def temporary_directory():
    base = (ROOT / ".tmp").resolve()
    path = base / ("feature_comparison_" + uuid4().hex)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        path.resolve().relative_to(base)
        shutil.rmtree(path)


def workflow():
    return read_experiment_config(PIPELINE / "experiments/PE_28/settings.toml")["feature_comparison_workflows"]["PE_28"]


def raw_fixture(count=12, cycles=60):
    rows = []
    for uav in range(count):
        for cycle in range(1, cycles + 1):
            row = {"uav_id": f"U{uav:02d}", "flight_cycle": cycle, "RUL": 180. - cycle * 2 + uav}
            row.update({sensor: i + uav * .1 + cycle * .02 + np.sin(cycle * (i+1) * .03)
                        for i, sensor in enumerate(features.SENSORS_22)})
            rows.append(row)
    return pd.DataFrame(rows)


def fixture():
    raw = raw_fixture()
    chosen = raw.loc[raw.flight_cycle.isin([10, 60])].reset_index(drop=True)
    metadata = chosen[["uav_id", "flight_cycle"]].rename(columns={"flight_cycle": "cutoff"})
    metadata["scenario"] = "test_scenario"
    metadata["sample_id"] = [f"sample_{i}" for i in range(len(chosen))]
    matrix = pd.DataFrame([features.prefix_features(group, cutoff, "I_four_sensors")
        for _, group in raw.groupby("uav_id") for cutoff in (10, 60)])
    data = TabularDataset(matrix, metadata, chosen.RUL.astype(float), pd.Series([.5] * len(chosen)))
    uavs = sorted(metadata.uav_id.unique())
    outer = pd.DataFrame([{"split_seed": seed, "outer_fold": i // 4, "uav_id": uav}
                         for seed in (7, 17, 27) for i, uav in enumerate(uavs)])
    inner = pd.DataFrame([{"split_seed": seed, "outer_fold": fold, "inner_fold": i % 2, "uav_id": uav}
        for seed in (7, 17, 27) for fold in range(3)
        for i, uav in enumerate(outer.loc[outer.split_seed.eq(seed) & outer.outer_fold.ne(fold), "uav_id"])])
    return raw, data, outer, inner


class FeatureComparisonTests(unittest.TestCase):
    def test_catalog_and_frozen_matrix(self):
        config = read_experiment_config(PIPELINE / "pipeline_experiments.toml")
        self.assertIn("PE_28", config["feature_comparison_workflows"])
        _, _, steps = _execution_plan(PIPELINE / "experiments/PE_28/settings.toml", "PE_28")
        self.assertEqual([s["name"] for s in steps], ["validate_inputs", "evaluate_feature_comparison"])
        self.assertEqual(len(runner.CONFIGURATIONS), 20)
        runner.validate_workflow(workflow())
        for change in ({"confirmation_split_seeds": [20270107, 20270127]}, {"feature_sets": ["A_current"]},
                       {"target_cap": 100.}, {"screen_minimum_fold_wins": 6}, {"max_workers": 0}, {"max_workers": 1.5}):
            with self.assertRaises(ValueError):
                runner.validate_workflow({**workflow(), **change})

    def test_counts_finiteness_and_causality_for_every_generated_set(self):
        raw = raw_fixture(1)
        changed = raw.copy()
        changed.loc[changed.flight_cycle.gt(20), list(features.SENSORS_22)] = -1e9
        for variant, spec in list(features.FEATURE_SETS.items())[1:]:
            actual = features.prefix_features(raw, 20, variant)
            self.assertEqual(len(actual), spec["count"])
            self.assertEqual(actual, features.prefix_features(changed, 20, variant))
            self.assertEqual(actual, features.prefix_features(raw.iloc[:20], 20, variant))
            self.assertTrue(np.isfinite(list(features.prefix_features(raw, 1, variant).values())).all())

    def test_other_representation_matches_reference_values(self):
        source = (ROOT / "other_pipelines/uav_rul_pipeline_v4 (1).py").read_text()
        nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_slope", "build_features"}]
        namespace = {"pd": pd, "np": np, "ROLLING_WINDOWS": (5, 10, 20)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<reference_features>", "exec"), namespace)
        raw = raw_fixture(1)
        expected = namespace["build_features"](raw, features.SENSORS_22)
        for cutoff in (1, 5, 20, 60):
            actual = features.prefix_features(raw, cutoff, "B_other")
            for sensor in features.SENSORS_22:
                mapping = {"last": sensor, "baseline_delta": f"{sensor}__baseline_delta",
                    "history_mean": f"{sensor}__hist_mean", "history_sd": f"{sensor}__hist_std",
                    "history_slope": f"{sensor}__hist_slope", "last_minus_history_mean": f"{sensor}__last_minus_hist_mean"}
                mapping.update({f"w{w}_{stat}": f"{sensor}__roll{w}_{'std' if stat == 'sd' else stat}"
                                for w in (5, 10, 20) for stat in ("mean", "sd")})
                for name, reference in mapping.items():
                    self.assertAlmostEqual(actual[f"feature__{sensor}__{name}"], expected.iloc[cutoff-1][reference], places=9)

    def test_matched_variants_change_only_declared_concepts(self):
        raw = raw_fixture(1)
        b = features.prefix_features(raw, 30, "B_other")
        f = features.prefix_features(raw, 30, "F_baseline_10")
        self.assertEqual(set(b), set(f))
        self.assertTrue(all("baseline_delta" in name for name in b if b[name] != f[name]))
        for variant in ("G_all_windows", "J_recent_slopes"):
            larger = features.prefix_features(raw, 30, variant)
            self.assertTrue(set(b) < set(larger))
            self.assertEqual(b, {name: larger[name] for name in b})
        for compact, full in (("E_compact_22", "B_other"), ("H_compact_14", "C_other_14")):
            small = features.prefix_features(raw, 30, compact)
            large = features.prefix_features(raw, 30, full)
            self.assertEqual(small, {name: large[name] for name in small})

    def test_build_views_preserves_labels_weights_identity_and_control(self):
        raw, data, _, _ = fixture()
        control = replace(data, features=pd.DataFrame(np.zeros((len(data), 298)), columns=[f"f{i}" for i in range(298)]))
        views = features.build_views(control, control, raw)
        self.assertIs(views["A_current"][0], control)
        for train, dev in views.values():
            pd.testing.assert_series_equal(train.target, control.target)
            pd.testing.assert_series_equal(train.sample_weights, control.sample_weights)
            pd.testing.assert_frame_equal(dev.metadata, control.metadata)
            self.assertGreater(train.target.max(), 125)
        changed = replace(control, target=control.target.clip(upper=125))
        with self.assertRaisesRegex(ValueError, "label differs"):
            features.build_views(changed, control, raw)

    def test_invalid_raw_and_missing_cutoffs_rejected(self):
        raw = raw_fixture(1)
        for bad in (pd.concat([raw, raw.iloc[:1]]), raw.drop(index=2)):
            with self.assertRaises(ValueError):
                features.validate_raw(bad)
        with self.assertRaisesRegex(ValueError, "cutoff"):
            features.prefix_features(raw, 61, "B_other")

    def test_simple_nested_selection_and_raw_blend_targets(self):
        _, data, outer, inner = fixture()
        job = evaluation_jobs(outer, inner, include_inner=False)[0]
        training = select_uavs(data, job.training_uavs)
        held = select_uavs(data, job.validation_uavs)
        config = workflow(); config["inner_fold_count"] = 2
        made = []
        class FakeModel:
            best_iteration = 2
            def __init__(self, family): self.family = family
            def get_best_iteration(self): return 3
            def predict(self, x): return np.full(len(x), 100 if self.family == "xgboost" else 110)
        def factory(family, settings, iterations=None, stopping=False):
            made.append((family, iterations, stopping)); return FakeModel(family)
        fits = []
        def fit(model, family, train, cap, stopping=None):
            ids = set(train.metadata.uav_id)
            self.assertFalse(ids & set(held.metadata.uav_id))
            if stopping is not None:
                self.assertFalse(ids & set(stopping.metadata.uav_id))
            fits.append((ids, set(stopping.metadata.uav_id) if stopping is not None else set()))
            return model
        with patch.object(models, "make_estimator", side_effect=factory), patch.object(models, "fit_estimator", side_effect=fit):
            prediction, audit = models.fit_simple(training, data, held, config)
        self.assertEqual(len(fits), 10)
        self.assertEqual(audit["final_iterations"], {"xgboost": 3, "catboost": 4})
        self.assertEqual(audit["xgboost_weight"], 0.0)  # raw mean >110; capped mean <100 would select XGB
        self.assertTrue(np.all(prediction == 110))
        for fold in audit["inner_selection"]:
            fit_ids, stop_ids, oof_ids = map(set, (fold["fit_uavs"], fold["stopping_uavs"], fold["oof_uavs"]))
            self.assertFalse(fit_ids & stop_ids or fit_ids & oof_ids or stop_ids & oof_ids)
            self.assertEqual(fit_ids | stop_ids | oof_ids, set(job.training_uavs))
        self.assertEqual(fits[-1][0], set(job.training_uavs))

    def test_real_small_simple_fit(self):
        _, data, outer, inner = fixture()
        job = evaluation_jobs(outer, inner, include_inner=False)[0]
        config = workflow(); config["inner_fold_count"] = 2
        config["simple"]["early_stopping_rounds"] = 2
        config["simple"]["xgboost"].update(n_estimators=4, max_depth=2, device="cpu")
        config["simple"]["catboost"].update(iterations=4, depth=2, thread_count=1)
        with threadpool_limits(limits=1):
            prediction, audit = models.fit_simple(select_uavs(data, job.training_uavs), data,
                select_uavs(data, job.validation_uavs), config)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(audit["base_estimator_fits"], 10)
        self.assertEqual(set(audit["calibration_uavs"]), set(job.training_uavs))

    def test_real_small_run7_uses_generated_calibration(self):
        from model_registry import ModelAdapterFactory
        original = ModelAdapterFactory.create
        _, data, outer, inner = fixture()
        job = evaluation_jobs(outer, inner, include_inner=False)[0]
        config = workflow(); config["inner_fold_count"] = 2
        def tiny(factory, *args, **kwargs):
            model = original(factory, *args, **kwargs)
            model.internal_folds = 2; model.member_seeds = (13,)
            model.contract["components"]["extra_trees"]["hyperparameters"].update(n_estimators=2, max_depth=2)
            model.contract["components"]["xgboost"]["training_iterations"] = 2
            model.contract["residual_model"]["maximum_iterations"] = 2
            model.calibration_path = Path("does_not_exist.csv")
            return model
        with patch.object(ModelAdapterFactory, "create", tiny), threadpool_limits(limits=1), redirect_stdout(io.StringIO()):
            prediction, audit = models.fit_run7(select_uavs(data, job.training_uavs), data,
                select_uavs(data, job.validation_uavs), config)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(set(audit["calibration_uavs"]), set(job.training_uavs))

    def test_screen_failure_skips_confirmation_and_promotion(self):
        with temporary_directory() as root:
            (root / "reporting").mkdir()
            calls = []
            def fit(jobs, methods): calls.append((jobs, methods)); return []
            result = runner.gated_stages(root, workflow(), ["screen"], ["confirmation"], fit,
                lambda *args: {"winner": "control", "status": "no_promotion", "promoted": False})
            self.assertEqual(len(calls), 1)
            self.assertFalse(result["confirmation_completed"])
            self.assertEqual(json.loads((root / "stages/confirmation/reporting/winner_manifest.json").read_text())["status"], "skipped")

    def test_only_one_frozen_challenger_enters_confirmation(self):
        with temporary_directory() as root:
            (root / "reporting").mkdir()
            calls = []
            def fit(jobs, methods): calls.append((jobs, methods)); return []
            def report(stage, payloads, challenger=None):
                return {"winner": "xgb_cat__B_other", "status": "screening_candidate" if stage == "screen" else "promoted",
                        "promoted": stage == "confirmation"}
            result = runner.gated_stages(root, workflow(), ["screen"], ["confirmation"], fit, report)
            self.assertEqual(calls[1], (["confirmation"], ["control", "xgb_cat__B_other"]))
            self.assertTrue(result["confirmation_completed"])
            self.assertTrue(result["promoted"])

    def test_checkpoint_rejects_corruption_and_wrong_membership(self):
        _, data, outer, inner = fixture()
        job = evaluation_jobs(outer, inner, include_inner=False)[0]
        training, held = select_uavs(data, job.training_uavs), select_uavs(data, job.validation_uavs)
        payload = {"method": "control", "feature_names": list(training.features),
            "training_uavs": sorted(job.training_uavs), "training_rows": len(training),
            "feature_sha256": runner.digest(list(training.features)), "audit": {"calibration_uavs": sorted(job.training_uavs)},
            "predictions": prediction_records(job, held, {"control": np.zeros(len(held))})}
        with temporary_directory() as root:
            path = root / "cell.json"
            runner.atomic_json(path, {"payload": payload, "sha256": runner.digest(payload)})
            self.assertIsNotNone(runner.load_cell(path, job, "control", training, held))
            payload["audit"]["calibration_uavs"] = sorted(job.validation_uavs)
            runner.atomic_json(path, {"payload": payload, "sha256": runner.digest(payload)})
            with self.assertRaisesRegex(ValueError, "membership"):
                runner.load_cell(path, job, "control", training, held)
            runner.atomic_json(path, {"payload": payload, "sha256": "corrupt"})
            with self.assertRaisesRegex(ValueError, "checksum"):
                runner.load_cell(path, job, "control", training, held)

    def test_full_workflow_reports_and_resumes_without_fits(self):
        _, data, outer, inner = fixture()
        config = workflow(); config.update(screen_split_seed=7, confirmation_split_seeds=[17,27],
            outer_fold_count=3, inner_fold_count=2, screen_minimum_fold_wins=2, confirmation_minimum_fold_wins=4, max_workers=1)
        views = {variant: (data, data) for variant in features.FEATURE_SETS}
        calls = []
        interrupted = []
        def fit(training, calibration, held, wf):
            if len(calls) == 4 and not interrupted:
                interrupted.append(True)
                raise RuntimeError("Simulated interruption between cells")
            self.assertFalse(set(training.metadata.uav_id) & set(held.metadata.uav_id))
            self.assertEqual(set(training.metadata.uav_id), set(calibration.metadata.uav_id))
            calls.append(len(training))
            return held.target.to_numpy() + 1, {"calibration_uavs": sorted(training.metadata.uav_id.unique()),
                "fit_seconds": .1, "predict_seconds": .01, "base_estimator_fits": 1}
        def control_fit(training, calibration, held, wf):
            prediction, audit = fit(training, calibration, held, wf)
            return prediction + 9, audit
        class Adapter:
            def _copied_path(self, name): return PIPELINE / "experiments/PE_28/settings.toml"
        with temporary_directory() as root, patch.object(runner, "prepare", return_value=(Adapter(), views, .1)), \
             patch.object(runner, "generated_partitions", return_value=(outer, inner)), \
             patch.object(runner, "fit_run7", side_effect=control_fit), patch.object(runner, "fit_simple", side_effect=fit), \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
                runner.run(config, root)
            self.assertEqual(len(list((root / "cells").glob("*.json"))), 4)
            result = runner.run(config, root)
            self.assertTrue(result["promoted"])
            self.assertEqual(len(calls), 72)
            again = runner.run(config, root)
            self.assertEqual(len(calls), 72)
            self.assertEqual(result["selected_configuration"], again["selected_configuration"])
            screen = pd.read_csv(root / "stages/screen/reporting/fold_metrics.csv")
            confirmation = pd.read_csv(root / "stages/confirmation/reporting/fold_metrics.csv")
            self.assertEqual(screen.method.nunique(), 20)
            self.assertFalse(set(screen.outer_fold) & set(confirmation.outer_fold))
            contrasts = pd.read_csv(root / "stages/screen/reporting/matched_contrasts.csv")
            self.assertEqual(len(contrasts), 32)
            self.assertTrue((root / "stages/screen/reporting/summary_with_costs.csv").is_file())
            cell = next((root / "cells").glob("*.json")); cell.write_text('{"payload": {}, "sha256": "bad"}')
            with self.assertRaisesRegex(ValueError, "checksum"):
                runner.run(config, root)
            self.assertEqual(len(calls), 72)

    def test_parallel_queue_is_bounded_and_results_keep_declared_order(self):
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        state = {"active": 0, "peak": 0}
        finished = []
        def worker(index):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            if index < 2:
                barrier.wait(timeout=5)
            time.sleep(.08 if index == 0 else .005)
            with lock:
                state["active"] -= 1
                finished.append(index)
            return index
        with patch.object(runner, "ProcessPoolExecutor", side_effect=lambda **kw: ThreadPoolExecutor(max_workers=kw["max_workers"])), \
             patch.object(runner, "execute_cell", side_effect=worker):
            result = runner.execute_pending([(i,) for i in range(5)], 2)
        self.assertEqual(result, list(range(5)))
        self.assertEqual(state["peak"], 2)
        self.assertEqual(finished[0], 1)

    def test_worker_failure_stops_new_dispatch_and_finishes_active_cell(self):
        barrier = threading.Barrier(2)
        started, finished = [], []
        def worker(index):
            started.append(index)
            barrier.wait(timeout=5)
            if index == 0:
                raise RuntimeError("Worker failed")
            time.sleep(.05)
            finished.append(index)
            return index
        with patch.object(runner, "ProcessPoolExecutor", side_effect=lambda **kw: ThreadPoolExecutor(max_workers=kw["max_workers"])), \
             patch.object(runner, "execute_cell", side_effect=worker):
            with self.assertRaisesRegex(RuntimeError, "Worker failed"):
                runner.execute_pending([(i,) for i in range(5)], 2)
        self.assertEqual(set(started), {0, 1})
        self.assertEqual(finished, [1])

    def test_real_spawned_cells_match_serial_and_resume_without_pool(self):
        _, data, outer, inner = fixture()
        jobs = evaluation_jobs(outer, inner, include_inner=False)[:2]
        config = workflow(); config.update(inner_fold_count=2, cpu_threads=1, max_workers=2)
        config["simple"]["early_stopping_rounds"] = 2
        config["simple"]["xgboost"].update(n_estimators=4, max_depth=2, device="cpu")
        config["simple"]["catboost"].update(iterations=4, depth=2, thread_count=1)
        method = "xgb_cat__I_four_sensors"
        views = {"I_four_sensors": (data, data)}
        with temporary_directory() as root:
            parallel = root / "parallel"; parallel.mkdir()
            serial = root / "serial"; serial.mkdir()
            results = runner.fit_stage_cells(jobs, [method], views, parallel, config)
            self.assertTrue(all(p["audit"]["worker_pid"] != os.getpid() for p in results))
            config["max_workers"] = 1
            expected = runner.fit_stage_cells(jobs, [method], views, serial, config)
            for actual, reference in zip(results, expected):
                np.testing.assert_allclose([r["predicted_rul"] for r in actual["predictions"]],
                    [r["predicted_rul"] for r in reference["predictions"]], rtol=0, atol=1e-12)
                self.assertEqual(actual["training_uavs"], reference["training_uavs"])
            config["max_workers"] = 2
            with patch.object(runner, "ProcessPoolExecutor", side_effect=AssertionError("Completed cells must not start a pool")):
                resumed = runner.fit_stage_cells(jobs, [method], views, parallel, config)
            self.assertEqual(results, resumed)


if __name__ == "__main__":
    unittest.main()
