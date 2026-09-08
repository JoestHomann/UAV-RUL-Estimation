"""Whole-UAV isolation, gated budgets and resumability of PE_27."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "2_architecture_experiments" / "1_pipeline_experiments"
sys.path.insert(0, str(PIPELINE))

import run_uav_subset_ensemble as runner  # noqa: E402
from confirmation_utils import EvaluationJob, evaluation_jobs, prediction_records  # noqa: E402
from experiment_config import read_experiment_config  # noqa: E402
from run_experiment_definition import _execution_plan  # noqa: E402
from tabular_data_adapter import TabularDataset  # noqa: E402


@contextmanager
def temporary_directory():
    base = (ROOT / ".tmp").resolve()
    path = base / ("uav_subset_" + uuid4().hex)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        path.resolve().relative_to(base)
        shutil.rmtree(path)


def workflow():
    return read_experiment_config(PIPELINE / "experiments/PE_27/settings.toml")["uav_subset_workflows"]["PE_27"]


def job():
    return EvaluationJob(7, 0, 0, "outer", -1,
        frozenset(f"U{i:02d}" for i in range(80)), frozenset(f"H{i:02d}" for i in range(20)))


def fixture():
    uavs = [f"U{i:02d}" for i in range(12)]
    outer = pd.DataFrame([{"split_seed": seed, "outer_fold": i // 4, "uav_id": uav}
                          for seed in (7, 17, 27) for i, uav in enumerate(uavs)])
    inner = pd.DataFrame([{"split_seed": seed, "outer_fold": fold, "uav_id": uav, "inner_fold": i % 2}
        for seed in (7, 17, 27) for fold in range(3)
        for i, uav in enumerate(outer.loc[outer.split_seed.eq(seed) & outer.outer_fold.ne(fold), "uav_id"])])
    data = TabularDataset(pd.DataFrame({"sensor": np.arange(24, dtype=float)}),
        pd.DataFrame({"uav_id": np.repeat(uavs, 2), "cutoff": np.tile([50., 150.], 12), "scenario": ["s1"]*24}),
        pd.Series(np.tile([60., 15.], 12)), pd.Series([.5]*24))
    return data, outer, inner


class UAVSubsetTests(unittest.TestCase):
    def test_catalog_and_execution_plan(self):
        config = read_experiment_config(PIPELINE / "pipeline_experiments.toml")
        self.assertIn("PE_27", config["uav_subset_workflows"])
        _, _, steps = _execution_plan(PIPELINE / "experiments/PE_27/settings.toml", "PE_27")
        self.assertEqual([s["name"] for s in steps], ["validate_inputs", "evaluate_subset_ensemble"])
        runner.validate_workflow(workflow())

    def test_members_are_reproducible_distinct_whole_uav_subsets(self):
        current = job()
        kwargs = dict(fraction=.8, seed=123, minimum_uavs=4)
        four = runner.subset_members(current, count=4, **kwargs)
        eight = runner.subset_members(current, count=8, **kwargs)
        self.assertEqual(four, eight[:4])
        self.assertEqual(eight, runner.subset_members(current, count=8, **kwargs))
        self.assertEqual(len(set(eight)), 8)
        for subset in eight:
            self.assertEqual(len(subset), 64)
            self.assertTrue(subset <= current.training_uavs)
            self.assertFalse(subset & current.validation_uavs)
        self.assertNotEqual(eight, runner.subset_members(current, count=8, fraction=.8, seed=124, minimum_uavs=4))

    def test_subset_rejects_overlap_and_insufficient_groups(self):
        overlapping = replace(job(), validation_uavs=frozenset({"U00"}))
        with self.assertRaisesRegex(ValueError, "overlap"):
            runner.subset_members(overlapping, count=8, fraction=.8, seed=1, minimum_uavs=4)
        with self.assertRaisesRegex(ValueError, "Too few"):
            runner.subset_members(job(), count=8, fraction=.01, seed=1, minimum_uavs=4)

    def test_workflow_requires_separate_confirmation_and_fixed_member_counts(self):
        for change in ({"confirmation_split_seeds": [20261207]}, {"subset_fraction": 1.},
                       {"subset_fraction": float("nan")}, {"screen_members": 3},
                       {"confirmation_minimum_fold_wins": 11}):
            with self.assertRaises(ValueError):
                runner.validate_workflow({**workflow(), **change})

    def test_failed_four_member_screen_never_expands_or_confirms(self):
        fits, reports = [], []
        def report(name, jobs, count, confirmation):
            reports.append(name)
            return {"status": "no_promotion", "promoted": False}
        with temporary_directory() as root:
            (root / "reporting").mkdir()
            result = runner.run_stages({}, root, ["screen"], ["confirmation"],
                lambda jobs, count: fits.append((jobs, count)), report)
            self.assertEqual(fits, [(["screen"], 4)])
            self.assertEqual(reports, ["screen_4"])
            self.assertFalse(result["confirmation_completed"])
            self.assertEqual(result["skipped_stages"], ["screen_8", "confirmation_8"])
            skipped = json.loads((root / "stages/confirmation_8/reporting/winner_manifest.json").read_text())
            self.assertFalse(skipped["promoted"])

    def test_failed_eight_member_screen_never_reads_confirmation(self):
        fits = []
        def report(name, jobs, count, confirmation):
            return {"status": "screening_candidate" if name == "screen_4" else "no_promotion", "promoted": False}
        with temporary_directory() as root:
            (root / "reporting").mkdir()
            result = runner.run_stages({}, root, ["screen"], ["confirmation"],
                lambda jobs, count: fits.append((jobs, count)), report)
        self.assertEqual(fits, [(["screen"], 4), (["screen"], 8)])
        self.assertEqual(result["winner"], "control")

    def test_only_confirmation_can_promote_and_reports_its_rows(self):
        seen = []
        def report(name, jobs, count, confirmation):
            seen.append((name, jobs, count, confirmation))
            return {"status": "promoted" if confirmation else "screening_candidate",
                    "promoted": confirmation, "winner": "uav_subset_mean_8", "folds": len(jobs)}
        with temporary_directory() as root:
            (root / "reporting").mkdir()
            result = runner.run_stages({}, root, ["screen"], ["confirm1", "confirm2"], lambda *args: None, report)
        self.assertTrue(result["promoted"])
        self.assertEqual(result["folds"], 2)
        self.assertEqual(seen[-1], ("confirmation_8", ["confirm1", "confirm2"], 8, True))

    def test_uniform_average_excludes_control_and_ignores_labels(self):
        data, outer, inner = fixture()
        current = evaluation_jobs(outer, inner, include_inner=False)[0]
        held = runner.select_uavs(data, current.validation_uavs)
        methods = {"control": np.full(len(held), 100.), **{runner.member_name(i): np.full(len(held), i+1.) for i in range(8)}}
        records = pd.DataFrame(prediction_records(current, held, methods))
        _, predictions = runner.average_members(records, [current], 4)
        np.testing.assert_array_equal(predictions["uav_subset_mean_4"], np.full(len(held), 2.5))
        records["observed_rul"] += 1000.
        _, predictions = runner.average_members(records, [current], 8)
        np.testing.assert_array_equal(predictions["uav_subset_mean_8"], np.full(len(held), 4.5))
        with self.assertRaises(ValueError):
            runner.average_members(records.loc[~records.method.eq("subset_07")], [current], 8)

    def test_calibration_rejects_uavs_outside_member_training(self):
        data, _, _ = fixture()
        selected = runner.select_uavs(data, {"U00", "U01"})
        class Model:
            def _calibration_data(self, training):
                return data
        with self.assertRaisesRegex(ValueError, "exactly its training"):
            runner.checked_calibration(Model(), selected)

    def test_checkpoint_cannot_reuse_different_training_membership(self):
        data, outer, inner = fixture()
        current = evaluation_jobs(outer, inner, include_inner=False)[0]
        held = runner.select_uavs(data, current.validation_uavs)
        training = runner.select_uavs(data, current.training_uavs)
        records = pd.DataFrame(prediction_records(current, held, {"control": np.full(len(held), 30.)}))
        records["training_uav_sha256"] = runner.uav_hash(current.training_uavs)
        records["calibration_uav_sha256"] = runner.uav_hash(current.training_uavs)
        records["training_uavs"] = len(current.training_uavs)
        records["training_rows"] = len(training)
        self.assertTrue(runner.checked_cell(records, current, "control", held, training))
        records["calibration_uav_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "membership"):
            runner.checked_cell(records, current, "control", held, training)

    def test_full_workflow_budget_member_isolation_and_zero_refits_on_resume(self):
        data, outer, inner = fixture()
        config = {**workflow(), "screen_split_seed": 7, "confirmation_split_seeds": [17, 27],
                  "outer_fold_count": 3, "inner_fold_count": 2, "screen_minimum_fold_wins": 2,
                  "confirmation_minimum_fold_wins": 4}
        fits = []
        report_calls = []
        class Model:
            internal_folds = 2
            def _calibration_data(self, training):
                return training
            def fit(self, training, validation):
                assert validation is None
                assert training.metadata.groupby("uav_id").size().eq(2).all()
                np.testing.assert_allclose(training.sample_weights.groupby(training.metadata.uav_id).sum(), 1.)
                self.training_uavs = set(training.metadata.uav_id)
                fits.append(self.training_uavs)
            def predict(self, held):
                assert not self.training_uavs & set(held.metadata.uav_id)
                return np.full(len(held), 30.)
        class Adapter:
            def __init__(self, path):
                pass
            def load_training(self, features):
                return data
            def load_development(self, features):
                return data
            def _copied_path(self, name):
                return ROOT / config["tabular_manifest"]
        def report(rows, methods, **kwargs):
            confirmation = kwargs["promotion_allowed"]
            report_calls.append((set(rows.outer_fold), confirmation))
            return {"status": "promoted" if confirmation else "screening_candidate", "promoted": confirmation,
                    "winner": "uav_subset_mean_8"}
        with temporary_directory() as root, patch.object(runner, "TabularDataAdapter", Adapter), \
                patch.object(runner, "generated_partitions", return_value=(outer, inner)), \
                patch.object(runner, "new_model", side_effect=lambda config: Model()), \
                patch.object(runner, "method_report", side_effect=report), patch("sys.stdout", new_callable=io.StringIO):
            result = runner.run(config, root)
            self.assertEqual(len(fits), 81)  # 15 screen + 12 expansion + 54 confirmation.
            self.assertEqual(sum(len(uavs) == 8 for uavs in fits), 9)
            self.assertEqual(sum(len(uavs) == 6 for uavs in fits), 72)
            self.assertEqual(result["completed_model_fits"], 81)
            self.assertTrue(result["promoted"])
            self.assertEqual(report_calls, [(set(range(3)), False), (set(range(3)), False), (set(range(3, 9)), True)])
            runner.run(config, root)
            self.assertEqual(len(fits), 81)
            checkpoint = pd.read_csv(root / "reporting/fold_predictions.csv")
            checkpoint.loc[0, "training_uav_sha256"] = "wrong"
            checkpoint.to_csv(root / "reporting/fold_predictions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "membership"):
                runner.run(config, root)


if __name__ == "__main__":
    unittest.main()
