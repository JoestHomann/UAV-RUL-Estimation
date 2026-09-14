"""Scientific-contract checks; run with the repository Python interpreter."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("pe39_runner", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

CHANNELS = ["telemetry_05", "telemetry_24"]
ARMS = ["baseline", "drop_05", "drop_24", "drop_05_24"]


@contextlib.contextmanager
def test_workspace():
    parent = (runner.HERE / "runs").resolve()
    output = parent / ("test_" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    try:
        yield output
    finally:
        assert output.resolve().parent == parent and output.name.startswith("test_")
        shutil.rmtree(output)


def synthetic_cells(output, errors, folds=5, seeds=2):
    for arm, error in errors.items():
        for fold in range(folds):
            rows = []
            for uid in range(fold * 20, (fold + 1) * 20):
                for seed in range(seeds):
                    target = float(uid + seed + 1)
                    rows.append(dict(uav_id=uid, evaluation_seed=seed, flight_cycle=50 + seed, fold=fold,
                                     target_capped=target, target_raw=target, prediction=target + error))
            runner.write_csv(output / "cells" / f"{arm}__fold_{fold}.csv", pd.DataFrame(rows))


class ArmTests(unittest.TestCase):
    def test_removing_a_channel_does_not_touch_longer_names(self):
        columns = ["flight_cycle", "telemetry_05", "telemetry_05__hist_mean", "telemetry_050", "telemetry_24__hist_std"]
        self.assertEqual(runner.ablation_columns(columns, ["telemetry_05"]),
                         ["flight_cycle", "telemetry_050", "telemetry_24__hist_std"])

    def test_arms_are_baseline_singles_then_the_pair(self):
        self.assertEqual(list(runner.build_arms(CHANNELS)), ARMS)
        self.assertEqual(runner.build_arms(CHANNELS)["drop_05_24"], CHANNELS)
        self.assertEqual(runner.arm_name(["telemetry_24"]), "drop_24")

    def test_pair_removes_exactly_the_union_of_the_singles(self):
        columns = ["flight_cycle", "telemetry_05", "telemetry_05__hist_mean", "telemetry_05__hist_std",
                   "telemetry_24", "telemetry_24__hist_mean", "telemetry_24__hist_std", "telemetry_13"]
        catalogs = {name: runner.ablation_columns(columns, removed) for name, removed in runner.build_arms(CHANNELS).items()}
        self.assertEqual(len(catalogs["baseline"]), 8)
        self.assertEqual(len(catalogs["drop_05"]), 5)
        self.assertEqual(len(catalogs["drop_24"]), 5)
        self.assertEqual(catalogs["drop_05_24"], ["flight_cycle", "telemetry_13"])

    def test_scoring_uavs_never_fit_or_stop(self):
        data = pd.DataFrame({"uav_id": np.repeat(np.arange(100), 3)})
        config = {"outer_folds": 5, "stopping_seed": 0, "stopping_uav_fraction": 0.1}
        scored = []
        for _, fit, stop, held in runner.partitions(data, config):
            fit_ids, stop_ids, held_ids = [set(data.iloc[idx].uav_id) for idx in (fit, stop, held)]
            self.assertEqual((len(fit_ids), len(stop_ids), len(held_ids)), (72, 8, 20))
            self.assertFalse(fit_ids & stop_ids or fit_ids & held_ids or stop_ids & held_ids)
            scored.extend(held_ids)
        self.assertEqual(sorted(scored), list(range(100)))


class AdditivityTests(unittest.TestCase):
    def setUp(self):
        self.summary = pd.DataFrame({"arm": ARMS, "rmse": [10.0, 10.2, 10.3, 10.6]})

    def test_interaction_is_the_pair_minus_both_singles(self):
        boot = {arm: np.full(500, value) for arm, value in zip(ARMS, [10.0, 10.2, 10.3, 10.6])}
        result = runner.additivity(boot, self.summary, "drop_05_24", ["drop_05", "drop_24"])
        self.assertAlmostEqual(result["pair_change"], 0.6)
        self.assertAlmostEqual(result["sum_of_single_changes"], 0.5)
        self.assertAlmostEqual(result["interaction"], 0.1)
        self.assertAlmostEqual(result["interaction_ci95_low"], 0.1)
        self.assertEqual(result["interpretation"], "worse_than_additive")

    def test_exactly_additive_pair_reports_zero_interaction(self):
        summary = pd.DataFrame({"arm": ARMS, "rmse": [10.0, 10.2, 10.3, 10.5]})
        boot = {arm: np.full(500, value) for arm, value in zip(ARMS, [10.0, 10.2, 10.3, 10.5])}
        result = runner.additivity(boot, summary, "drop_05_24", ["drop_05", "drop_24"])
        self.assertAlmostEqual(result["interaction"], 0.0)
        self.assertEqual(result["interpretation"], "inconclusive")

    def test_interaction_interval_widens_with_bootstrap_spread(self):
        rng = np.random.default_rng(39)
        boot = {arm: np.full(20000, value) for arm, value in zip(ARMS, [10.0, 10.2, 10.3, 10.6])}
        boot["drop_05_24"] = boot["drop_05_24"] + rng.normal(0, 0.05, 20000)
        result = runner.additivity(boot, self.summary, "drop_05_24", ["drop_05", "drop_24"])
        self.assertLess(result["interaction_ci95_low"], 0.1)
        self.assertGreater(result["interaction_ci95_high"], 0.1)


class ReportTests(unittest.TestCase):
    def test_paired_report_direction_and_uav_bootstrap(self):
        catalogs = {"baseline": ["a", "b", "c"], "drop_05": ["a", "b"], "drop_24": ["a", "b"], "drop_05_24": ["a"]}
        config = {"outer_folds": 5, "bootstrap_repetitions": 1000, "bootstrap_seed": 35, "channels": CHANNELS,
                  "secondary_contrasts": [["drop_05_24", "drop_05"]], "reproduction_check": ""}
        with test_workspace() as output:
            synthetic_cells(output, {"baseline": 2.0, "drop_05": 1.0, "drop_24": 3.0, "drop_05_24": 2.0})
            with contextlib.redirect_stdout(io.StringIO()):
                runner.report(output, config, catalogs)
            paired = pd.read_csv(output / "reporting/paired_ablation.csv").set_index("arm")
            self.assertAlmostEqual(paired.loc["drop_05", "rmse_change"], -1.0)
            self.assertEqual(paired.loc["drop_05", "interpretation"], "removal_candidate")
            self.assertEqual(paired.loc["drop_05", "improved_folds"], 5)
            self.assertEqual(paired.loc["drop_24", "interpretation"], "removal_harmful")
            self.assertAlmostEqual(paired.loc["drop_05_24", "rmse_change"], 0.0)
            self.assertTrue((paired.comparisons_adjusted == 3).all())
            interaction = json.loads((output / "reporting/additivity.json").read_text())
            self.assertAlmostEqual(interaction["sum_of_single_changes"], 0.0)
            self.assertAlmostEqual(interaction["interaction"], 0.0)
            secondary = pd.read_csv(output / "reporting/secondary_contrasts.csv").iloc[0]
            self.assertAlmostEqual(secondary.rmse_change, 1.0)
            self.assertIn("Is the pair more than the sum of its parts?",
                          (output / "reporting/report.md").read_text(encoding="utf-8"))
            damaged = output / "cells/drop_05__fold_0.csv"
            runner.write_csv(damaged, pd.read_csv(damaged).iloc[1:])
            with self.assertRaises(AssertionError), contextlib.redirect_stdout(io.StringIO()):
                runner.report(output, config, catalogs)


class ReproductionCheckTests(unittest.TestCase):
    REGISTRATION = {"hashes": {"source": "a", "feature_contract": "b", "train": "c", "test_cutoffs": "d"},
                    "versions": {"xgboost": "2.1.0"}, "xgboost_params": {"max_depth": 5},
                    "early_stopping_rounds": 100, "target_cap": 125}

    def _reference(self, output, errors):
        reference = output / "reference"
        (reference).mkdir(parents=True, exist_ok=True)
        (reference / "registration.json").write_text(json.dumps(self.REGISTRATION))
        synthetic_cells(reference, errors)
        return reference

    def test_missing_reference_run_is_skipped_not_failed(self):
        config = {"outer_folds": 5, "reproduction_check": "does/not/exist"}
        with test_workspace() as output:
            note = runner.reproduction_note(output, config, {"baseline": []}, self.REGISTRATION)
            self.assertEqual(note["status"], "skipped")

    def test_identical_refits_are_reported_as_reproduced(self):
        with test_workspace() as output:
            reference = self._reference(output, {"baseline": 2.0})
            synthetic_cells(output, {"baseline": 2.0})
            config = {"outer_folds": 5, "reproduction_check": str(reference.relative_to(runner.ROOT))}
            note = runner.reproduction_note(output, config, {"baseline": []}, self.REGISTRATION)
            self.assertTrue(note["identical"])
            self.assertEqual(note["cells_compared"], 5)
            self.assertEqual(note["max_abs_prediction_difference"], 0.0)

    def test_drift_under_identical_settings_is_raised_not_reported(self):
        with test_workspace() as output:
            reference = self._reference(output, {"baseline": 2.0})
            synthetic_cells(output, {"baseline": 2.5})
            config = {"outer_folds": 5, "reproduction_check": str(reference.relative_to(runner.ROOT))}
            with self.assertRaises(RuntimeError):
                runner.reproduction_note(output, config, {"baseline": []}, self.REGISTRATION)

    def test_drift_under_a_different_library_version_is_reported_not_raised(self):
        with test_workspace() as output:
            reference = self._reference(output, {"baseline": 2.0})
            synthetic_cells(output, {"baseline": 2.5})
            config = {"outer_folds": 5, "reproduction_check": str(reference.relative_to(runner.ROOT))}
            registration = dict(self.REGISTRATION, versions={"xgboost": "9.9.9"})
            note = runner.reproduction_note(output, config, {"baseline": []}, registration)
            self.assertFalse(note["identical"])
            self.assertFalse(note["versions_match"])
            self.assertAlmostEqual(note["max_abs_prediction_difference"], 0.5)


if __name__ == "__main__":
    unittest.main()
