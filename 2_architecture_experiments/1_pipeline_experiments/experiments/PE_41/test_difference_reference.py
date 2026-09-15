"""Scientific-contract checks; run with the repository Python interpreter."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tomllib
import unittest
import uuid

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("pe41_runner", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

CONFIG = tomllib.loads((Path(__file__).with_name("settings.toml")).read_text())["study"]
BASELINE_SUFFIX = CONFIG["baseline_difference_suffix"]
HISTORY_SUFFIX = CONFIG["history_difference_suffix"]

# A miniature v13: two channels carrying both differences, one carrying only the
# baseline difference -- the medium-tier situation the matched arm exists for.
COLUMNS = [
    "flight_cycle", "flight_cycle_log",
    f"telemetry_13__{BASELINE_SUFFIX}", f"telemetry_13__{HISTORY_SUFFIX}", "telemetry_13__hist_mean",
    f"telemetry_15__{BASELINE_SUFFIX}", f"telemetry_15__{HISTORY_SUFFIX}", "telemetry_15__hist_mean",
    f"telemetry_07__{BASELINE_SUFFIX}", "telemetry_07__hist_mean",
    "telemetry_01__raw", "telemetry_01__hist_std",
]


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


class RemovalTests(unittest.TestCase):
    def setUp(self):
        self.removals = runner.build_removals(COLUMNS, CONFIG)
        self.catalogs = {arm: [c for c in COLUMNS if c not in set(dropped)]
                         for arm, dropped in self.removals.items()}

    def test_arms_are_declared_in_reporting_order(self):
        self.assertEqual(list(self.removals), runner.ARM_ORDER)

    def test_baseline_removes_nothing(self):
        self.assertEqual(self.removals[runner.BASELINE], [])
        self.assertEqual(self.catalogs[runner.BASELINE], COLUMNS)

    def test_each_single_arm_removes_exactly_its_own_family(self):
        self.assertEqual(sorted(self.removals[runner.DROP_BASELINE]),
                         sorted(c for c in COLUMNS if c.endswith("__" + BASELINE_SUFFIX)))
        self.assertEqual(sorted(self.removals[runner.DROP_HISTORY]),
                         sorted(c for c in COLUMNS if c.endswith("__" + HISTORY_SUFFIX)))

    def test_no_other_feature_family_is_touched(self):
        under_test = {BASELINE_SUFFIX, HISTORY_SUFFIX}
        for arm, dropped in self.removals.items():
            for column in dropped:
                self.assertIn(column.split("__")[-1], under_test, f"{arm} removed {column}")

    def test_dropping_both_is_exactly_the_union_of_the_singles(self):
        singles = set(self.removals[runner.DROP_BASELINE]) | set(self.removals[runner.DROP_HISTORY])
        self.assertEqual(set(self.removals[runner.DROP_BOTH]), singles)
        self.assertFalse(set(self.removals[runner.DROP_BASELINE]) & set(self.removals[runner.DROP_HISTORY]))

    def test_matched_arm_adds_only_the_unpaired_baseline_columns(self):
        extra = set(self.removals[runner.DROP_HISTORY_MATCHED]) - set(self.removals[runner.DROP_HISTORY])
        self.assertEqual(extra, {f"telemetry_07__{BASELINE_SUFFIX}"})
        self.assertTrue(set(self.removals[runner.DROP_HISTORY_MATCHED]) <= set(self.removals[runner.DROP_BOTH]))

    def test_head_to_head_arms_have_the_same_feature_count(self):
        self.assertEqual(len(self.catalogs[runner.DROP_HISTORY_MATCHED]),
                         len(self.catalogs[runner.DROP_BASELINE]))
        # ... and each keeps exactly one reference on the fully paired channels.
        kept_matched = set(self.catalogs[runner.DROP_HISTORY_MATCHED])
        kept_baseline = set(self.catalogs[runner.DROP_BASELINE])
        self.assertIn(f"telemetry_13__{BASELINE_SUFFIX}", kept_matched)
        self.assertNotIn(f"telemetry_13__{HISTORY_SUFFIX}", kept_matched)
        self.assertIn(f"telemetry_13__{HISTORY_SUFFIX}", kept_baseline)
        self.assertNotIn(f"telemetry_13__{BASELINE_SUFFIX}", kept_baseline)

    def test_a_channel_without_the_paired_family_cannot_unbalance_the_match(self):
        """If v13 ever gives both families the same reach, the matched arm collapses
        onto the plain history removal and the head-to-head is still balanced."""
        symmetric = [c for c in COLUMNS if not c.startswith("telemetry_07__")]
        removals = runner.build_removals(symmetric, CONFIG)
        self.assertEqual(sorted(removals[runner.DROP_HISTORY_MATCHED]),
                         sorted(removals[runner.DROP_HISTORY]))
        catalogs = {a: [c for c in symmetric if c not in set(d)] for a, d in removals.items()}
        self.assertEqual(len(catalogs[runner.DROP_HISTORY_MATCHED]), len(catalogs[runner.DROP_BASELINE]))

    def test_the_families_under_test_must_be_present(self):
        with self.assertRaises(AssertionError):
            bare = [c for c in COLUMNS if HISTORY_SUFFIX not in c and BASELINE_SUFFIX not in c]
            removals = runner.build_removals(bare, CONFIG)
            assert removals[runner.DROP_HISTORY] and removals[runner.DROP_BASELINE]


class PartitionTests(unittest.TestCase):
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


class InteractionTests(unittest.TestCase):
    def frame(self, values):
        return pd.DataFrame({"arm": list(values), "rmse": list(values.values())})

    def test_complementary_references_report_a_positive_interaction(self):
        # Each family alone costs 0.2; together they cost 0.6, so the second
        # removal costs more than the first predicts.
        values = {runner.BASELINE: 10.0, runner.DROP_HISTORY: 10.2, runner.DROP_BASELINE: 10.2,
                  runner.DROP_HISTORY_MATCHED: 10.2, runner.DROP_BOTH: 10.6}
        boot = {arm: np.full(500, value) for arm, value in values.items()}
        result = runner.additivity(boot, self.frame(values), runner.DROP_BOTH,
                                   [runner.DROP_HISTORY, runner.DROP_BASELINE])
        self.assertAlmostEqual(result["pair_change"], 0.6)
        self.assertAlmostEqual(result["sum_of_single_changes"], 0.4)
        self.assertAlmostEqual(result["interaction"], 0.2)
        self.assertEqual(result["interpretation"], "complementary")

    def test_redundant_references_report_a_negative_interaction(self):
        values = {runner.BASELINE: 10.0, runner.DROP_HISTORY: 10.2, runner.DROP_BASELINE: 10.2,
                  runner.DROP_HISTORY_MATCHED: 10.2, runner.DROP_BOTH: 10.25}
        boot = {arm: np.full(500, value) for arm, value in values.items()}
        result = runner.additivity(boot, self.frame(values), runner.DROP_BOTH,
                                   [runner.DROP_HISTORY, runner.DROP_BASELINE])
        self.assertAlmostEqual(result["interaction"], -0.15)
        self.assertEqual(result["interpretation"], "redundant")

    def test_additive_effects_report_no_interaction(self):
        # Binary-exact values. With a degenerate bootstrap the interval has zero
        # width, so a decimal such as 10.3 - 10.2 - 10.1 leaves ~1e-15 of float
        # residue and the sign test would call that an interaction. Real runs
        # resample 10,000 times and never produce a zero-width interval.
        values = {runner.BASELINE: 10.0, runner.DROP_HISTORY: 10.25, runner.DROP_BASELINE: 10.125,
                  runner.DROP_HISTORY_MATCHED: 10.125, runner.DROP_BOTH: 10.375}
        boot = {arm: np.full(500, value) for arm, value in values.items()}
        result = runner.additivity(boot, self.frame(values), runner.DROP_BOTH,
                                   [runner.DROP_HISTORY, runner.DROP_BASELINE])
        self.assertAlmostEqual(result["interaction"], 0.0)
        self.assertEqual(result["interpretation"], "inconclusive")


class ReportTests(unittest.TestCase):
    def test_paired_report_direction_and_declared_contrasts(self):
        config = {"outer_folds": 5, "bootstrap_repetitions": 1000, "bootstrap_seed": 35,
                  "secondary_contrasts": CONFIG["secondary_contrasts"], "reproduction_check": ""}
        # Keeping the history-mean difference wins the matched head-to-head:
        # drop_baseline_difference (which keeps it) has the smaller error.
        errors = {runner.BASELINE: 1.0, runner.DROP_HISTORY: 2.0, runner.DROP_BASELINE: 1.5,
                  runner.DROP_HISTORY_MATCHED: 2.0, runner.DROP_BOTH: 3.0}
        catalogs = {runner.BASELINE: ["a"] * 153, runner.DROP_HISTORY: ["a"] * 144,
                    runner.DROP_BASELINE: ["a"] * 143, runner.DROP_HISTORY_MATCHED: ["a"] * 143,
                    runner.DROP_BOTH: ["a"] * 134}
        removals = {arm: ["x"] * (153 - len(cols)) for arm, cols in catalogs.items()}
        with test_workspace() as output:
            for arm, error in errors.items():
                for fold in range(5):
                    rows = []
                    for uid in range(fold * 20, (fold + 1) * 20):
                        for seed in range(2):
                            target = float(uid + seed + 1)
                            rows.append(dict(uav_id=uid, evaluation_seed=seed, flight_cycle=50 + seed, fold=fold,
                                             target_capped=target, target_raw=target, prediction=target + error))
                    runner.write_csv(output / "cells" / f"{arm}__fold_{fold}.csv", pd.DataFrame(rows))
            with contextlib.redirect_stdout(io.StringIO()):
                runner.report(output, config, catalogs, removals)
            paired = pd.read_csv(output / "reporting/paired_ablation.csv").set_index("arm")
            self.assertEqual(list(paired.index), runner.PRIMARY_REMOVALS)
            self.assertTrue((paired.comparisons_adjusted == 3).all())
            self.assertAlmostEqual(paired.loc[runner.DROP_BOTH, "rmse_change"], 2.0)
            self.assertEqual(paired.loc[runner.DROP_BOTH, "interpretation"], "removal_harmful")

            secondary = pd.read_csv(output / "reporting/secondary_contrasts.csv")
            head = secondary[(secondary.arm == runner.DROP_BASELINE)
                             & (secondary.reference == runner.DROP_HISTORY_MATCHED)]
            self.assertEqual(len(head), 1)
            # Negative means the arm that keeps the history-mean difference wins.
            self.assertAlmostEqual(float(head.rmse_change.iloc[0]), -0.5)

            summary = pd.read_csv(output / "reporting/summary.csv")
            self.assertEqual(list(summary.arm), runner.ARM_ORDER)
            self.assertEqual(list(summary.features), [153, 144, 143, 143, 134])
            report = (output / "reporting/report.md").read_text(encoding="utf-8")
            self.assertIn("which difference reference should v13 keep", report)

            damaged = output / f"cells/{runner.BASELINE}__fold_0.csv"
            runner.write_csv(damaged, pd.read_csv(damaged).iloc[1:])
            with self.assertRaises(AssertionError), contextlib.redirect_stdout(io.StringIO()):
                runner.report(output, config, catalogs, removals)


if __name__ == "__main__":
    unittest.main()
