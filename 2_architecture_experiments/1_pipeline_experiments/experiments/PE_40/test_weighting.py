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

SPEC = importlib.util.spec_from_file_location("pe40_runner", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

ARMS = ["baseline", "similarity_uav_equal", "uav_equal_only", "unweighted"]


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


class WeightTests(unittest.TestCase):
    def setUp(self):
        # Two UAVs of very different length, the situation the study is about.
        self.uavs = np.array(["A"] * 3 + ["B"] * 9)
        self.similarity = np.array([1.0, 2.0, 3.0] + [0.5] * 9)

    def test_uav_totals_are_equalised_regardless_of_length(self):
        w = runner.equalise_by_uav(self.similarity, self.uavs)
        totals = pd.Series(w).groupby(pd.Series(self.uavs)).sum()
        self.assertAlmostEqual(totals["A"], totals["B"])
        self.assertAlmostEqual(w.mean(), 1.0)

    def test_within_uav_profile_is_preserved(self):
        w = runner.equalise_by_uav(self.similarity, self.uavs)
        before = self.similarity[:3] / self.similarity[:3].sum()
        after = w[:3] / w[:3].sum()
        np.testing.assert_allclose(before, after, rtol=1e-12)

    def test_already_balanced_input_is_left_alone_up_to_scale(self):
        uavs = np.array(["A"] * 4 + ["B"] * 4)
        w = runner.equalise_by_uav(np.ones(8), uavs)
        np.testing.assert_allclose(w, np.ones(8), rtol=1e-12)

    def test_the_four_arms_do_what_they_claim(self):
        recipes = runner.ARMS
        weights = {arm: runner.arm_weights(recipes[arm], self.similarity, self.uavs) for arm in ARMS}
        for arm, w in weights.items():
            self.assertAlmostEqual(w.mean(), 1.0, msg=arm)
            self.assertTrue((w > 0).all(), arm)
        np.testing.assert_allclose(weights["unweighted"], np.ones(len(self.uavs)), rtol=1e-12)
        np.testing.assert_allclose(weights["baseline"], self.similarity / self.similarity.mean(), rtol=1e-12)
        for arm in ("similarity_uav_equal", "uav_equal_only"):
            totals = pd.Series(weights[arm]).groupby(pd.Series(self.uavs)).sum()
            self.assertAlmostEqual(totals.max() / totals.min(), 1.0, msg=arm)
        # Only the uav_equal factor may change UAV totals; only the similarity
        # factor may change the profile inside a UAV.
        for a, b in (("baseline", "unweighted"), ("similarity_uav_equal", "uav_equal_only")):
            self.assertFalse(np.allclose(weights[a][:3], weights[b][:3]))

    def test_longer_uav_loses_influence_under_normalisation(self):
        plain = runner.arm_weights(runner.ARMS["unweighted"], self.similarity, self.uavs)
        equal = runner.arm_weights(runner.ARMS["uav_equal_only"], self.similarity, self.uavs)
        share = lambda w: pd.Series(w).groupby(pd.Series(self.uavs)).sum()["B"] / w.sum()
        self.assertGreater(share(plain), 0.7)
        self.assertAlmostEqual(share(equal), 0.5)

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
    def test_interaction_is_the_difference_of_the_two_simple_effects(self):
        summary = pd.DataFrame({"arm": ARMS, "rmse": [10.0, 9.6, 9.9, 10.1]})
        boot = {arm: np.full(500, v) for arm, v in zip(ARMS, [10.0, 9.6, 9.9, 10.1])}
        result = runner.interaction(boot, summary)
        self.assertAlmostEqual(result["uav_equal_effect_with_similarity"], -0.4)
        self.assertAlmostEqual(result["uav_equal_effect_without_similarity"], -0.2)
        self.assertAlmostEqual(result["interaction"], -0.2)
        self.assertEqual(result["interpretation"], "factors_interact")

    def test_parallel_effects_report_no_interaction(self):
        summary = pd.DataFrame({"arm": ARMS, "rmse": [10.0, 9.8, 9.9, 10.1]})
        boot = {arm: np.full(500, v) for arm, v in zip(ARMS, [10.0, 9.8, 9.9, 10.1])}
        result = runner.interaction(boot, summary)
        self.assertAlmostEqual(result["interaction"], 0.0)
        self.assertEqual(result["interpretation"], "inconclusive")


class ReportTests(unittest.TestCase):
    def test_paired_report_direction_and_uav_bootstrap(self):
        config = {"outer_folds": 5, "bootstrap_repetitions": 1000, "bootstrap_seed": 35,
                  "secondary_contrasts": [["uav_equal_only", "unweighted"]]}
        errors = {"baseline": 2.0, "similarity_uav_equal": 1.0, "uav_equal_only": 2.0, "unweighted": 3.0}
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
                runner.report(output, config, runner.ARMS)
            paired = pd.read_csv(output / "reporting/paired_weighting.csv").set_index("arm")
            self.assertAlmostEqual(paired.loc["similarity_uav_equal", "rmse_change"], -1.0)
            self.assertEqual(paired.loc["similarity_uav_equal", "interpretation"], "weighting_candidate")
            self.assertEqual(paired.loc["similarity_uav_equal", "improved_folds"], 5)
            self.assertEqual(paired.loc["unweighted", "interpretation"], "weighting_harmful")
            self.assertEqual(paired.loc["uav_equal_only", "interpretation"], "inconclusive")
            self.assertTrue((paired.comparisons_adjusted == 3).all())
            cross = json.loads((output / "reporting/interaction.json").read_text())
            self.assertAlmostEqual(cross["uav_equal_effect_with_similarity"], -1.0)
            self.assertAlmostEqual(cross["uav_equal_effect_without_similarity"], -1.0)
            self.assertAlmostEqual(cross["interaction"], 0.0)
            summary = pd.read_csv(output / "reporting/summary.csv")
            self.assertEqual(list(summary.arm), ARMS)
            damaged = output / "cells/baseline__fold_0.csv"
            runner.write_csv(damaged, pd.read_csv(damaged).iloc[1:])
            with self.assertRaises(AssertionError), contextlib.redirect_stdout(io.StringIO()):
                runner.report(output, config, runner.ARMS)


if __name__ == "__main__":
    unittest.main()
