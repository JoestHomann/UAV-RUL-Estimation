"""Scientific-contract checks; run with the repository Python interpreter."""
import contextlib
import importlib.util
import io
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("pe38_runner", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

CHANNEL = "telemetry_07"


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


class StateEncodingTests(unittest.TestCase):
    def test_levels_group_repeated_values_and_ignore_float_noise(self):
        values = [140.5912754437698, 140.5912754437698 + 1e-12, 140.76317284, 140.50532674]
        self.assertEqual(len(runner.derive_levels(values, 1e-06)), 3)
        self.assertEqual(runner.derive_levels(values, 1e-06)[0], 140.50532674)

    def test_unseen_value_maps_to_the_nearest_known_state(self):
        levels = [0.0, 1.0, 2.0]
        np.testing.assert_array_equal(runner.assign_states([-5.0, 0.4, 0.6, 1.9, 7.0], levels), [0, 0, 1, 2, 2])

    def test_state_features_match_hand_computed_causal_values(self):
        sequence = [0, 0, 1, 1, 1, 0, 2]
        frame = pd.DataFrame({"uav_id": ["UAV_0001"] * 7, "flight_cycle": range(1, 8)})
        built = runner.build_state_features(frame, sequence, 4, CHANNEL)
        cycles = np.arange(1, 8, dtype=float)
        expected = {
            "state": [0, 0, 1, 1, 1, 0, 2],
            "state_n_unique": [1, 1, 2, 2, 2, 2, 3],
            "state_n_transitions": [0, 0, 1, 1, 1, 2, 3],
            "state_transition_rate": np.array([0, 0, 1, 1, 1, 2, 3]) / cycles,
            "state_run_length": [1, 2, 1, 2, 3, 1, 1],
            "state_run_fraction": np.array([1, 2, 1, 2, 3, 1, 1]) / cycles,
            "state_max_so_far": [0, 0, 1, 1, 1, 1, 2],
            "state_frac_l0": np.array([1, 2, 2, 2, 2, 3, 3]) / cycles,
            "state_frac_l1": np.array([0, 0, 1, 2, 3, 3, 3]) / cycles,
            "state_frac_l2": np.array([0, 0, 0, 0, 0, 0, 1]) / cycles,
            "state_frac_l3": np.zeros(7),
        }
        self.assertEqual(list(built.columns), runner.state_feature_names(CHANNEL, 4))
        for suffix, values in expected.items():
            np.testing.assert_allclose(built[f"{CHANNEL}__{suffix}"].to_numpy(), values, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(built[[f"{CHANNEL}__state_frac_l{k}" for k in range(4)]].sum(axis=1), np.ones(7))

    def test_state_features_never_read_a_later_cycle(self):
        rng = np.random.default_rng(38)
        frame = pd.DataFrame({"uav_id": np.repeat(["UAV_0001", "UAV_0002"], 40),
                              "flight_cycle": np.tile(np.arange(1, 41), 2)})
        states = rng.integers(0, 4, size=80)
        full = runner.build_state_features(frame, states, 4, CHANNEL)
        for cutoff in (1, 7, 23, 40):
            mask = (frame.uav_id == "UAV_0001") & (frame.flight_cycle <= cutoff)
            prefix = frame[mask].reset_index(drop=True)
            rebuilt = runner.build_state_features(prefix, states[mask.to_numpy()], 4, CHANNEL)
            np.testing.assert_allclose(rebuilt.to_numpy(), full[mask.to_numpy()].to_numpy(), rtol=1e-12, atol=1e-12)

    def test_uavs_are_summarised_independently(self):
        frame = pd.DataFrame({"uav_id": ["UAV_0001"] * 3 + ["UAV_0002"] * 3, "flight_cycle": [1, 2, 3] * 2})
        built = runner.build_state_features(frame, [0, 1, 1, 2, 2, 2], 4, CHANNEL)
        np.testing.assert_array_equal(built[f"{CHANNEL}__state_n_transitions"].to_numpy(), [0, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(built[f"{CHANNEL}__state_run_length"].to_numpy(), [1, 1, 2, 1, 2, 3])


class CatalogTests(unittest.TestCase):
    def test_channel_columns_do_not_collide_with_longer_names(self):
        columns = ["flight_cycle", "telemetry_07", "telemetry_07__hist_mean", "telemetry_070", "telemetry_070__hist_mean"]
        self.assertEqual(runner.channel_columns(columns, "telemetry_07"), ["telemetry_07", "telemetry_07__hist_mean"])

    def test_master_order_keeps_new_columns_beside_the_channel(self):
        contract = ["flight_cycle", "telemetry_07", "telemetry_07__hist_mean", "telemetry_01"]
        order = runner.master_order(contract, "telemetry_07", ["telemetry_07__roll5_mean"], ["telemetry_07__state"])
        self.assertEqual(order, ["flight_cycle", "telemetry_07", "telemetry_07__hist_mean",
                                 "telemetry_07__roll5_mean", "telemetry_07__state", "telemetry_01"])

    def test_every_arm_differs_only_in_the_studied_channel(self):
        catalogs = {arm: list(cols) for arm, cols in ARM_CATALOGS.items()}
        others = {arm: [c for c in cols if not (c == CHANNEL or c.startswith(CHANNEL + "__"))] for arm, cols in catalogs.items()}
        reference = others["baseline"]
        self.assertEqual(len(reference), 146)
        for arm, cols in others.items():
            self.assertEqual(cols, reference, arm)
        for arm, expected in runner.EXPECTED_FEATURES.items():
            self.assertEqual(len(catalogs[arm]), expected, arm)


class ReportTests(unittest.TestCase):
    def test_paired_report_direction_and_uav_bootstrap(self):
        catalogs = {"baseline": ["a", "b"], "medium_plus_state": ["a"], "numeric_strong": ["a"],
                    "state_only": ["a"], "strong_plus_state": ["a"], "drop_07": ["a"]}
        config = {"outer_folds": 5, "bootstrap_repetitions": 1000, "bootstrap_seed": 35, "channel": CHANNEL,
                  "secondary_contrasts": [["medium_plus_state", "drop_07"]]}
        errors = {"baseline": 2.0, "medium_plus_state": 1.0, "numeric_strong": 2.0,
                  "state_only": 2.0, "strong_plus_state": 2.0, "drop_07": 3.0}
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
                runner.report(output, config, catalogs)
            paired = pd.read_csv(output / "reporting/paired_representations.csv").set_index("arm")
            winner = paired.loc["medium_plus_state"]
            self.assertAlmostEqual(winner.rmse_change, -1.0)
            self.assertAlmostEqual(winner.familywise_ci95_high, -1.0)
            self.assertEqual(winner.improved_folds, 5)
            self.assertEqual(winner.interpretation, "representation_candidate")
            self.assertEqual(paired.loc["drop_07", "interpretation"], "representation_harmful")
            self.assertEqual(paired.loc["numeric_strong", "interpretation"], "inconclusive")
            self.assertTrue((paired.comparisons_adjusted == 5).all())
            secondary = pd.read_csv(output / "reporting/secondary_contrasts.csv").iloc[0]
            self.assertEqual(secondary.reference, "drop_07")
            self.assertAlmostEqual(secondary.rmse_change, -2.0)
            self.assertEqual(secondary.comparisons_adjusted, 1)
            damaged = output / "cells/medium_plus_state__fold_0.csv"
            runner.write_csv(damaged, pd.read_csv(damaged).iloc[1:])
            with self.assertRaises(AssertionError), contextlib.redirect_stdout(io.StringIO()):
                runner.report(output, config, catalogs)


def build_arm_catalogs():
    """The real catalogs, built from the shipped contract without touching any model."""
    import json
    import tomllib
    config = tomllib.loads((runner.HERE / "settings.toml").read_text())["study"]
    contract = json.loads((runner.ROOT / config["feature_contract"]).read_text())
    columns = contract["feature_columns"]
    extras = [f"{CHANNEL}__{family}" for family in runner.STRONG_ONLY_FAMILIES]
    state_names = runner.state_feature_names(CHANNEL, config["state_levels"])
    master = runner.master_order(columns, CHANNEL, extras, state_names)
    medium_block = runner.channel_columns(columns, CHANNEL)
    strong_block = medium_block + extras
    keep = {"none": set(), "medium": set(medium_block), "strong": set(strong_block)}
    catalogs = {}
    for arm, recipe in runner.ARMS.items():
        selected = keep[recipe["numeric"]] | (set(state_names) if recipe["state"] else set())
        catalogs[arm] = [c for c in master if c not in set(strong_block) | set(state_names) or c in selected]
    return catalogs


ARM_CATALOGS = build_arm_catalogs()

if __name__ == "__main__":
    unittest.main()
