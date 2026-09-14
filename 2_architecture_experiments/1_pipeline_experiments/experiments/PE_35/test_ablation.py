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

SPEC = importlib.util.spec_from_file_location("pe35_runner", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


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


class AblationContractTests(unittest.TestCase):
    def test_remove_entire_channel_without_prefix_collisions(self):
        columns = ["flight_cycle", "telemetry_02", "telemetry_02__hist_mean",
                   "telemetry_02__hist_std", "telemetry_020", "telemetry_07__hist_mean"]
        self.assertEqual(runner.ablation_columns(columns, ["telemetry_02"]),
                         ["flight_cycle", "telemetry_020", "telemetry_07__hist_mean"])

    def test_scoring_uavs_never_fit_or_stop(self):
        data = pd.DataFrame({"uav_id": np.repeat(np.arange(100), 3)})
        config = {"outer_folds": 5, "stopping_seed": 0, "stopping_uav_fraction": 0.1}
        partitions = runner.partitions(data, config)
        scored = []
        for _, fit, stop, held in partitions:
            fit_ids, stop_ids, held_ids = [set(data.iloc[idx].uav_id) for idx in (fit, stop, held)]
            self.assertEqual((len(fit_ids), len(stop_ids), len(held_ids)), (72, 8, 20))
            self.assertFalse(fit_ids & stop_ids or fit_ids & held_ids or stop_ids & held_ids)
            scored.extend(held_ids)
        self.assertEqual(sorted(scored), list(range(100)))

    def test_paired_report_direction_and_uav_bootstrap(self):
        catalogs = {"baseline": ["a", "b"], "drop_02": ["a"]}
        config = {"outer_folds": 5, "bootstrap_repetitions": 1000, "bootstrap_seed": 35}
        with test_workspace() as output:
            for arm, error in (("baseline", 2.0), ("drop_02", 1.0)):
                for fold in range(5):
                    rows = []
                    for uid in range(fold * 20, (fold + 1) * 20):
                        for seed in range(2):
                            target = float(uid + seed + 1)
                            rows.append(dict(uav_id=uid, evaluation_seed=seed, flight_cycle=50 + seed,
                                             fold=fold, target_capped=target, target_raw=target,
                                             prediction=target + error))
                    runner.write_csv(output / "cells" / f"{arm}__fold_{fold}.csv", pd.DataFrame(rows))
            with contextlib.redirect_stdout(io.StringIO()):
                runner.report(output, config, catalogs)
            paired = pd.read_csv(output / "reporting/paired_ablation.csv").iloc[0]
            self.assertAlmostEqual(paired.rmse_change, -1.0)
            self.assertAlmostEqual(paired.uav_bootstrap_ci95_high, -1.0)
            self.assertAlmostEqual(paired.familywise_ci95_high, -1.0)
            self.assertEqual(paired.improved_folds, 5)
            self.assertEqual(paired.interpretation, "removal_candidate")
            damaged = output / "cells/drop_02__fold_0.csv"
            frame = pd.read_csv(damaged)
            runner.write_csv(damaged, frame.iloc[1:])
            with self.assertRaises(AssertionError), contextlib.redirect_stdout(io.StringIO()):
                runner.report(output, config, catalogs)


if __name__ == "__main__":
    unittest.main()
