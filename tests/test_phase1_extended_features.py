from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PHASE_1_ROOT = REPOSITORY_ROOT / "1_dataset_construction"
DIAGNOSTIC_ROOT = (
    REPOSITORY_ROOT / "0_data_analysis" / "model_guided_feature_analysis"
)
PHASE_2_SETTINGS_ROOT = (
    REPOSITORY_ROOT
    / "2_model_architecture_study"
    / "1_architecture_study_settings"
)
for path in (
    PHASE_1_ROOT,
    PHASE_1_ROOT / "4_training_prefixes",
    PHASE_1_ROOT / "5_prefix_feature_engineering",
    DIAGNOSTIC_ROOT,
    PHASE_2_SETTINGS_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_prefix_features import extract_prefix_features
from common import MODEL_CHANNELS
from create_training_prefixes import make_training_prefixes
from feature_recipes import feature_catalog
from phase_1_config import load_phase_one_profile
from run_feature_diagnostics import feature_block
import verify_architecture_study_settings as phase2_settings_verifier
from verify_architecture_study_settings import (
    SettingsError,
    _verify_training_prefix_counts,
)


def synthetic_history(uav_id: str, lifetime: int) -> pd.DataFrame:
    cycles = np.arange(1, lifetime + 1)
    values: dict[str, object] = {
        "uav_id": uav_id,
        "flight_cycle": cycles,
        "RUL": lifetime - cycles,
    }
    for number in range(1, 29):
        values[f"telemetry_{number:02d}"] = number + cycles * (number / 100.0)
    return pd.DataFrame(values)


class PhaseOneExtendedFeatureTests(unittest.TestCase):
    def test_phase2_accepts_bounded_prefix_counts_and_rejects_wrong_exact_count(
        self,
    ) -> None:
        prefix_rows = pd.DataFrame(
            {"uav_id": ["UAV_A"] * 39 + ["UAV_B"] * 40}
        )
        artifact = SimpleNamespace(path="unused_training_prefixes.csv")
        bounded = SimpleNamespace(
            phase_1=SimpleNamespace(
                artifacts={"training_prefixes": artifact},
                expected_training_uavs=2,
                expected_prefixes_per_training_uav=None,
                minimum_prefixes_per_training_uav=39,
                maximum_prefixes_per_training_uav=40,
            )
        )
        with patch.object(
            phase2_settings_verifier.pd,
            "read_csv",
            return_value=prefix_rows,
        ):
            _verify_training_prefix_counts(bounded)

            bounded.phase_1.expected_prefixes_per_training_uav = 40
            bounded.phase_1.minimum_prefixes_per_training_uav = None
            bounded.phase_1.maximum_prefixes_per_training_uav = None
            with self.assertRaises(SettingsError):
                _verify_training_prefix_counts(bounded)

    def test_feature_blocks_distinguish_legacy_and_acceleration_differences(
        self,
    ) -> None:
        self.assertEqual(
            feature_block("feature__telemetry_07__w5_last_minus_mean"),
            "window_5",
        )
        self.assertEqual(
            feature_block("feature__telemetry_07__w5_minus_w20_slope"),
            "acceleration",
        )

    def test_legacy_and_extended_profiles_preserve_expected_counts(self) -> None:
        history = synthetic_history("UAV_A", 80)
        legacy = extract_prefix_features(history, 60)
        extended = extract_prefix_features(
            history,
            60,
            feature_profile="extended",
        )

        self.assertEqual(len(MODEL_CHANNELS), 22)
        self.assertEqual(len(legacy), 606)
        self.assertEqual(len(extended), 1288)
        self.assertTrue(set(legacy).issubset(extended))

    def test_extended_catalog_declares_control_and_candidate_recipes(self) -> None:
        history = synthetic_history("UAV_A", 80)
        features = extract_prefix_features(
            history,
            60,
            feature_profile="extended",
        )
        profile = load_phase_one_profile("extended_features")
        catalog = feature_catalog(
            list(features),
            feature_profile=profile.feature_profile,
            feature_sets=profile.feature_sets,
        )

        counts = {name: int(catalog[name].sum()) for name in profile.feature_sets}
        self.assertEqual(
            counts,
            {
                "age_only": 2,
                "last_values": 24,
                "screened": 310,
                "all_nonconstant": 606,
                "screened_v1": 310,
                "screened_robust": 558,
                "screened_acceleration": 400,
                "screened_compact": 256,
                "all_generated_v2": 1288,
            },
        )
        self.assertTrue(catalog["screened"].equals(catalog["screened_v1"]))

    def test_drift_ablation_prunes_and_replaces_unstable_features(self) -> None:
        history = synthetic_history("UAV_A", 80)
        features = extract_prefix_features(
            history,
            60,
            feature_profile="extended",
        )
        profile = load_phase_one_profile("drift_ablation_features")
        catalog = feature_catalog(
            list(features),
            feature_profile=profile.feature_profile,
            feature_sets=profile.feature_sets,
        ).set_index("feature_name")

        counts = {name: int(catalog[name].sum()) for name in profile.feature_sets}
        self.assertEqual(
            counts,
            {
                "age_only": 2,
                "screened_v1": 310,
                "screened_drift_pruned": 298,
                "screened_drift_replaced": 342,
            },
        )
        for channel in ("telemetry_15", "telemetry_16"):
            unstable = [
                f"feature__{channel}__history_sd",
                f"feature__{channel}__history_min",
                f"feature__{channel}__history_max",
                f"feature__{channel}__mean_abs_delta",
                f"feature__{channel}__max_abs_delta",
                f"feature__{channel}__w50_sd",
            ]
            self.assertTrue(catalog.loc[unstable, "screened_v1"].all())
            self.assertFalse(
                catalog.loc[unstable, "screened_drift_pruned"].any()
            )
            self.assertFalse(
                catalog.loc[unstable, "screened_drift_replaced"].any()
            )
            robust = [
                f"feature__{channel}__history_median",
                f"feature__{channel}__history_iqr",
                f"feature__{channel}__history_q10",
                f"feature__{channel}__history_q90",
                f"feature__{channel}__median_abs_delta",
                f"feature__{channel}__w50_median",
                f"feature__{channel}__w50_iqr",
                f"feature__{channel}__w50_median_abs_delta",
            ]
            self.assertTrue(
                catalog.loc[robust, "screened_drift_replaced"].all()
            )

    def test_stratified_policy_caps_only_when_unique_cutoffs_require_it(self) -> None:
        train = pd.concat(
            [
                synthetic_history("UAV_A", 44),
                synthetic_history("UAV_B", 70),
            ],
            ignore_index=True,
        )
        histories = pd.DataFrame(
            {
                "uav_id": ["UAV_A", "UAV_B"],
                "final_cycle": [44, 70],
            }
        )
        test_lengths = np.arange(5, 65, dtype=int)
        prefixes = make_training_prefixes(
            train,
            histories,
            test_lengths,
            cutoffs_per_uav=40,
            seed=19,
            strategy="stratified_empirical",
        )

        counts = prefixes.groupby("uav_id").size()
        self.assertEqual(counts.to_dict(), {"UAV_A": 39, "UAV_B": 40})
        self.assertFalse(prefixes.duplicated(["uav_id", "cutoff"]).any())
        self.assertTrue(
            np.allclose(
                prefixes.groupby("uav_id")["sample_weight"].sum().to_numpy(),
                1.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
