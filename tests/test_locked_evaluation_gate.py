"""Tests for locked-evaluation compatibility and hyperparameter gating."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GATE_DIR = (
    REPOSITORY_ROOT
    / "2_architecture_experiments"
    / "2_model_architecture_study"
    / "6_locked_outer_evaluation"
)
if str(GATE_DIR) not in sys.path:
    sys.path.insert(0, str(GATE_DIR))
MODULE_SPEC = importlib.util.spec_from_file_location(
    "locked_evaluation_gate_test_module",
    GATE_DIR / "evaluation_gate.py",
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError("Cannot load locked evaluation gate")
evaluation_gate = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = evaluation_gate
MODULE_SPEC.loader.exec_module(evaluation_gate)


class LockedEvaluationGateCompatibilityTests(unittest.TestCase):
    def test_restores_only_fixed_noop_transform_settings(self) -> None:
        search = {
            "max_depth": {"kind": "fixed", "value": 4},
            "fault_mode_strategy": {"kind": "fixed", "value": "none"},
            "signal_compression_strategy": {
                "kind": "fixed",
                "value": "none",
            },
        }

        restored = evaluation_gate._restore_legacy_noop_hyperparameters(
            {"max_depth": 4},
            search,
        )

        self.assertEqual(
            restored,
            {
                "max_depth": 4,
                "fault_mode_strategy": "none",
                "signal_compression_strategy": "none",
            },
        )

    def test_does_not_restore_an_active_transform(self) -> None:
        search = {
            "fault_mode_strategy": {"kind": "fixed", "value": "indicator"},
        }

        restored = evaluation_gate._restore_legacy_noop_hyperparameters({}, search)

        self.assertEqual(restored, {})
        with self.assertRaisesRegex(
            evaluation_gate.LockedEvaluationGateError,
            "hyperparameter names differ",
        ):
            evaluation_gate._validate_hyperparameter_values(
                "xgboost",
                restored,
                search,
            )

    def test_does_not_restore_missing_tuned_parameters(self) -> None:
        search = {
            "max_depth": {"kind": "categorical", "values": [2, 4, 8]},
            "fault_mode_strategy": {"kind": "fixed", "value": "none"},
        }

        restored = evaluation_gate._restore_legacy_noop_hyperparameters({}, search)

        self.assertEqual(restored, {"fault_mode_strategy": "none"})
        with self.assertRaisesRegex(
            evaluation_gate.LockedEvaluationGateError,
            "hyperparameter names differ",
        ):
            evaluation_gate._validate_hyperparameter_values(
                "xgboost",
                restored,
                search,
            )


if __name__ == "__main__":
    unittest.main()
