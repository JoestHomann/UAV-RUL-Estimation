"""Tests for Phase 3 final-search metadata handling."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIR = (
    REPOSITORY_ROOT
    / "3_final_model_training_and_inference"
    / "2_final_configuration_search"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "phase3_final_search_test_module",
    SEARCH_DIR / "final_configuration_search.py",
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError("Cannot load Phase 3 final configuration search")
final_search = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = final_search
MODULE_SPEC.loader.exec_module(final_search)


class Phase3ScenarioLabelTests(unittest.TestCase):
    def test_preserves_named_scenario(self) -> None:
        self.assertEqual(
            final_search._scenario_label("development_01"),
            "development_01",
        )

    def test_normalizes_legacy_numeric_scenario_to_text(self) -> None:
        self.assertEqual(final_search._scenario_label(np.int64(3)), "3")

    def test_rejects_missing_scenario(self) -> None:
        with self.assertRaisesRegex(
            final_search.FinalConfigurationSearchError,
            "missing scenario",
        ):
            final_search._scenario_label(np.nan)


if __name__ == "__main__":
    unittest.main()
