"""Regression tests for Phase 2 architecture-comparison metric metadata."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DIR = (
    REPOSITORY_ROOT
    / "2_model_architecture_study"
    / "7_architecture_comparison"
)
if str(COMPARISON_DIR) not in sys.path:
    sys.path.insert(0, str(COMPARISON_DIR))

from architecture_comparison import (  # noqa: E402
    METRIC_DIFFERENCE_INTERPRETATIONS,
    METRICS,
)


class ArchitectureComparisonMetricTests(unittest.TestCase):
    def test_every_reported_metric_has_a_paired_difference_interpretation(self) -> None:
        self.assertEqual(
            set(METRIC_DIFFERENCE_INTERPRETATIONS),
            set(METRICS),
        )


if __name__ == "__main__":
    unittest.main()
