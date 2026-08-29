"""Tests for standalone and experiment-owned Phase 2 run roots."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PHASE_2_ROOT = REPOSITORY_ROOT / "2_model_architecture_study"
if str(PHASE_2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_2_ROOT))
PHASE_3_ROOT = REPOSITORY_ROOT / "3_final_model_training_and_inference"
if str(PHASE_3_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_3_ROOT))

from run_layout import (  # noqa: E402
    RUN_ROOT_ENVIRONMENT_VARIABLE,
    STEP_5_DIRECTORY_NAME,
    STEP_6_DIRECTORY_NAME,
    run_root_for_specification,
    step_directory_for_specification,
    tensorboard_log_root_for_specification,
)
from phase_3_common import phase_2_manifest_paths  # noqa: E402


class Phase2RunLayoutTests(unittest.TestCase):
    def test_explicit_run_root_does_not_require_the_standalone_specification(self) -> None:
        root = (REPOSITORY_ROOT / ".test_phase2_run_root").resolve()
        missing_specification = root / "missing.json"
        with patch.dict(
            os.environ,
            {RUN_ROOT_ENVIRONMENT_VARIABLE: str(root)},
            clear=False,
        ):
            self.assertEqual(
                step_directory_for_specification(
                    STEP_5_DIRECTORY_NAME,
                    specification_path=missing_specification,
                ),
                root / STEP_5_DIRECTORY_NAME,
            )
            self.assertEqual(
                tensorboard_log_root_for_specification(
                    specification_path=missing_specification,
                ),
                root / "tensorboard_logs",
            )

    def test_standalone_run_root_still_uses_the_specification_run_number(self) -> None:
        specification = REPOSITORY_ROOT / ".test_phase2_specification.json"
        with patch(
            "run_layout.read_run_number",
            return_value=42,
        ), patch.dict(os.environ, {}, clear=False):
            os.environ.pop(RUN_ROOT_ENVIRONMENT_VARIABLE, None)
            expected = PHASE_2_ROOT / "runs" / "run_42"
            self.assertEqual(
                run_root_for_specification(specification_path=specification),
                expected,
            )
            self.assertEqual(
                step_directory_for_specification(
                    STEP_6_DIRECTORY_NAME,
                    specification_path=specification,
                ),
                expected / STEP_6_DIRECTORY_NAME,
            )

    def test_phase3_can_resolve_an_experiment_owned_phase2_run(self) -> None:
        root = (
            REPOSITORY_ROOT
            / "pipeline_experiments"
            / "runs"
            / "PE_test"
            / "phase2"
        )
        paths = phase_2_manifest_paths(17, run_root=root)
        self.assertEqual(
            paths["selection"],
            root / "5_inner_model_selection" / "selection_manifest.json",
        )
        self.assertEqual(
            paths["comparison"],
            root / "7_architecture_comparison" / "comparison_manifest.json",
        )


if __name__ == "__main__":
    unittest.main()
