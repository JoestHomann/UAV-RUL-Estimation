"""Resolve the numbered run folder that Steps 5, 6 and 7 read and write.

Every artifact those three steps produce lives under::

    2_model_architecture_study/runs/run_<n>/<step directory name>/

where ``n`` is the ``run_number`` recorded in the architecture study settings.
One folder therefore holds one complete run of the expensive part of Phase 2,
which is what makes a finished run archivable and two runs comparable without
either of them having to be moved out of the way first.

The run number is read, never written. Nothing in the pipeline advances it, so
interrupting Phase 2 and resuming it later resolves to the same folder and the
resumed work joins the work that already finished. A run only ever changes when
the operator edits the settings file, and that deliberately starts the new run
from nothing rather than resuming the previous one.

Steps 1 through 4 keep their fixed ``artifacts`` directories. Their outputs are
the validated settings, the copied data adapters and the model registry, all of
which are shared by every run and are reproduced identically from the same
inputs, so giving each run its own copy would duplicate large datasets for no
traceability gain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PHASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = PHASE_DIR / "runs"
SPECIFICATION_PATH = (
    PHASE_DIR
    / "1_architecture_study_settings"
    / "artifacts"
    / "experiment_specification.json"
)

# The step directories that hold per-run results. The names match the directory
# each step's code lives in, so a run folder reads the same way as the phase.
STEP_5_DIRECTORY_NAME = "5_inner_model_selection"
STEP_6_DIRECTORY_NAME = "6_locked_outer_evaluation"
STEP_7_DIRECTORY_NAME = "7_architecture_comparison"


class RunLayoutError(ValueError):
    """Explain a missing or unusable run number."""


def read_run_number(
    specification_path: Path = SPECIFICATION_PATH,
) -> int:
    """Read the current run number from Step 1's generated specification.

    Reading it from the specification rather than the TOML keeps one validation
    path: the value has already passed the settings schema by the time any step
    can see it, and a step therefore cannot run against a settings file that
    Step 1 has not accepted.
    """

    try:
        payload = json.loads(specification_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RunLayoutError(
            "Cannot determine the run number because Step 1's experiment "
            f"specification is missing at {specification_path}. Run Step 1 "
            "before Steps 5, 6 or 7."
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise RunLayoutError(
            f"Cannot read the experiment specification {specification_path}: "
            f"{error}"
        ) from error
    return run_number_from_settings(payload.get("settings"))


def run_number_from_settings(settings: Any) -> int:
    """Extract and validate the run number from a resolved settings object."""

    if not isinstance(settings, dict):
        raise RunLayoutError(
            "The experiment specification has no settings object"
        )
    if "run_number" not in settings:
        raise RunLayoutError(
            "The experiment specification has no run_number. Rebuild it with "
            "Step 1 after adding run_number to the settings file."
        )
    value = settings["run_number"]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RunLayoutError(
            f"run_number must be a positive integer, found {value!r}"
        )
    return value


def run_root(run_number: int, runs_dir: Path = RUNS_DIR) -> Path:
    """Return the folder holding every artifact of one numbered run."""

    if isinstance(run_number, bool) or not isinstance(run_number, int):
        raise RunLayoutError(f"run_number must be an integer, found {run_number!r}")
    if run_number <= 0:
        raise RunLayoutError(f"run_number must be positive, found {run_number}")
    return runs_dir / f"run_{run_number}"


def step_directory(
    step_directory_name: str,
    *,
    run_number: int,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """Return one step's output directory inside a numbered run."""

    if step_directory_name not in {
        STEP_5_DIRECTORY_NAME,
        STEP_6_DIRECTORY_NAME,
        STEP_7_DIRECTORY_NAME,
    }:
        raise RunLayoutError(
            f"Unknown per-run step directory {step_directory_name!r}"
        )
    return run_root(run_number, runs_dir) / step_directory_name


def step_directory_for_specification(
    step_directory_name: str,
    *,
    specification_path: Path = SPECIFICATION_PATH,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """Resolve one step's output directory from the specification on disk.

    This is the call every step uses to build its default ``--output-dir`` and
    its default upstream input paths, so a run number changes all of them at
    once and no step can silently write into a different run than it reads.
    """

    return step_directory(
        step_directory_name,
        run_number=read_run_number(specification_path),
        runs_dir=runs_dir,
    )
