"""Resolve the run folder that Steps 5, 6 and 7 read and write.

Standalone Phase 2 runs store those artifacts under::

    2_architecture_experiments/2_model_architecture_study/runs/run_<n>/<step directory name>/

where ``n`` is the ``run_number`` recorded in the architecture study settings.
An owning orchestrator may instead set ``PHASE2_RUN_ROOT`` so the same steps
write beneath its own run directory. Pipeline experiments use this to keep all
seven Phase 2 steps together under
``2_architecture_experiments/1_pipeline_experiments/runs/<experiment>/phase2``.

The run number is read, never written. Nothing in the pipeline advances it, so
interrupting Phase 2 and resuming it later resolves to the same folder and the
resumed work joins the work that already finished. A run only ever changes when
the operator edits the settings file, and that deliberately starts the new run
from nothing rather than resuming the previous one.

The override is scoped to the runner process and inherited by its workers.
Without it, the established standalone layout and resume behavior are
unchanged.
"""

from __future__ import annotations

import json
import os
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

# TensorBoard event files sit beside the step folders rather than in the
# monitoring package, so a run's curves are archived, copied and deleted with
# the results they describe instead of outliving them in a shared directory.
TENSORBOARD_DIRECTORY_NAME = "tensorboard_logs"

# An orchestrator may own the run folder outside the standalone Phase 2 runs
# directory. The override is process-local and inherited by study subprocesses;
# omitting it preserves the numbered standalone layout above.
RUN_ROOT_ENVIRONMENT_VARIABLE = "PHASE2_RUN_ROOT"


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


def run_root_for_specification(
    *,
    specification_path: Path = SPECIFICATION_PATH,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """Resolve the explicit orchestration root or the numbered default."""

    override = os.environ.get(RUN_ROOT_ENVIRONMENT_VARIABLE, "").strip()
    if override:
        return Path(override).resolve()
    return run_root(read_run_number(specification_path), runs_dir)


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


def tensorboard_log_root(run_number: int, runs_dir: Path = RUNS_DIR) -> Path:
    """Return the TensorBoard event directory belonging to one numbered run."""

    return run_root(run_number, runs_dir) / TENSORBOARD_DIRECTORY_NAME


def tensorboard_log_root_for_specification(
    *,
    specification_path: Path = SPECIFICATION_PATH,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """Resolve the current run's TensorBoard event directory from the settings.

    Steps 5 and 6 call this so their curves land in the same run folder as
    their results, and the dashboard launcher calls it so it opens on the run
    the pipeline is actually writing.
    """

    return (
        run_root_for_specification(
            specification_path=specification_path,
            runs_dir=runs_dir,
        )
        / TENSORBOARD_DIRECTORY_NAME
    )


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

    if step_directory_name not in {
        STEP_5_DIRECTORY_NAME,
        STEP_6_DIRECTORY_NAME,
        STEP_7_DIRECTORY_NAME,
    }:
        raise RunLayoutError(
            f"Unknown per-run step directory {step_directory_name!r}"
        )
    return (
        run_root_for_specification(
            specification_path=specification_path,
            runs_dir=runs_dir,
        )
        / step_directory_name
    )
