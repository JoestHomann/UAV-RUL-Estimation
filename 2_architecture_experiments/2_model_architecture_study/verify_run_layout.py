"""Verify the numbered-run layout and the Step 7 settings snapshot.

This check runs no training and reads no locked data. It confirms three things
that are easy to break silently the next time a path is touched:

- every per-run path resolves under "runs/run_<n>/", and an invalid run number
  is refused rather than quietly producing a folder named after it;
- Steps 5, 6 and 7 agree on the same run for a given specification, so no step
  can write into one run while reading another;
- the flattened settings snapshot is lossless, with one row per leaf value and
  no duplicated or empty keys.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import tomllib
import types


PHASE_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = (
    PHASE_DIR
    / "1_architecture_study_settings"
    / "architecture_study_settings.toml"
)
for dependency_dir in (PHASE_DIR, PHASE_DIR / "7_architecture_comparison"):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from run_layout import (  # noqa: E402
    RunLayoutError,
    STEP_5_DIRECTORY_NAME,
    STEP_6_DIRECTORY_NAME,
    STEP_7_DIRECTORY_NAME,
    TENSORBOARD_DIRECTORY_NAME,
    read_run_number,
    run_number_from_settings,
    run_root,
    step_directory,
    step_directory_for_specification,
    tensorboard_log_root,
    tensorboard_log_root_for_specification,
)


def _require(condition: bool, message: str) -> None:
    """Fail the whole verification on the first broken expectation."""

    if not condition:
        raise RuntimeError(message)


def _verify_run_paths() -> None:
    """Confirm run folders are built from the run number and nothing else."""

    _require(
        run_root(3).name == "run_3",
        "run_root must name the folder after the run number",
    )
    _require(
        run_root(3).parent.name == "runs",
        "run folders must live under the shared runs directory",
    )
    for step_name in (
        STEP_5_DIRECTORY_NAME,
        STEP_6_DIRECTORY_NAME,
        STEP_7_DIRECTORY_NAME,
    ):
        resolved = step_directory(step_name, run_number=7)
        _require(
            resolved.parent == run_root(7),
            f"{step_name} must sit directly inside its run folder",
        )
        _require(
            resolved.name == step_name,
            f"{step_name} must keep its own directory name inside the run",
        )

    # A run number that cannot be trusted must stop the pipeline instead of
    # creating a folder named after a nonsense value.
    for rejected, description in (
        (0, "zero"),
        (-1, "a negative number"),
        (True, "a boolean"),
    ):
        try:
            run_root(rejected)
        except RunLayoutError:
            continue
        raise RuntimeError(f"run_root accepted {description}")
    try:
        step_directory("4_model_adapters", run_number=1)
    except RunLayoutError:
        pass
    else:
        raise RuntimeError("step_directory accepted a step that has no per-run folder")

    # TensorBoard events belong to the run they describe, so their directory
    # sits beside the step folders and moves with the run.
    events = tensorboard_log_root(7)
    _require(
        events.parent == run_root(7),
        "the TensorBoard log root must sit directly inside its run folder",
    )
    _require(
        events.name == TENSORBOARD_DIRECTORY_NAME,
        "the TensorBoard log root must use the shared directory name",
    )
    _require(
        tensorboard_log_root(7) != tensorboard_log_root(8),
        "two runs must not share one TensorBoard log root",
    )
    for rejected in (0, -1, True):
        try:
            tensorboard_log_root(rejected)
        except RunLayoutError:
            continue
        raise RuntimeError(f"tensorboard_log_root accepted {rejected!r}")


def _verify_settings_reading() -> None:
    """Confirm the run number is read from a specification and validated."""

    for rejected, description in (
        ({}, "settings without a run number"),
        ({"run_number": 0}, "a zero run number"),
        ({"run_number": -4}, "a negative run number"),
        ({"run_number": True}, "a boolean run number"),
        ({"run_number": "2"}, "a string run number"),
        (None, "a missing settings object"),
    ):
        try:
            run_number_from_settings(rejected)
        except RunLayoutError:
            continue
        raise RuntimeError(f"run_number_from_settings accepted {description}")
    _require(
        run_number_from_settings({"run_number": 12}) == 12,
        "a valid run number must be returned unchanged",
    )

    with tempfile.TemporaryDirectory() as scratch:
        specification_path = Path(scratch) / "experiment_specification.json"
        specification_path.write_text(
            json.dumps({"settings": {"run_number": 9}}),
            encoding="utf-8",
        )
        _require(
            read_run_number(specification_path) == 9,
            "the run number must be read from the specification on disk",
        )
        # Every step must resolve the same run from the same specification.
        # Disagreement here is what would let one step read run 9 while another
        # writes run 10.
        runs_dir = Path(scratch) / "runs"
        resolved = {
            name: step_directory_for_specification(
                name,
                specification_path=specification_path,
                runs_dir=runs_dir,
            )
            for name in (
                STEP_5_DIRECTORY_NAME,
                STEP_6_DIRECTORY_NAME,
                STEP_7_DIRECTORY_NAME,
            )
        }
        events = tensorboard_log_root_for_specification(
            specification_path=specification_path,
            runs_dir=runs_dir,
        )
        parents = {path.parent for path in resolved.values()} | {events.parent}
        _require(
            len(parents) == 1 and parents.pop().name == "run_9",
            "Steps 5, 6, 7 and TensorBoard must resolve to one shared run folder",
        )

        missing = Path(scratch) / "absent.json"
        try:
            read_run_number(missing)
        except RunLayoutError:
            pass
        else:
            raise RuntimeError("a missing specification must be reported clearly")


def _verify_settings_snapshot() -> int:
    """Confirm the Step 7 settings CSV is a lossless flat view of the settings.

    Returns the row count so the summary can report the snapshot's size.
    """

    # The comparison module imports its gate for a type only; stubbing it keeps
    # this check free of the heavy Step 6 and model-registry imports.
    gate = types.ModuleType("comparison_gate")

    class _Plan:
        """Stand in for the plan type the comparison module annotates against."""

    gate.ArchitectureComparisonPlan = _Plan
    sys.modules.setdefault("comparison_gate", gate)
    from architecture_comparison import settings_table  # noqa: E402

    settings = tomllib.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    table = settings_table(settings)

    _require(
        list(table.columns) == ["setting", "value"],
        "the settings snapshot must have exactly a setting and a value column",
    )
    duplicated = table["setting"][table["setting"].duplicated()].tolist()
    _require(not duplicated, f"the settings snapshot has duplicate keys: {duplicated}")
    _require(
        not (table["setting"] == "").any(),
        "the settings snapshot must not contain an empty key",
    )
    for required in ("settings_version", "run_number"):
        _require(
            (table["setting"] == required).any(),
            f"the settings snapshot must record {required}",
        )

    # Every scalar in the source settings must survive into exactly one row.
    def count_leaves(value: object) -> int:
        if isinstance(value, dict):
            return sum(count_leaves(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(count_leaves(item) for item in value)
        return 1

    expected = count_leaves(settings)
    _require(
        len(table) == expected,
        f"the settings snapshot has {len(table)} rows for {expected} settings",
    )
    return len(table)


def main() -> None:
    """Run every layout and snapshot check against the real settings file."""

    try:
        _verify_run_paths()
        _verify_settings_reading()
        rows = _verify_settings_snapshot()
    except (RuntimeError, OSError, ValueError) as error:
        print(f"Run layout verification failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    settings = tomllib.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    run_number = int(settings["run_number"])
    print("Run layout verified")
    print(f"Settings run_number: {run_number}")
    print(f"Current run folder:  {run_root(run_number)}")
    for name in (
        STEP_5_DIRECTORY_NAME,
        STEP_6_DIRECTORY_NAME,
        STEP_7_DIRECTORY_NAME,
    ):
        print(f"  {step_directory(name, run_number=run_number)}")
    print(f"  {tensorboard_log_root(run_number)}")
    print(f"Settings snapshot rows: {rows}")


if __name__ == "__main__":
    main()
