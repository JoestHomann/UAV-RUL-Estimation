"""Resolve Phase 3's numbered run and step directories."""

from __future__ import annotations

from pathlib import Path
import json
import tomllib
from typing import Any


PHASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = PHASE_DIR / "runs"
SETTINGS_PATH = (
    PHASE_DIR
    / "1_winning_architecture_selection"
    / "phase_3_settings.toml"
)

STEP_DIRECTORY_NAMES = {
    1: "1_winning_architecture_selection",
    2: "2_final_configuration_search",
    3: "3_final_training_contract",
    4: "4_final_model_training",
    5: "5_test_inference",
    6: "6_submission_verification",
    7: "7_post_run_reporting",
}


class Phase3RunLayoutError(ValueError):
    """Explain an invalid Phase 3 run number or step path."""


def positive_integer(value: Any, name: str) -> int:
    """Return a strict positive integer for a run identity."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Phase3RunLayoutError(
            f"{name} must be a positive integer, found {value!r}"
        )
    return value


def read_run_number(settings_path: Path = SETTINGS_PATH) -> int:
    """Read the active Phase 3 run number from its human-edited TOML."""

    try:
        if settings_path.suffix.lower() == ".json":
            payload = json.loads(settings_path.read_text(encoding="utf-8"))
        else:
            with settings_path.open("rb") as stream:
                payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError, json.JSONDecodeError) as error:
        raise Phase3RunLayoutError(
            f"Cannot read Phase 3 settings {settings_path}: {error}"
        ) from error
    return positive_integer(payload.get("run_number"), "run_number")


def run_root(run_number: int, runs_dir: Path = RUNS_DIR) -> Path:
    """Return the root containing one complete Phase 3 run."""

    return runs_dir / f"run_{positive_integer(run_number, 'run_number')}"


def step_directory(
    step_number: int,
    *,
    run_number: int,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """Return one step output directory inside a Phase 3 run."""

    try:
        name = STEP_DIRECTORY_NAMES[step_number]
    except KeyError as error:
        raise Phase3RunLayoutError(
            f"Unknown Phase 3 step {step_number!r}"
        ) from error
    return run_root(run_number, runs_dir) / name


def active_step_directory(
    step_number: int,
    *,
    settings_path: Path = SETTINGS_PATH,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """Resolve one step directory from the active TOML run number."""

    return step_directory(
        step_number,
        run_number=read_run_number(settings_path),
        runs_dir=runs_dir,
    )


def tensorboard_log_root(
    run_number: int,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """Return the run-local TensorBoard directory."""

    return run_root(run_number, runs_dir) / "tensorboard_logs"
