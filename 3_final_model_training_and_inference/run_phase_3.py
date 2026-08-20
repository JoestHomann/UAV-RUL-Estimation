"""Run, inspect, and resume the complete seven-step Phase 3 pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any


PHASE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PHASE_DIR.parent
if str(PHASE_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE_DIR))

from phase_3_common import (  # noqa: E402
    manifest_path,
    read_optional_json,
    run_root,
)
from phase_3_run_layout import (  # noqa: E402
    SETTINGS_PATH,
    Phase3RunLayoutError,
    read_run_number,
)


class Phase3PipelineError(ValueError):
    """Explain an invalid requested range or failed delegated stage."""


@dataclass(frozen=True)
class StepDefinition:
    number: int
    name: str
    script: Path


STEPS = {
    1: StepDefinition(
        1,
        "Winning architecture selection",
        PHASE_DIR
        / "1_winning_architecture_selection"
        / "build_selected_architecture.py",
    ),
    2: StepDefinition(
        2,
        "Final configuration search",
        PHASE_DIR
        / "2_final_configuration_search"
        / "run_final_configuration_search.py",
    ),
    3: StepDefinition(
        3,
        "Final training contract",
        PHASE_DIR
        / "3_final_training_contract"
        / "build_final_training_contract.py",
    ),
    4: StepDefinition(
        4,
        "Final model training",
        PHASE_DIR / "4_final_model_training" / "run_final_model_training.py",
    ),
    5: StepDefinition(
        5,
        "Test inference",
        PHASE_DIR / "5_test_inference" / "run_test_inference.py",
    ),
    6: StepDefinition(
        6,
        "Submission verification",
        PHASE_DIR / "6_submission_verification" / "build_submission.py",
    ),
    7: StepDefinition(
        7,
        "Post-run reporting",
        PHASE_DIR / "7_post_run_reporting" / "build_phase_3_report.py",
    ),
}


def _settings_summary(settings_path: Path) -> dict[str, Any]:
    try:
        with settings_path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise Phase3PipelineError(f"Cannot read settings {settings_path}: {error}") from error
    return payload


def print_status(settings_path: Path = SETTINGS_PATH) -> None:
    """Report current run and manifest states without writing files."""

    settings = _settings_summary(settings_path)
    run_number = read_run_number(settings_path)
    print("Phase 3 status")
    print(f"Settings version: {settings.get('settings_version')}")
    print(f"Phase 3 run:     {run_number}")
    print(f"Phase 2 source:  run_{settings.get('phase_2_run_number')}")
    print(f"Selected family: {settings.get('selected_model_family')}")
    print(f"Run folder:      {run_root(run_number)}")
    for number, step in STEPS.items():
        manifest = read_optional_json(
            manifest_path(number, run_number),
            f"Step {number} manifest",
        )
        state = "not started" if manifest is None else str(manifest.get("status"))
        if number == 2 and manifest is None:
            progress_path = manifest_path(number, run_number).parent / "final_search_status.json"
            if progress_path.is_file():
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                state = (
                    f"{progress.get('state')}: "
                    f"{progress.get('completed_candidates', 0)}/"
                    f"{progress.get('candidate_budget', '?')} candidates"
                )
        print(f"  Step {number}: {step.name:<34} {state}")


def _run_step(
    step: StepDefinition,
    *,
    settings_path: Path,
    force: bool,
) -> None:
    command = [sys.executable, str(step.script), "--settings", str(settings_path)]
    if force and 2 <= step.number <= 6:
        command.append("--force")
    print("", flush=True)
    print("=" * 78, flush=True)
    print(f"Step {step.number}: {step.name}", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    print("=" * 78, flush=True)
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if completed.returncode != 0:
        raise Phase3PipelineError(
            f"Step {step.number} failed with exit code {completed.returncode}"
        )


def run_pipeline(
    from_step: int,
    through_step: int,
    *,
    settings_path: Path = SETTINGS_PATH,
    force: bool = False,
) -> None:
    if from_step > through_step:
        raise Phase3PipelineError("--from-step cannot exceed --through-step")
    settings_path = settings_path.resolve()
    settings = _settings_summary(settings_path)
    print(
        f"Phase 3 Run {settings.get('run_number')} from Step {from_step} "
        f"through Step {through_step}",
        flush=True,
    )
    print("Final configuration search runs sequentially (one fit at a time)", flush=True)
    if force:
        print("Force mode will replace completed work in requested Steps 2-6", flush=True)
        if through_step >= 7:
            print("Step 7 reporting will be regenerated from the resulting artifacts", flush=True)
    for number in range(from_step, through_step + 1):
        _run_step(STEPS[number], settings_path=settings_path, force=force)
    print("\nRequested Phase 3 range completed successfully", flush=True)
    print_status(settings_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--from-step", type=int, choices=range(1, 8), default=1)
    parser.add_argument("--through-step", type=int, choices=range(1, 8), default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--settings", type=Path, default=SETTINGS_PATH)
    args = parser.parse_args()
    try:
        if args.status:
            print_status(args.settings)
            return
        run_pipeline(
            args.from_step,
            args.through_step,
            settings_path=args.settings,
            force=args.force,
        )
    except KeyboardInterrupt:
        print(
            "\nPhase 3 interrupted; completed checkpoints remain available for resume",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except (Phase3PipelineError, Phase3RunLayoutError, ValueError) as error:
        print(f"Phase 3 pipeline stopped:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
