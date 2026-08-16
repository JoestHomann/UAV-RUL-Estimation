"""Run and resume the complete seven-step Phase 2 architecture study.

This file is the single user-facing entry point for Phase 2. It delegates each
stage to the existing step runner with the same Python interpreter that started
this process. Scientific logic remains inside the step modules, while this file
provides ordering, progress reporting, safe resume behavior, and fail-fast
subprocess handling.
"""

from __future__ import annotations

import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Callable


PHASE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PHASE_DIR.parent

from tensorboard_monitoring import (  # noqa: E402
    DEFAULT_LOG_ROOT,
    TensorBoardMonitoringError,
    ensure_tensorboard_available,
)

SPECIFICATION_PATH = (
    PHASE_DIR
    / "1_architecture_study_settings"
    / "artifacts"
    / "experiment_specification.json"
)
TABULAR_REPORT_PATH = (
    PHASE_DIR
    / "2_tabular_data_adapter"
    / "artifacts"
    / "copy_verification.json"
)
SEQUENCE_REPORT_PATH = (
    PHASE_DIR
    / "3_sequence_data_adapter"
    / "artifacts"
    / "copy_verification.json"
)
MODEL_REGISTRY_PATH = (
    PHASE_DIR / "4_model_adapters" / "artifacts" / "model_registry.json"
)
STEP_5_MANIFEST_PATH = (
    PHASE_DIR
    / "5_inner_model_selection"
    / "artifacts"
    / "selection_manifest.json"
)
STEP_6_MANIFEST_PATH = (
    PHASE_DIR
    / "6_locked_outer_evaluation"
    / "artifacts"
    / "locked_evaluation_manifest.json"
)
STEP_7_MANIFEST_PATH = (
    PHASE_DIR
    / "7_architecture_comparison"
    / "artifacts"
    / "comparison_manifest.json"
)


class Phase2PipelineError(ValueError):
    """Explain a failed prerequisite, subprocess, or generated artifact."""


class _StudySkipped(Exception):
    """Signal that a queued family/outer-fold study was skipped after a
    sibling study already failed, rather than that this study itself failed.
    """

    def __init__(self, family: str, outer_fold: int) -> None:
        super().__init__(f"Skipped {family} outer fold {outer_fold}")


@dataclass(frozen=True)
class StepDefinition:
    """Describe one delegated Phase 2 step without duplicating its behavior."""

    number: int
    name: str
    script: Path


STEPS = {
    1: StepDefinition(
        1,
        "Architecture study settings",
        PHASE_DIR
        / "1_architecture_study_settings"
        / "build_architecture_study_settings.py",
    ),
    2: StepDefinition(
        2,
        "Tabular data adapter",
        PHASE_DIR / "2_tabular_data_adapter" / "build_tabular_data_adapter.py",
    ),
    3: StepDefinition(
        3,
        "Sequence data adapter",
        PHASE_DIR / "3_sequence_data_adapter" / "build_sequence_data_adapter.py",
    ),
    4: StepDefinition(
        4,
        "Model adapters",
        PHASE_DIR / "4_model_adapters" / "build_model_registry.py",
    ),
    5: StepDefinition(
        5,
        "Inner model selection",
        PHASE_DIR / "5_inner_model_selection" / "run_inner_model_selection.py",
    ),
    6: StepDefinition(
        6,
        "Locked outer evaluation",
        PHASE_DIR
        / "6_locked_outer_evaluation"
        / "run_locked_outer_evaluation.py",
    ),
    7: StepDefinition(
        7,
        "Architecture comparison",
        PHASE_DIR
        / "7_architecture_comparison"
        / "run_architecture_comparison.py",
    ),
}


def _read_json(path: Path, description: str) -> dict[str, Any]:
    """Read one required JSON object and keep errors tied to its pipeline role."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase2PipelineError(
            f"Cannot read {description} at {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise Phase2PipelineError(f"{description} must contain a JSON object")
    return value


def _read_optional_json(path: Path, description: str) -> dict[str, Any] | None:
    """Read an optional progress artifact while rejecting malformed content."""

    if not path.is_file():
        return None
    return _read_json(path, description)


def _load_settings() -> dict[str, Any]:
    """Return the resolved settings produced by Step 1."""

    specification = _read_json(
        SPECIFICATION_PATH,
        "Step 1 experiment specification",
    )
    settings = specification.get("settings")
    if not isinstance(settings, dict):
        raise Phase2PipelineError(
            "Step 1 experiment specification has no settings object"
        )
    return settings


def _enabled_families(settings: dict[str, Any]) -> tuple[str, ...]:
    """Preserve the family order declared before performance was observed."""

    study = settings["study"]
    declared_order = (
        study["required_architectures"]
        + study["conditional_architectures"]
        + study["optional_architectures"]
    )
    return tuple(
        family
        for family in declared_order
        if settings["architectures"][family]["enabled"]
    )


def _command_text(command: list[str]) -> str:
    """Format one command for readable Windows terminal output."""

    return subprocess.list2cmdline(command)


def _default_max_workers() -> int:
    """Use every available core unless the user asks for something else."""

    return max(1, os.cpu_count() or 1)


def _single_process_environment() -> dict[str, str]:
    """Cap BLAS/OpenMP thread pools so concurrent studies cannot oversubscribe.

    Each Step 5/6 subprocess already trains with ``n_jobs=1`` (XGBoost,
    Random Forest) or a single-threaded PyTorch device. Those settings only
    control the library's own worker count, not the lower-level thread pools
    NumPy/PyTorch create by default (often "one thread per core"). Running
    several such processes at once without this cap would let every process
    try to use every core, which oversubscribes the machine instead of
    speeding it up. Setting these variables before the child interpreter
    starts is the reliable way to bound them, since some libraries read them
    only at import time.
    """

    environment = dict(os.environ)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    return environment


def _run_script(
    step: StepDefinition,
    extra_arguments: list[str] | None = None,
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Run one existing step with this process's exact Python interpreter."""

    arguments = extra_arguments or []
    command = [sys.executable, str(step.script), *arguments]
    print("", flush=True)
    print("=" * 78, flush=True)
    print(f"Step {step.number}: {step.name}", flush=True)
    print(_command_text(command), flush=True)
    print("=" * 78, flush=True)
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        raise Phase2PipelineError(
            f"Step {step.number} failed with exit code {completed.returncode}"
        )


def _run_pairs_in_parallel(
    *,
    step: StepDefinition,
    pairs: list[tuple[str, int]],
    max_workers: int,
    describe: Callable[[tuple[str, int]], str],
) -> None:
    """Run one independent family/outer-fold study per pair, fanned out.

    Each pair becomes its own subprocess invoked with a single ``--family``
    and a single ``--outer-fold``, matching the granularity that Step 5's and
    Step 6's checkpointing, TensorBoard writers, and manifest consolidation
    already treat as one independent unit of work (see the runtime-reduction
    note in ``literature_and_planning/phase_2_runtime_reduction.md``). A
    thread pool is used rather than a process pool because each unit of work
    is itself an OS subprocess: the pool threads spend almost all their time
    blocked in ``subprocess.run``, which releases the GIL, so this adds
    concurrency without needing multiprocessing.

    Fail-fast behavior: once any study fails, no *new* studies are started,
    but studies already running are allowed to finish (rather than being
    killed mid-write) before the first recorded error is raised. This keeps
    on-disk checkpoints consistent, since every checkpoint write in Step 5
    and Step 6 is a full, atomic file replace. A shared stop signal is
    checked immediately before each subprocess actually launches, not just
    relied on through future cancellation: a worker thread can otherwise
    pull the next pair off the pool's internal queue before a cancellation
    request reaches it, which would let one extra study start after a
    failure has already been recorded.
    """

    worker_count = max(1, min(max_workers, len(pairs)))
    print(
        f"Dispatching {len(pairs)} independent studies across up to "
        f"{worker_count} concurrent worker process(es)",
        flush=True,
    )
    environment = _single_process_environment()
    stop_after_failure = threading.Event()

    def _run_one(family: str, outer_fold: int) -> None:
        if stop_after_failure.is_set():
            raise _StudySkipped(family, outer_fold)
        _run_script(
            step,
            ["--family", family, "--outer-fold", str(outer_fold)],
            env=environment,
        )

    first_error: Phase2PipelineError | None = None
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_run_one, family, outer_fold): (family, outer_fold)
            for family, outer_fold in pairs
        }
        for future in as_completed(futures):
            pair = futures[future]
            try:
                future.result()
            except (CancelledError, _StudySkipped):
                # Expected once a failure stops new studies from starting;
                # the original failure is what gets raised below.
                continue
            except Phase2PipelineError as error:
                if first_error is None:
                    first_error = error
                    stop_after_failure.set()
                    print(
                        f"Study {describe(pair)} failed; letting already-running "
                        "studies finish before stopping",
                        file=sys.stderr,
                        flush=True,
                    )
                    for pending_future in futures:
                        pending_future.cancel()
    if first_error is not None:
        raise first_error


def _settings_version_matches(
    artifact: dict[str, Any] | None,
    settings_version: int,
) -> bool:
    """Treat artifacts from another settings version as incomplete work."""

    return (
        artifact is not None
        and artifact.get("settings_version") == settings_version
    )


def _run_step_5(*, force: bool, max_workers: int) -> None:
    """Run only missing family/fold tuning studies unless forced to rerun all."""

    settings = _load_settings()
    settings_version = int(settings["settings_version"])
    families = _enabled_families(settings)
    outer_folds = tuple(
        range(int(settings["phase_1"]["expected_outer_folds"]))
    )
    expected_studies = {
        f"{family}__outer_{outer_fold:02d}"
        for family in families
        for outer_fold in outer_folds
    }
    manifest = _read_optional_json(
        STEP_5_MANIFEST_PATH,
        "Step 5 selection manifest",
    )
    completed_studies: set[str] = set()
    if not force and _settings_version_matches(manifest, settings_version):
        completed_value = manifest.get("completed_studies", [])
        if not isinstance(completed_value, list):
            raise Phase2PipelineError(
                "Step 5 manifest completed_studies must be a list"
            )
        completed_studies = set(map(str, completed_value)) & expected_studies

    if force:
        print("Step 5 force mode: all tuning studies will be replaced", flush=True)
    elif completed_studies:
        print(
            f"Step 5 resume: preserving {len(completed_studies)}/"
            f"{len(expected_studies)} completed studies",
            flush=True,
        )

    pending_pairs = [
        (family, outer_fold)
        for family in families
        for outer_fold in outer_folds
        if force or f"{family}__outer_{outer_fold:02d}" not in completed_studies
    ]
    if not pending_pairs:
        print("Step 5: every family/fold study is already complete; skipping", flush=True)
    else:
        _run_pairs_in_parallel(
            step=STEPS[5],
            pairs=pending_pairs,
            max_workers=max_workers,
            describe=lambda pair: f"{pair[0]} outer fold {pair[1]}",
        )

    final_manifest = _read_json(
        STEP_5_MANIFEST_PATH,
        "Step 5 selection manifest",
    )
    if (
        final_manifest.get("status") != "complete"
        or final_manifest.get("completed_study_count") != len(expected_studies)
    ):
        raise Phase2PipelineError(
            "Step 5 did not finish every required family/fold study: "
            f"{final_manifest.get('completed_study_count', 0)}/"
            f"{len(expected_studies)} complete"
        )


def _step_6_expected_runs(
    settings: dict[str, Any],
) -> tuple[dict[tuple[str, int], set[str]], int]:
    """Derive complete family/fold run sets from the settings and registry."""

    registry = _read_json(MODEL_REGISTRY_PATH, "Step 4 model registry")
    settings_version = int(settings["settings_version"])
    if registry.get("settings_version") != settings_version:
        raise Phase2PipelineError(
            "Step 4 model registry uses a different settings version"
        )
    registry_families = registry.get("families")
    if not isinstance(registry_families, dict):
        raise Phase2PipelineError("Step 4 model registry has no families object")

    families = _enabled_families(settings)
    outer_folds = tuple(
        range(int(settings["phase_1"]["expected_outer_folds"]))
    )
    retraining_seeds = tuple(
        int(seed) for seed in settings["tuning"]["retraining_seeds"]
    )
    if not retraining_seeds:
        raise Phase2PipelineError("The settings do not define retraining seeds")

    expected_by_family_fold: dict[tuple[str, int], set[str]] = {}
    for family in families:
        family_registry = registry_families.get(family)
        if not isinstance(family_registry, dict):
            raise Phase2PipelineError(
                f"Step 4 model registry has no entry for {family!r}"
            )
        family_seeds = (
            retraining_seeds
            if family_registry.get("stochastic") is True
            else (retraining_seeds[0],)
        )
        for outer_fold in outer_folds:
            expected_by_family_fold[(family, outer_fold)] = {
                f"{family}__outer_{outer_fold:02d}__seed_{seed:03d}"
                for seed in family_seeds
            }
    expected_run_count = sum(len(names) for names in expected_by_family_fold.values())
    return expected_by_family_fold, expected_run_count


def _run_step_6(*, force: bool, max_workers: int) -> None:
    """Run missing complete family/fold evaluations and preserve finished ones."""

    settings = _load_settings()
    settings_version = int(settings["settings_version"])
    expected_by_pair, expected_run_count = _step_6_expected_runs(settings)
    manifest = _read_optional_json(
        STEP_6_MANIFEST_PATH,
        "Step 6 locked-evaluation manifest",
    )
    completed_runs: set[str] = set()
    if not force and _settings_version_matches(manifest, settings_version):
        completed_value = manifest.get("completed_runs", [])
        if not isinstance(completed_value, list):
            raise Phase2PipelineError(
                "Step 6 manifest completed_runs must be a list"
            )
        completed_runs = set(map(str, completed_value))

    if force:
        print("Step 6 force mode: all locked evaluations will be replaced", flush=True)
    elif completed_runs:
        print(
            f"Step 6 resume: found {len(completed_runs)}/"
            f"{expected_run_count} completed runs",
            flush=True,
        )

    pending_pairs = [
        pair
        for pair, required_runs in expected_by_pair.items()
        if force or not required_runs.issubset(completed_runs)
    ]
    if not pending_pairs:
        print("Step 6: every family/fold evaluation is already complete; skipping", flush=True)
    else:
        _run_pairs_in_parallel(
            step=STEPS[6],
            pairs=pending_pairs,
            max_workers=max_workers,
            describe=lambda pair: f"{pair[0]} outer fold {pair[1]}",
        )

    final_manifest = _read_json(
        STEP_6_MANIFEST_PATH,
        "Step 6 locked-evaluation manifest",
    )
    if (
        final_manifest.get("status") != "complete"
        or final_manifest.get("completed_run_count") != expected_run_count
    ):
        raise Phase2PipelineError(
            "Step 6 did not finish every required family/fold/seed run: "
            f"{final_manifest.get('completed_run_count', 0)}/"
            f"{expected_run_count} complete"
        )


def _run_step_7() -> None:
    """Regenerate the inexpensive comparison from the current Step 6 outputs."""

    # Step 7 is intentionally rerun whenever requested. This keeps its tables
    # and figures synchronized with Step 6 without introducing timestamps or
    # hashes, both of which are deliberately absent from this project design.
    _run_script(STEPS[7])
    manifest = _read_json(
        STEP_7_MANIFEST_PATH,
        "Step 7 comparison manifest",
    )
    if manifest.get("status") != "complete":
        raise Phase2PipelineError("Step 7 comparison manifest is not complete")
    if manifest.get("automatic_architecture_selection") is not False:
        raise Phase2PipelineError(
            "Step 7 unexpectedly enabled automatic architecture selection"
        )


def _simple_status(
    path: Path,
    description: str,
    *,
    status_key: str = "status",
) -> str:
    """Return one concise status value without modifying pipeline state."""

    artifact = _read_optional_json(path, description)
    if artifact is None:
        return "not generated"
    return str(artifact.get(status_key, "available"))


def _progress_status(
    path: Path,
    description: str,
    completed_key: str,
    expected_key: str,
) -> str:
    """Return status plus completed and expected work counts."""

    artifact = _read_optional_json(path, description)
    if artifact is None:
        return "not run"
    status = artifact.get("status", "unknown")
    completed = artifact.get(completed_key, "?")
    expected = artifact.get(expected_key, "?")
    return f"{status} ({completed}/{expected})"


def print_status() -> None:
    """Print the seven-step state without importing or running a step module."""

    print("Phase 2 pipeline status")
    print(f"TensorBoard logs: {DEFAULT_LOG_ROOT}")
    print(
        "1. Architecture study settings: "
        + ("available" if SPECIFICATION_PATH.is_file() else "not generated")
    )
    print(
        "2. Tabular data adapter: "
        + _simple_status(TABULAR_REPORT_PATH, "Step 2 copy report")
    )
    print(
        "3. Sequence data adapter: "
        + _simple_status(SEQUENCE_REPORT_PATH, "Step 3 copy report")
    )
    print(
        "4. Model adapters: "
        + ("available" if MODEL_REGISTRY_PATH.is_file() else "not generated")
    )
    print(
        "5. Inner model selection: "
        + _progress_status(
            STEP_5_MANIFEST_PATH,
            "Step 5 selection manifest",
            "completed_study_count",
            "expected_study_count",
        )
    )
    print(
        "6. Locked outer evaluation: "
        + _progress_status(
            STEP_6_MANIFEST_PATH,
            "Step 6 locked-evaluation manifest",
            "completed_run_count",
            "expected_run_count",
        )
    )
    print(
        "7. Architecture comparison: "
        + _simple_status(STEP_7_MANIFEST_PATH, "Step 7 comparison manifest")
    )


def run_pipeline(
    first_step: int,
    last_step: int,
    *,
    force: bool,
    max_workers: int,
) -> None:
    """Execute a validated inclusive step range in dependency order."""

    if first_step > last_step:
        raise Phase2PipelineError(
            "--from-step cannot be greater than --through-step"
        )
    try:
        tensorboard_version = ensure_tensorboard_available()
    except TensorBoardMonitoringError as error:
        raise Phase2PipelineError(str(error)) from error
    requested_steps = range(first_step, last_step + 1)
    print(
        f"Running Phase 2 Steps {first_step}-{last_step} with "
        f"{sys.executable}",
        flush=True,
    )
    print(
        f"TensorBoard {tensorboard_version} logging to {DEFAULT_LOG_ROOT}",
        flush=True,
    )
    if force:
        print(
            "Force mode is enabled for the expensive Step 5 and Step 6 work",
            flush=True,
        )
    if 5 in requested_steps or 6 in requested_steps:
        print(
            f"Step 5/6 studies run up to {max_workers} at a time "
            "(--max-workers); output from concurrent studies interleaves in "
            "this console, but each study's own checkpoint files stay separate",
            flush=True,
        )

    for step_number in requested_steps:
        if step_number <= 4:
            _run_script(STEPS[step_number])
        elif step_number == 5:
            _run_step_5(force=force, max_workers=max_workers)
        elif step_number == 6:
            _run_step_6(force=force, max_workers=max_workers)
        else:
            _run_step_7()

    print("", flush=True)
    print("Requested Phase 2 pipeline range completed successfully", flush=True)
    print_status()


def main() -> None:
    """Parse the small orchestration interface and run or inspect Phase 2."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current progress without running or modifying any step.",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        choices=range(1, 8),
        default=1,
        help="First step to run; defaults to Step 1.",
    )
    parser.add_argument(
        "--through-step",
        type=int,
        choices=range(1, 8),
        default=7,
        help="Last step to run; defaults to Step 7.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace completed Step 5 studies and Step 6 evaluations. "
            "Without this option, expensive completed work is preserved."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=_default_max_workers(),
        help=(
            "Maximum number of independent Step 5/6 family/outer-fold "
            "studies to run at the same time, each as its own subprocess. "
            "Defaults to the number of available CPU cores. Use 1 to run "
            "sequentially, matching the previous behavior."
        ),
    )
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")

    try:
        if args.status:
            print_status()
            return
        run_pipeline(
            args.from_step,
            args.through_step,
            force=args.force,
            max_workers=args.max_workers,
        )
    except KeyboardInterrupt:
        print(
            "\nPhase 2 interrupted; completed step artifacts remain available for resume",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Phase2PipelineError as error:
        print(f"Phase 2 pipeline stopped:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
