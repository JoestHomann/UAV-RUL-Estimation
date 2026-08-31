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
ARCHITECTURE_EXPERIMENTS_ROOT = PHASE_DIR.parent
REPOSITORY_ROOT = ARCHITECTURE_EXPERIMENTS_ROOT.parent

from run_layout import (  # noqa: E402
    RUN_ROOT_ENVIRONMENT_VARIABLE,
    RunLayoutError,
    STEP_5_DIRECTORY_NAME,
    STEP_6_DIRECTORY_NAME,
    STEP_7_DIRECTORY_NAME,
    read_run_number,
    run_root_for_specification,
    step_directory_for_specification,
    tensorboard_log_root_for_specification,
)
from tensorboard_monitoring import (  # noqa: E402
    FIT_CURVE_ENVIRONMENT_VARIABLE,
    TensorBoardMonitoringError,
    ensure_tensorboard_available,
    step_5_fit_curves_enabled,
    log_global_progress,
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
TRAJECTORY_REPORT_PATH = (
    PHASE_DIR
    / "3_trajectory_data_adapter"
    / "artifacts"
    / "trajectory_verification.json"
)
MODEL_REGISTRY_PATH = (
    PHASE_DIR / "4_model_adapters" / "artifacts" / "model_registry.json"
)

# These paths default to the standard Phase 2 artifacts, but the experiment
# manager can replace them for one invocation so parallel Steps 5 and 6 use
# that experiment's local inputs without changing the global defaults.
INPUT_ARGUMENTS: list[str] = []


def _configure_invocation(
    *,
    specification_path: Path,
    tabular_manifest_path: Path | None,
    sequence_manifest_path: Path | None,
    trajectory_manifest_path: Path | None,
    model_registry_path: Path,
    run_root_path: Path | None,
) -> None:
    """Configure optional experiment-local inputs for this process."""

    global SPECIFICATION_PATH, MODEL_REGISTRY_PATH, INPUT_ARGUMENTS

    SPECIFICATION_PATH = specification_path.resolve()
    MODEL_REGISTRY_PATH = model_registry_path.resolve()
    if run_root_path is not None:
        os.environ[RUN_ROOT_ENVIRONMENT_VARIABLE] = str(run_root_path.resolve())
    INPUT_ARGUMENTS = []
    for flag, path in (
        ("--tabular-manifest", tabular_manifest_path),
        ("--sequence-manifest", sequence_manifest_path),
        ("--trajectory-manifest", trajectory_manifest_path),
    ):
        if path is not None:
            INPUT_ARGUMENTS.extend((flag, str(path.resolve())))


def _step_5_manifest_path() -> Path:
    """Locate Step 5's manifest inside the run folder the settings select."""

    return (
        step_directory_for_specification(
            STEP_5_DIRECTORY_NAME,
            specification_path=SPECIFICATION_PATH,
        )
        / "selection_manifest.json"
    )


def _step_6_manifest_path() -> Path:
    """Locate Step 6's manifest inside the run folder the settings select."""

    return (
        step_directory_for_specification(
            STEP_6_DIRECTORY_NAME,
            specification_path=SPECIFICATION_PATH,
        )
        / "locked_evaluation_manifest.json"
    )


def _step_7_manifest_path() -> Path:
    """Locate Step 7's manifest inside the run folder the settings select."""

    return (
        step_directory_for_specification(
            STEP_7_DIRECTORY_NAME,
            specification_path=SPECIFICATION_PATH,
        )
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

TRAJECTORY_STEP = StepDefinition(
    3,
    "Trajectory data adapter",
    PHASE_DIR
    / "3_trajectory_data_adapter"
    / "build_trajectory_data_adapter.py",
)


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
        study["architectures_to_run"]
        + study["conditional_architectures"]
        + study["optional_architectures"]
    )
    return tuple(
        family
        for family in declared_order
        if settings["study"]["enabled"][family]
    )


def _command_text(command: list[str]) -> str:
    """Format one command for readable Windows terminal output."""

    return subprocess.list2cmdline(command)


def _default_max_workers() -> int:
    """Use every available core unless the user asks for something else."""

    return max(1, os.cpu_count() or 1)


def _resolve_max_workers(settings: dict[str, Any], cli_value: int | None) -> int:
    """Resolve the effective Step 5/6 concurrency for this run.

    The TOML "[execution].max_workers" setting is the source of truth. The
    "--max-workers" CLI flag is an optional one-off override, consistent with
    "--force": it never changes the settings file, and it is honored only
    when the user actually passes it (the argparse default is "None").
    """

    if cli_value is not None:
        return cli_value
    configured = settings["execution"]["max_workers"]
    if configured == "auto":
        return _default_max_workers()
    return int(configured)


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


def _input_arguments() -> list[str]:
    """Return the local adapter arguments configured for this invocation."""

    return INPUT_ARGUMENTS.copy()


def _study_arguments(step: StepDefinition) -> list[str]:
    """Build the runner arguments for one parallel family/fold study."""

    if step.number == 5:
        output_dir = step_directory_for_specification(
            STEP_5_DIRECTORY_NAME,
            specification_path=SPECIFICATION_PATH,
        )
        return [
            "--specification",
            str(SPECIFICATION_PATH),
            "--output-dir",
            str(output_dir),
            *_input_arguments(),
        ]
    if step.number == 6:
        output_dir = step_directory_for_specification(
            STEP_6_DIRECTORY_NAME,
            specification_path=SPECIFICATION_PATH,
        )
        step_5_dir = step_directory_for_specification(
            STEP_5_DIRECTORY_NAME,
            specification_path=SPECIFICATION_PATH,
        )
        return [
            "--specification",
            str(SPECIFICATION_PATH),
            "--selection-manifest",
            str(step_5_dir / "selection_manifest.json"),
            "--selected-configurations",
            str(step_5_dir / "selected_configurations.csv"),
            "--output-dir",
            str(output_dir),
            *_input_arguments(),
        ]
    raise Phase2PipelineError(
        f"Parallel study arguments are only supported for Steps 5 and 6, "
        f"not Step {step.number}"
    )


def _interleave_by_family(
    pairs: list[tuple[str, int]],
) -> list[tuple[str, int]]:
    """Order studies round-robin across families instead of family by family.

    Callers build pairs family-major, and a FIFO pool therefore keeps a
    contiguous window of that list in flight: all five folds of one family,
    plus one fold of the next. Families are not interchangeable in what they
    load. The sequence architectures place their tensors on CUDA whenever it
    is available, while the tree families never leave the CPU, so a
    family-major window saturates one device and leaves the other idle for
    however long that family takes. Round-robin spreads the in-flight window
    across as many distinct families as there are workers, which overlaps
    CPU-bound and GPU-bound studies for the whole of Steps 5 and 6.

    This is an execution concern only, in the same sense as
    "[execution].max_workers": it cannot change results. Every study is an
    independent subprocess with its own sampler seeded from the settings, and
    consolidation reads finished studies back from disk in a fixed order, so
    the order studies are started in is not observable in any artifact.
    """

    remaining: dict[str, list[tuple[str, int]]] = {}
    for pair in pairs:
        remaining.setdefault(pair[0], []).append(pair)
    ordered: list[tuple[str, int]] = []
    while remaining:
        for family in list(remaining):
            ordered.append(remaining[family].pop(0))
            if not remaining[family]:
                del remaining[family]
    return ordered


def _run_pairs_in_parallel(
    *,
    step: StepDefinition,
    pairs: list[tuple[str, int]],
    max_workers: int,
    describe: Callable[[tuple[str, int]], str],
    initial_counts: dict[str, int] | None = None,
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

    # Track completions per family and pre-log so the dashboard starts immediately
    if initial_counts is None:
        completed_counts = {family: 0 for family, _ in pairs}
    else:
        completed_counts = initial_counts.copy()
        
    for family in completed_counts:
        log_global_progress(step.number, family, completed_counts[family])

    def _run_one(family: str, outer_fold: int) -> None:
        if stop_after_failure.is_set():
            raise _StudySkipped(family, outer_fold)
        _run_script(
            step,
            [
                "--family",
                family,
                "--outer-fold",
                str(outer_fold),
                *_study_arguments(step),
            ],
            env=environment,
        )

    first_error: Phase2PipelineError | None = None
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_run_one, family, outer_fold): (family, outer_fold)
            for family, outer_fold in _interleave_by_family(pairs)
        }
        for future in as_completed(futures):
            pair = futures[future]
            try:
                future.result()
                
                # On success, increment and log progress
                family = pair[0]
                completed_counts[family] += 1
                log_global_progress(step.number, family, completed_counts[family])
                
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


def _run_step_3() -> None:
    """Build fixed windows first, then the shared variable-length interface."""

    _run_script(STEPS[3])
    _run_script(TRAJECTORY_STEP)


def _settings_version_matches(
    artifact: dict[str, Any] | None,
    settings_version: int,
) -> bool:
    """Treat artifacts from another settings version as incomplete work."""

    return (
        artifact is not None
        and artifact.get("settings_version") == settings_version
    )


def _run_step_5(*, force: bool, max_workers: int | None) -> None:
    """Run only missing family/fold tuning studies unless forced to rerun all."""

    settings = _load_settings()
    settings_version = int(settings["settings_version"])
    resolved_max_workers = _resolve_max_workers(settings, max_workers)
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
        _step_5_manifest_path(),
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
    
    initial_counts = {family: 0 for family in families}
    pending_pair_set = set(pending_pairs)
    expected_pairs = [(f, o) for f in families for o in outer_folds]
    for pair in expected_pairs:
        if pair not in pending_pair_set:
            initial_counts[pair[0]] += 1

    if not pending_pairs:
        print("Step 5: every family/fold study is already complete; skipping", flush=True)
    else:
        _run_pairs_in_parallel(
            step=STEPS[5],
            pairs=pending_pairs,
            max_workers=resolved_max_workers,
            describe=lambda pair: f"{pair[0]} outer fold {pair[1]}",
            initial_counts=initial_counts,
        )

    final_manifest = _read_json(
        _step_5_manifest_path(),
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


def _run_step_6(*, force: bool, max_workers: int | None) -> None:
    """Run missing complete family/fold evaluations and preserve finished ones."""

    settings = _load_settings()
    settings_version = int(settings["settings_version"])
    resolved_max_workers = _resolve_max_workers(settings, max_workers)
    expected_by_pair, expected_run_count = _step_6_expected_runs(settings)
    manifest = _read_optional_json(
        _step_6_manifest_path(),
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
    
    # We can compute initial_counts from expected_by_pair using the same logic
    # _enabled_families(settings) was used to build expected_by_pair
    families = _enabled_families(settings)
    initial_counts = {family: 0 for family in families}
    pending_pair_set = set(pending_pairs)
    for pair in expected_by_pair:
        if pair not in pending_pair_set:
            initial_counts[pair[0]] += 1

    if not pending_pairs:
        print("Step 6: every family/fold evaluation is already complete; skipping", flush=True)
    else:
        _run_pairs_in_parallel(
            step=STEPS[6],
            pairs=pending_pairs,
            max_workers=resolved_max_workers,
            describe=lambda pair: f"{pair[0]} outer fold {pair[1]}",
            initial_counts=initial_counts,
        )

    final_manifest = _read_json(
        _step_6_manifest_path(),
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
    output_dir = step_directory_for_specification(
        STEP_7_DIRECTORY_NAME,
        specification_path=SPECIFICATION_PATH,
    )
    _run_script(
        STEPS[7],
        [
            "--specification",
            str(SPECIFICATION_PATH),
            "--locked-manifest",
            str(_step_6_manifest_path()),
            "--output-dir",
            str(output_dir),
        ],
    )
    manifest = _read_json(
        _step_7_manifest_path(),
        "Step 7 comparison manifest",
    )
    if manifest.get("status") != "complete":
        raise Phase2PipelineError("Step 7 comparison manifest is not complete")


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
        "3b. Trajectory data adapter: "
        + _simple_status(
            TRAJECTORY_REPORT_PATH,
            "Step 3 trajectory verification report",
        )
    )
    print(
        "4. Model adapters: "
        + ("available" if MODEL_REGISTRY_PATH.is_file() else "not generated")
    )

    # Steps 5 to 7 live inside a numbered run folder, so their state can only be
    # reported once Step 1 has produced the settings that name the run.
    try:
        run_number = read_run_number(SPECIFICATION_PATH)
    except RunLayoutError as error:
        print(f"Run: unknown ({error})")
        for line in (
            "5. Inner model selection: ",
            "6. Locked outer evaluation: ",
            "7. Architecture comparison: ",
        ):
            print(line + "unknown until Step 1 has run")
        return
    resolved_run_root = run_root_for_specification(
        specification_path=SPECIFICATION_PATH,
    )
    print(f"Run: {run_number} ({resolved_run_root})")
    print(
        "TensorBoard logs: "
        f"{tensorboard_log_root_for_specification(specification_path=SPECIFICATION_PATH)}"
    )
    print(
        "5. Inner model selection: "
        + _progress_status(
            _step_5_manifest_path(),
            "Step 5 selection manifest",
            "completed_study_count",
            "expected_study_count",
        )
    )
    print(
        "6. Locked outer evaluation: "
        + _progress_status(
            _step_6_manifest_path(),
            "Step 6 locked-evaluation manifest",
            "completed_run_count",
            "expected_run_count",
        )
    )
    print(
        "7. Architecture comparison: "
        + _simple_status(_step_7_manifest_path(), "Step 7 comparison manifest")
    )


def run_pipeline(
    first_step: int,
    last_step: int,
    *,
    force: bool,
    max_workers: int | None,
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
    print(f"TensorBoard {tensorboard_version} available", flush=True)
    if step_5_fit_curves_enabled():
        print(
            f"{FIT_CURVE_ENVIRONMENT_VARIABLE} is set: Step 5 inner fits will "
            "write per-epoch curves",
            flush=True,
        )
    # Steps 1 to 4 rebuild the settings that name the run, so the folder can
    # only be reported once those steps are not part of this request.
    if first_step > 1:
        try:
            run_number = read_run_number(SPECIFICATION_PATH)
        except RunLayoutError as error:
            raise Phase2PipelineError(str(error)) from error
        print(
            "Run "
            f"{run_number}: Steps 5-7 read and write "
            f"{run_root_for_specification(specification_path=SPECIFICATION_PATH)}",
            flush=True,
        )
        print(
            "TensorBoard logs: "
            f"{tensorboard_log_root_for_specification(specification_path=SPECIFICATION_PATH)}",
            flush=True,
        )
    if force:
        print(
            "Force mode is enabled for the expensive Step 5 and Step 6 work",
            flush=True,
        )
    if 5 in requested_steps or 6 in requested_steps:
        if max_workers is None:
            workers_description = (
                "up to [execution].max_workers from the settings (resolved "
                "once Step 1 has run)"
            )
        else:
            workers_description = f"up to {max_workers} at a time (--max-workers)"
        print(
            f"Step 5/6 studies run {workers_description}; output from "
            "concurrent studies interleaves in this console, but each "
            "study's own checkpoint files stay separate",
            flush=True,
        )

    for step_number in requested_steps:
        if step_number == 3:
            _run_step_3()
        elif step_number <= 4:
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
        "--specification",
        type=Path,
        default=SPECIFICATION_PATH,
        help="Location of Step 1's generated experiment specification.",
    )
    parser.add_argument(
        "--tabular-manifest",
        type=Path,
        default=None,
        help="Optional experiment-local Step 2 tabular adapter manifest.",
    )
    parser.add_argument(
        "--sequence-manifest",
        type=Path,
        default=None,
        help="Optional experiment-local Step 3 sequence adapter manifest.",
    )
    parser.add_argument(
        "--trajectory-manifest",
        type=Path,
        default=None,
        help="Optional experiment-local Step 3 trajectory adapter manifest.",
    )
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=MODEL_REGISTRY_PATH,
        help="Optional experiment-local Step 4 model registry.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help=(
            "Optional root for Steps 5-7 and TensorBoard artifacts. Omit it "
            "to use the standalone runs/run_<run_number> layout."
        ),
    )
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
        default=None,
        help=(
            "Maximum number of independent Step 5/6 family/outer-fold "
            "studies to run at the same time, each as its own subprocess. "
            "Defaults to [execution].max_workers in the settings TOML "
            "(itself normally \"auto\", meaning every available CPU core). "
            "Passing this flag overrides the settings for this run only, "
            "without editing the TOML. Use 1 to run sequentially."
        ),
    )
    args = parser.parse_args()
    if args.max_workers is not None and args.max_workers < 1:
        parser.error("--max-workers must be at least 1")

    try:
        _configure_invocation(
            specification_path=args.specification,
            tabular_manifest_path=args.tabular_manifest,
            sequence_manifest_path=args.sequence_manifest,
            trajectory_manifest_path=args.trajectory_manifest,
            model_registry_path=args.model_registry,
            run_root_path=args.run_root,
        )
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
