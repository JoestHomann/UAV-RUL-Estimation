"""Run and resume the complete seven-step Phase 2 architecture study.

This file is the single user-facing entry point for Phase 2. It delegates each
stage to the existing step runner with the same Python interpreter that started
this process. Scientific logic remains inside the step modules, while this file
provides ordering, progress reporting, safe resume behavior, and fail-fast
subprocess handling.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PHASE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PHASE_DIR.parent

SPECIFICATION_PATH = (
    PHASE_DIR
    / "1_experiment_contract"
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


@dataclass(frozen=True)
class StepDefinition:
    """Describe one delegated Phase 2 step without duplicating its behavior."""

    number: int
    name: str
    script: Path


STEPS = {
    1: StepDefinition(
        1,
        "Experiment contract",
        PHASE_DIR / "1_experiment_contract" / "build_experiment_contract.py",
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


def _load_contract() -> dict[str, Any]:
    """Return the resolved contract produced by Step 1."""

    specification = _read_json(
        SPECIFICATION_PATH,
        "Step 1 experiment specification",
    )
    contract = specification.get("contract")
    if not isinstance(contract, dict):
        raise Phase2PipelineError(
            "Step 1 experiment specification has no contract object"
        )
    return contract


def _enabled_families(contract: dict[str, Any]) -> tuple[str, ...]:
    """Preserve the family order declared before performance was observed."""

    study = contract["study"]
    declared_order = (
        study["required_architectures"]
        + study["conditional_architectures"]
        + study["optional_architectures"]
    )
    return tuple(
        family
        for family in declared_order
        if contract["architectures"][family]["enabled"]
    )


def _command_text(command: list[str]) -> str:
    """Format one command for readable Windows terminal output."""

    return subprocess.list2cmdline(command)


def _run_script(step: StepDefinition, extra_arguments: list[str] | None = None) -> None:
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
    )
    if completed.returncode != 0:
        raise Phase2PipelineError(
            f"Step {step.number} failed with exit code {completed.returncode}"
        )


def _contract_version_matches(
    artifact: dict[str, Any] | None,
    contract_version: int,
) -> bool:
    """Treat artifacts from another contract version as incomplete work."""

    return (
        artifact is not None
        and artifact.get("contract_version") == contract_version
    )


def _run_step_5(*, force: bool) -> None:
    """Run only missing family/fold tuning studies unless forced to rerun all."""

    contract = _load_contract()
    contract_version = int(contract["contract_version"])
    families = _enabled_families(contract)
    outer_folds = tuple(
        range(int(contract["phase_1"]["expected_outer_folds"]))
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
    if not force and _contract_version_matches(manifest, contract_version):
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

    for family in families:
        missing_folds = [
            fold
            for fold in outer_folds
            if force
            or f"{family}__outer_{fold:02d}" not in completed_studies
        ]
        if not missing_folds:
            print(f"Step 5: {family} is already complete; skipping", flush=True)
            continue
        arguments = ["--family", family]
        for outer_fold in missing_folds:
            arguments.extend(["--outer-fold", str(outer_fold)])
        _run_script(STEPS[5], arguments)

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
    contract: dict[str, Any],
) -> tuple[dict[tuple[str, int], set[str]], int]:
    """Derive complete family/fold run sets from the contract and registry."""

    registry = _read_json(MODEL_REGISTRY_PATH, "Step 4 model registry")
    contract_version = int(contract["contract_version"])
    if registry.get("contract_version") != contract_version:
        raise Phase2PipelineError(
            "Step 4 model registry uses a different contract version"
        )
    registry_families = registry.get("families")
    if not isinstance(registry_families, dict):
        raise Phase2PipelineError("Step 4 model registry has no families object")

    families = _enabled_families(contract)
    outer_folds = tuple(
        range(int(contract["phase_1"]["expected_outer_folds"]))
    )
    retraining_seeds = tuple(
        int(seed) for seed in contract["tuning"]["retraining_seeds"]
    )
    if not retraining_seeds:
        raise Phase2PipelineError("The contract does not define retraining seeds")

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


def _run_step_6(*, force: bool) -> None:
    """Run missing complete family/fold evaluations and preserve finished ones."""

    contract = _load_contract()
    contract_version = int(contract["contract_version"])
    expected_by_pair, expected_run_count = _step_6_expected_runs(contract)
    manifest = _read_optional_json(
        STEP_6_MANIFEST_PATH,
        "Step 6 locked-evaluation manifest",
    )
    completed_runs: set[str] = set()
    if not force and _contract_version_matches(manifest, contract_version):
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

    families = _enabled_families(contract)
    for family in families:
        family_pairs = [
            (pair, required_runs)
            for pair, required_runs in expected_by_pair.items()
            if pair[0] == family
        ]
        missing_folds = [
            outer_fold
            for (_, outer_fold), required_runs in family_pairs
            if force or not required_runs.issubset(completed_runs)
        ]
        if not missing_folds:
            print(f"Step 6: {family} is already complete; skipping", flush=True)
            continue
        arguments = ["--family", family]
        for outer_fold in missing_folds:
            arguments.extend(["--outer-fold", str(outer_fold)])
        _run_script(STEPS[6], arguments)

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
    print(
        "1. Experiment contract: "
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


def run_pipeline(first_step: int, last_step: int, *, force: bool) -> None:
    """Execute a validated inclusive step range in dependency order."""

    if first_step > last_step:
        raise Phase2PipelineError(
            "--from-step cannot be greater than --through-step"
        )
    requested_steps = range(first_step, last_step + 1)
    print(
        f"Running Phase 2 Steps {first_step}-{last_step} with "
        f"{sys.executable}",
        flush=True,
    )
    if force:
        print(
            "Force mode is enabled for the expensive Step 5 and Step 6 work",
            flush=True,
        )

    for step_number in requested_steps:
        if step_number <= 4:
            _run_script(STEPS[step_number])
        elif step_number == 5:
            _run_step_5(force=force)
        elif step_number == 6:
            _run_step_6(force=force)
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
    args = parser.parse_args()

    try:
        if args.status:
            print_status()
            return
        run_pipeline(args.from_step, args.through_step, force=args.force)
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
