"""Shared paths, validation helpers, and atomic artifact writers for Phase 3."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import sleep
import tomllib
from typing import Any, Iterable
from uuid import uuid4

import pandas as pd

from phase_3_run_layout import PHASE_DIR, read_run_number, run_root, step_directory


REPOSITORY_ROOT = PHASE_DIR.parent
PHASE_2_DIR = REPOSITORY_ROOT / "2_model_architecture_study"
PHASE_2_SPECIFICATION_PATH = (
    PHASE_2_DIR
    / "1_architecture_study_settings"
    / "artifacts"
    / "experiment_specification.json"
)
PHASE_2_MODEL_REGISTRY_PATH = (
    PHASE_2_DIR / "4_model_adapters" / "artifacts" / "model_registry.json"
)

STEP_MANIFEST_NAMES = {
    1: "architecture_selection_manifest.json",
    2: "final_search_manifest.json",
    3: "final_training_contract_manifest.json",
    4: "final_training_manifest.json",
    5: "inference_manifest.json",
    6: "submission_manifest.json",
}


class Phase3Error(ValueError):
    """Represent an invalid prerequisite, artifact, or pipeline operation."""


def read_json(path: Path, description: str) -> dict[str, Any]:
    """Read one required JSON object with a role-specific error."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase3Error(f"Cannot read {description} at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise Phase3Error(f"{description} must contain a JSON object")
    return payload


def read_optional_json(path: Path, description: str) -> dict[str, Any] | None:
    """Read an optional JSON object while rejecting malformed files."""

    if not path.is_file():
        return None
    return read_json(path, description)


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")


def _replace_with_retry(temporary: Path, destination: Path) -> None:
    """Atomically replace a file while tolerating brief Windows reader locks."""

    for attempt in range(6):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            sleep(0.05 * (attempt + 1))


def atomic_replace(temporary: Path, destination: Path) -> None:
    """Expose the shared atomic replacement for non-JSON model artifacts."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    _replace_with_retry(temporary, destination)


def write_json(payload: dict[str, Any], path: Path) -> Path:
    """Write deterministic JSON through an atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_csv(
    records: Iterable[dict[str, Any]] | pd.DataFrame,
    columns: list[str],
    path: Path,
    *,
    compression: str | None = None,
) -> Path:
    """Write a stable CSV table through an atomic replacement."""

    table = (
        records.loc[:, columns].copy()
        if isinstance(records, pd.DataFrame)
        else pd.DataFrame(list(records), columns=columns)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        table.to_csv(temporary, index=False, compression=compression)
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def repository_relative(path: Path) -> str:
    """Return a portable repository-relative path."""

    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise Phase3Error(f"Path is outside the repository: {path}") from error


def phase_2_run_root(run_number: int) -> Path:
    """Return a referenced Phase 2 run without creating it."""

    if isinstance(run_number, bool) or not isinstance(run_number, int) or run_number <= 0:
        raise Phase3Error("phase_2_run_number must be a positive integer")
    return PHASE_2_DIR / "runs" / f"run_{run_number}"


def phase_2_manifest_paths(run_number: int) -> dict[str, Path]:
    """Return the three completed Phase 2 manifests required by Step 1."""

    root = phase_2_run_root(run_number)
    return {
        "selection": root / "5_inner_model_selection" / "selection_manifest.json",
        "locked_evaluation": root
        / "6_locked_outer_evaluation"
        / "locked_evaluation_manifest.json",
        "comparison": root
        / "7_architecture_comparison"
        / "comparison_manifest.json",
    }


def artifacts_directory(step_number: int, run_number: int) -> Path:
    """Return one run-local generated-artifact directory."""

    return step_directory(step_number, run_number=run_number) / "artifacts"


def manifest_path(step_number: int, run_number: int) -> Path:
    """Return one run-local step manifest path."""

    return (
        step_directory(step_number, run_number=run_number)
        / STEP_MANIFEST_NAMES[step_number]
    )


def resolved_settings_path(run_number: int) -> Path:
    return artifacts_directory(1, run_number) / "resolved_phase_3_settings.json"


def selected_architecture_path(run_number: int) -> Path:
    return artifacts_directory(1, run_number) / "selected_architecture.json"


def selected_configuration_path(run_number: int) -> Path:
    return artifacts_directory(2, run_number) / "selected_configuration.json"


def training_contract_path(run_number: int) -> Path:
    return artifacts_directory(3, run_number) / "final_training_contract.json"


def final_model_path(run_number: int) -> Path:
    return artifacts_directory(4, run_number) / "final_model.joblib"


def final_preprocessor_path(run_number: int) -> Path:
    return artifacts_directory(4, run_number) / "final_preprocessor.joblib"


def test_predictions_path(run_number: int) -> Path:
    return artifacts_directory(5, run_number) / "test_predictions.csv"


def submission_path(run_number: int) -> Path:
    return artifacts_directory(6, run_number) / "submission.csv"


def load_resolved_phase_3_settings(run_number: int) -> dict[str, Any]:
    """Load the generated settings and return its strict settings object."""

    payload = read_json(
        resolved_settings_path(run_number),
        "resolved Phase 3 settings",
    )
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise Phase3Error("Resolved Phase 3 settings have no settings object")
    if settings.get("run_number") != run_number:
        raise Phase3Error("Resolved Phase 3 settings identify another run")
    return settings


def require_current_settings(settings_path: Path) -> int:
    """Require a downstream command to match the run's resolved TOML exactly."""

    run_number = read_run_number(settings_path)
    try:
        with settings_path.open("rb") as stream:
            current = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise Phase3Error(f"Cannot read Phase 3 settings {settings_path}: {error}") from error
    resolved = load_resolved_phase_3_settings(run_number)
    if current != resolved:
        raise Phase3Error(
            "Current Phase 3 TOML differs from this run's resolved settings; "
            "restore the settings or start a new run"
        )
    return run_number


def complete_manifest(
    step_number: int,
    run_number: int,
    settings_version: int,
) -> dict[str, Any] | None:
    """Return a compatible complete manifest, otherwise no completed work."""

    payload = read_optional_json(
        manifest_path(step_number, run_number),
        f"Phase 3 Step {step_number} manifest",
    )
    if (
        payload is not None
        and payload.get("status") == "complete"
        and payload.get("phase_3_run_number") == run_number
        and payload.get("settings_version") == settings_version
    ):
        return payload
    return None


def final_run_manifest_path(run_number: int) -> Path:
    return run_root(run_number) / "final_run_manifest.json"


def invalidate_downstream_manifests(after_step: int, run_number: int) -> None:
    """Remove completion claims derived from work that is being replaced."""

    for step_number in range(after_step + 1, 7):
        manifest_path(step_number, run_number).unlink(missing_ok=True)
    final_run_manifest_path(run_number).unlink(missing_ok=True)
