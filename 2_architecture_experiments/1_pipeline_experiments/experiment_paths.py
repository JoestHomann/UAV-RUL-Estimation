"""Resolve pipeline-experiment, run, and sub-experiment artifact directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any


MOVED_REPOSITORY_PREFIXES = {
    "pipeline_experiments": Path(
        "2_architecture_experiments/1_pipeline_experiments"
    ),
    "2_model_architecture_study": Path(
        "2_architecture_experiments/2_model_architecture_study"
    ),
}


def repository_path(repository_root: Path, value: str) -> Path:
    """Resolve current paths and legacy paths recorded before the Phase 2 move."""

    supplied = Path(value)
    direct = (repository_root / supplied).resolve()
    if direct.exists() or not supplied.parts:
        return direct
    replacement = MOVED_REPOSITORY_PREFIXES.get(supplied.parts[0])
    if replacement is None:
        return direct
    return (repository_root / replacement.joinpath(*supplied.parts[1:])).resolve()


def _directory_name(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{field} must be one directory name")
    return value


def pipeline_experiment_name(
    specification: dict[str, Any] | None = None,
) -> str | None:
    """Return the PE_X owner, or None for legacy/test specifications."""

    if specification is None or "pipeline_experiment" not in specification:
        return None
    return _directory_name(
        specification["pipeline_experiment"],
        field="pipeline_experiment",
    )


def pipeline_run_name(name: str, specification: dict[str, Any] | None = None) -> str:
    value = name if specification is None else specification.get("pipeline_run", name)
    return _directory_name(value, field="pipeline_run")


def pipeline_owner(
    name: str,
    specification: dict[str, Any] | None = None,
) -> tuple[str | None, str]:
    """Return the collision-free (PE_X, run_N) artifact owner."""

    return (
        pipeline_experiment_name(specification),
        pipeline_run_name(name, specification),
    )


def run_directory(
    experiments_dir: Path,
    name: str,
    specification: dict[str, Any] | None = None,
) -> Path:
    """Return the root directory for one execution of a pipeline experiment."""

    pipeline_experiment, run_name = pipeline_owner(name, specification)
    if pipeline_experiment is None:
        return experiments_dir / run_name
    return experiments_dir / pipeline_experiment / "runs" / run_name


def artifact_directory(
    experiments_dir: Path,
    name: str,
    specification: dict[str, Any] | None = None,
) -> Path:
    owner_dir = run_directory(experiments_dir, name, specification)
    if specification is not None and specification.get("artifact_root") is True:
        return owner_dir
    owner = pipeline_run_name(name, specification)
    return owner_dir if owner == name else owner_dir / name


def gallery_directory(
    experiments_dir: Path,
    name: str,
    specification: dict[str, Any] | None = None,
) -> Path:
    return run_directory(experiments_dir, name, specification) / "figures"
