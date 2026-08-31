"""Resolve parent pipeline-run and sub-experiment artifact directories."""

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


def pipeline_run_name(name: str, specification: dict[str, Any] | None = None) -> str:
    value = name if specification is None else specification.get("pipeline_run", name)
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError("pipeline_run must be one directory name")
    return value


def artifact_directory(
    runs_dir: Path,
    name: str,
    specification: dict[str, Any] | None = None,
) -> Path:
    owner = pipeline_run_name(name, specification)
    owner_dir = runs_dir / owner
    return owner_dir if owner == name else owner_dir / name


def gallery_directory(
    runs_dir: Path,
    name: str,
    specification: dict[str, Any] | None = None,
) -> Path:
    owner = pipeline_run_name(name, specification)
    return runs_dir / owner / "figures"
