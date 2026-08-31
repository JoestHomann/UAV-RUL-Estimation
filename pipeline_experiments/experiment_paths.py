"""Resolve parent pipeline-run and sub-experiment artifact directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
