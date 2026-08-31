"""Load composable pipeline-experiment TOML definitions."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import tomllib
from typing import Any


class ExperimentConfigError(ValueError):
    """Explain an unreadable or invalid composed experiment definition."""


PIPELINE_OWNED_TABLES = (
    "run_definitions",
    "experiments",
    "experiment_groups",
    "experiment_workflows",
    "conditional_calibration_workflows",
)


def _directory_name(value: Any, *, field: str, source: Path) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ExperimentConfigError(
            f"{source}: pipeline.{field} must be one directory name"
        )
    return value


def _apply_pipeline_identity(
    payload: dict[str, Any],
    *,
    source: Path,
) -> dict[str, Any]:
    """Apply one PE_X/run_N identity to every definition owned by this file."""

    pipeline = payload.get("pipeline")
    if pipeline is None:
        return payload
    if not isinstance(pipeline, dict):
        raise ExperimentConfigError(f"{source}: pipeline must be a TOML table")
    experiment_name = _directory_name(
        pipeline.get("experiment"),
        field="experiment",
        source=source,
    )
    run_name = _directory_name(
        pipeline.get("run"),
        field="run",
        source=source,
    )
    if re.fullmatch(r"PE_[1-9]\d*", experiment_name) is None:
        raise ExperimentConfigError(
            f"{source}: pipeline.experiment must use the PE_<number> form"
        )
    if re.fullmatch(r"run_[1-9]\d*", run_name) is None:
        raise ExperimentConfigError(
            f"{source}: pipeline.run must use the run_<number> form"
        )
    resolved = copy.deepcopy(payload)
    for table_name in PIPELINE_OWNED_TABLES:
        table = resolved.get(table_name, {})
        if not isinstance(table, dict):
            raise ExperimentConfigError(
                f"{source}: {table_name} must be a TOML table"
            )
        for item_name, specification in table.items():
            if not isinstance(specification, dict):
                raise ExperimentConfigError(
                    f"{source}: {table_name}.{item_name} must be a TOML table"
                )
            specification["pipeline_experiment"] = experiment_name
            specification["pipeline_run"] = run_name
    return resolved


def _merge(base: dict[str, Any], addition: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in addition.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _merge(current, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _include_paths(value: Any, *, source: Path) -> list[Path]:
    if value is None:
        return []
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not all(
        isinstance(item, str) and item for item in values
    ):
        raise ExperimentConfigError(
            f"{source}: include must be a path or a list of paths"
        )
    return [(source.parent / item).resolve() for item in values]


def read_experiment_config(
    path: Path,
    *,
    _stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Read JSON or TOML, recursively merging TOML ``include`` files first."""

    source = path.resolve()
    if source in _stack:
        cycle = " -> ".join(str(item) for item in (*_stack, source))
        raise ExperimentConfigError(f"Experiment config include cycle: {cycle}")
    try:
        if source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
        else:
            with source.open("rb") as stream:
                payload = tomllib.load(stream)
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        raise ExperimentConfigError(
            f"Cannot read experiment config {source}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ExperimentConfigError(f"{source}: config must contain a table/object")
    if source.suffix.lower() == ".json":
        return payload

    local = _apply_pipeline_identity(dict(payload), source=source)
    includes = _include_paths(local.pop("include", None), source=source)
    merged: dict[str, Any] = {}
    for include_path in includes:
        merged = _merge(
            merged,
            read_experiment_config(include_path, _stack=(*_stack, source)),
        )
    return _merge(merged, local)
