"""Load composable pipeline-experiment TOML definitions."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tomllib
from typing import Any


class ExperimentConfigError(ValueError):
    """Explain an unreadable or invalid composed experiment definition."""


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

    local = dict(payload)
    includes = _include_paths(local.pop("include", None), source=source)
    merged: dict[str, Any] = {}
    for include_path in includes:
        merged = _merge(
            merged,
            read_experiment_config(include_path, _stack=(*_stack, source)),
        )
    return _merge(merged, local)
