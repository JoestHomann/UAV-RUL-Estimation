"""Open Step 7 only when Step 6 contains a complete locked evaluation.

The gate reads metadata before any locked prediction table is opened. It makes
the comparison stage fail closed when Step 6 is missing, partial, inconsistent
with the architecture study settings, or reports prohibited test-data access.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any


STEP_DIR = Path(__file__).resolve().parent
PHASE_DIR = STEP_DIR.parent
if str(PHASE_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE_DIR))

from run_layout import (  # noqa: E402
    STEP_6_DIRECTORY_NAME,
    step_directory_for_specification,
)

DEFAULT_SPECIFICATION_PATH = (
    PHASE_DIR
    / "1_architecture_study_settings"
    / "artifacts"
    / "experiment_specification.json"
)


def default_locked_manifest_path() -> Path:
    """Locate Step 6's manifest inside the current run folder.

    Resolved when it is called rather than at import time, so importing this
    gate does not require Step 1 to have run yet.
    """

    return (
        step_directory_for_specification(STEP_6_DIRECTORY_NAME)
        / "locked_evaluation_manifest.json"
    )


# These families are retrained with all configured seeds in Step 6. The other
# implemented families are deterministic and therefore have only the first
# configured seed. Keeping this rule explicit mirrors the model adapters while
# avoiding the expensive model-library imports in this metadata-only gate.
STOCHASTIC_FAMILIES = {
    "random_forest",
    "extra_trees",
    "xgboost",
    "catboost",
    "mlp",
    "tcn",
    "multiscale_cnn",
    "sensor_graph_tcn",
    "lstm",
    "transformer",
}


class ArchitectureComparisonGateError(ValueError):
    """Explain why locked architecture comparison must not begin."""


@dataclass(frozen=True)
class ArchitectureComparisonPlan:
    """Store validated inputs and fixed reporting choices for Step 7."""

    settings: dict[str, Any]
    enabled_families: tuple[str, ...]
    outer_fold_labels: tuple[int, ...]
    retraining_seeds: tuple[int, ...]
    seeds_by_family: dict[str, tuple[int, ...]]
    expected_run_count: int
    expected_prediction_rows: int
    predictions_path: Path
    model_runs_path: Path


def _read_json(path: Path, description: str) -> dict[str, Any]:
    """Read one required JSON object with a prerequisite-focused error."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArchitectureComparisonGateError(
            f"Cannot read {description} at {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ArchitectureComparisonGateError(
            f"{description} must contain a JSON object"
        )
    return value


def _enabled_families(settings: dict[str, Any]) -> tuple[str, ...]:
    """Return enabled families in the predeclared, non-performance order."""

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


def _artifact_path(
    manifest_path: Path,
    manifest: dict[str, Any],
    artifact_name: str,
) -> Path:
    """Resolve one Step 6 artifact without allowing directory traversal."""

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ArchitectureComparisonGateError(
            "Step 6 manifest has no readable artifacts object"
        )
    relative_value = artifacts.get(artifact_name)
    if not isinstance(relative_value, str) or not relative_value.strip():
        raise ArchitectureComparisonGateError(
            f"Step 6 manifest does not name artifact {artifact_name!r}"
        )

    artifact_root = manifest_path.resolve().parent
    candidate = (artifact_root / relative_value).resolve()
    if candidate != artifact_root and artifact_root not in candidate.parents:
        raise ArchitectureComparisonGateError(
            f"Step 6 artifact {artifact_name!r} resolves outside its artifact folder"
        )
    if not candidate.is_file():
        raise ArchitectureComparisonGateError(
            f"Step 6 artifact {artifact_name!r} is missing at {candidate}"
        )
    return candidate


def build_architecture_comparison_plan(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    locked_manifest_path: Path | None = None,
) -> ArchitectureComparisonPlan:
    """Validate the complete Step 6 manifest before returning data paths."""

    if locked_manifest_path is None:
        locked_manifest_path = default_locked_manifest_path()
    specification = _read_json(
        specification_path,
        "Step 1 experiment specification",
    )
    settings = specification.get("settings")
    if not isinstance(settings, dict):
        raise ArchitectureComparisonGateError(
            "Step 1 experiment specification has no settings object"
        )

    # The manifest is the only Step 6 file read before the completion decision.
    # A missing or partial manifest therefore prevents locked predictions from
    # entering memory, even if a stale prediction CSV happens to exist locally.
    manifest = _read_json(
        locked_manifest_path,
        "Step 6 locked-evaluation manifest",
    )
    if manifest.get("status") != "complete":
        completed = manifest.get("completed_run_count", 0)
        expected = manifest.get("expected_run_count", "unknown")
        raise ArchitectureComparisonGateError(
            "Step 6 is incomplete: "
            f"completed {completed}/{expected} family/fold/seed runs"
        )

    settings_version = int(settings["settings_version"])
    if manifest.get("settings_version") != settings_version:
        raise ArchitectureComparisonGateError(
            "Step 1 and Step 6 use different settings versions"
        )
    required_manifest_values = {
        "step_5_prerequisite": "complete",
        "locked_results_used_for_tuning": False,
        "fixed_training_duration_from_step_5": True,
        "locked_data_loaded": True,
        "test_data_loaded": False,
    }
    for key, expected_value in required_manifest_values.items():
        if manifest.get(key) != expected_value:
            raise ArchitectureComparisonGateError(
                f"Step 6 manifest requires {key}={expected_value!r}"
            )

    families = _enabled_families(settings)
    fold_count = int(settings["phase_1"]["expected_outer_folds"])
    outer_folds = tuple(range(fold_count))
    retraining_seeds = tuple(int(seed) for seed in settings["tuning"]["retraining_seeds"])
    if not retraining_seeds:
        raise ArchitectureComparisonGateError(
            "The architecture study settings do not define retraining seeds"
        )
    seeds_by_family = {
        family: (
            retraining_seeds
            if family in STOCHASTIC_FAMILIES
            else (retraining_seeds[0],)
        )
        for family in families
    }
    expected_runs = sum(
        len(seeds_by_family[family]) * len(outer_folds)
        for family in families
    )
    validation_uavs = int(settings["phase_1"]["expected_training_uavs"]) // fold_count
    locked_scenarios = int(settings["phase_1"]["expected_locked_scenarios"])
    rows_per_run = validation_uavs * locked_scenarios
    expected_prediction_rows = expected_runs * rows_per_run

    exact_manifest_values = {
        "enabled_families": list(families),
        "outer_fold_labels": list(outer_folds),
        "retraining_seeds": list(retraining_seeds),
        "expected_run_count": expected_runs,
        "completed_run_count": expected_runs,
        "expected_prediction_rows": expected_prediction_rows,
        "prediction_rows": expected_prediction_rows,
    }
    for key, expected_value in exact_manifest_values.items():
        if manifest.get(key) != expected_value:
            raise ArchitectureComparisonGateError(
                f"Step 6 manifest has inconsistent {key}: "
                f"expected {expected_value!r}, got {manifest.get(key)!r}"
            )
    completed_runs = manifest.get("completed_runs")
    incomplete_runs = manifest.get("incomplete_runs")
    if not isinstance(completed_runs, list) or len(completed_runs) != expected_runs:
        raise ArchitectureComparisonGateError(
            "Step 6 manifest does not list every completed run"
        )
    if incomplete_runs != []:
        raise ArchitectureComparisonGateError(
            "Step 6 manifest still lists incomplete runs"
        )

    predictions_path = _artifact_path(
        locked_manifest_path,
        manifest,
        "locked_predictions",
    )
    model_runs_path = _artifact_path(
        locked_manifest_path,
        manifest,
        "model_runs",
    )
    return ArchitectureComparisonPlan(
        settings=settings,
        enabled_families=families,
        outer_fold_labels=outer_folds,
        retraining_seeds=retraining_seeds,
        seeds_by_family=seeds_by_family,
        expected_run_count=expected_runs,
        expected_prediction_rows=expected_prediction_rows,
        predictions_path=predictions_path,
        model_runs_path=model_runs_path,
    )
