"""Build the traceable input boundary for raw telemetry sequence models.

The builder reads Step 1's resolved experiment specification, copies the eight
raw-data, cutoff, and fold files needed by sequence models, and writes a compact
manifest. Copies preserve source modification times and are compared directly
with their sources after every build. No hashes or materialized sequence arrays
are created.

Only generated files in this step's artifact directory are replaced. Raw data
and Phase 1 artifacts are never modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import filecmp
import json
from pathlib import Path
import shutil
import sys
from typing import Any


STEP_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STEP_DIR.parent.parent
DEFAULT_SPECIFICATION_PATH = (
    STEP_DIR.parent
    / "1_architecture_study_settings"
    / "artifacts"
    / "experiment_specification.json"
)
DEFAULT_OUTPUT_DIR = STEP_DIR / "artifacts"
MANIFEST_FILENAME = "sequence_dataset_manifest.json"
REPORT_FILENAME = "copy_verification.json"

# The raw CSV files are repository inputs rather than generated Phase 1
# artifacts. Their fixed repository-relative paths are therefore declared here.
RAW_TRAIN_PATH = "data/train.csv"
RAW_TEST_PATH = "data/test.csv"
TEST_ENDPOINTS_PATH = (
    "1_dataset_construction/3_test_like_validation_scenarios/"
    "artifacts/test_endpoints.csv"
)


class SequenceBuildError(ValueError):
    """Represent a readable specification, copying, or verification failure."""


# A plan entry contains an optional Step 1 artifact key, a fixed fallback path,
# and the unchanged destination filename. Artifact keys are preferred whenever
# the experiment specification already declares the source.
COPY_PLAN: dict[str, dict[str, str | None]] = {
    "raw_train": {
        "artifact_key": None,
        "source_path": RAW_TRAIN_PATH,
        "destination": "train.csv",
    },
    "raw_test": {
        "artifact_key": None,
        "source_path": RAW_TEST_PATH,
        "destination": "test.csv",
    },
    "training_endpoints": {
        "artifact_key": "training_prefixes",
        "source_path": None,
        "destination": "training_prefixes.csv",
    },
    "development_endpoints": {
        "artifact_key": "development_scenarios",
        "source_path": None,
        "destination": "development_validation_scenarios.csv",
    },
    "locked_endpoints": {
        "artifact_key": "locked_scenarios",
        "source_path": None,
        "destination": "locked_validation_scenarios.csv",
    },
    "test_endpoints": {
        "artifact_key": None,
        "source_path": TEST_ENDPOINTS_PATH,
        "destination": "test_endpoints.csv",
    },
    "outer_folds": {
        "artifact_key": "outer_folds",
        "source_path": None,
        "destination": "outer_folds.csv",
    },
    "inner_folds": {
        "artifact_key": "inner_folds",
        "source_path": None,
        "destination": "inner_folds.csv",
    },
}


# Endpoint interfaces describe which columns become metadata, targets, and
# sample weights in the runtime SequenceDataset objects.
ENDPOINT_INTERFACES: dict[str, dict[str, Any]] = {
    "training_endpoints": {
        "role": "model_fitting_prefixes",
        "history_file": "raw_train",
        "metadata_columns": [
            "sample_id",
            "uav_id",
            "prefix_number",
            "cutoff",
        ],
        "target_column": "RUL",
        "sample_weight_column": "sample_weight",
    },
    "development_endpoints": {
        "role": "inner_selection_scenarios",
        "history_file": "raw_train",
        "metadata_columns": [
            "sample_id",
            "scenario",
            "outer_fold",
            "uav_id",
            "cutoff",
            "terminal_lifetime",
            "lifetime_quantile",
        ],
        "target_column": "RUL",
        "sample_weight_column": None,
    },
    "locked_endpoints": {
        "role": "locked_outer_evaluation_scenarios",
        "history_file": "raw_train",
        "metadata_columns": [
            "sample_id",
            "scenario",
            "outer_fold",
            "uav_id",
            "cutoff",
            "terminal_lifetime",
            "lifetime_quantile",
        ],
        "target_column": "RUL",
        "sample_weight_column": None,
    },
    "test_endpoints": {
        "role": "final_unlabelled_test_endpoints",
        "history_file": "raw_test",
        "metadata_columns": ["sample_id", "uav_id", "cutoff"],
        "target_column": None,
        "sample_weight_column": None,
    },
}


def _load_experiment_specification(path: Path) -> dict[str, Any]:
    """Read Step 1's JSON and require sequence and Phase 1 sections."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SequenceBuildError(
            "Cannot read the Step 1 experiment specification. Run the Step 1 "
            f"builder first. Details: {error}"
        ) from error

    try:
        settings = payload["settings"]
        representations = settings["representations"]
        phase_1_artifacts = settings["phase_1"]["artifacts"]
        settings_version = settings["settings_version"]
    except (KeyError, TypeError) as error:
        raise SequenceBuildError(
            f"Experiment specification is missing required field {error}"
        ) from error

    if not isinstance(representations, dict):
        raise SequenceBuildError("Sequence representation settings must be an object")
    if not isinstance(phase_1_artifacts, dict):
        raise SequenceBuildError("Phase 1 artifact settings must be an object")
    if not isinstance(settings_version, int):
        raise SequenceBuildError("Experiment settings version must be an integer")
    return payload


def _repository_relative(path: Path) -> str:
    """Return a portable path relative to the repository root."""

    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise SequenceBuildError(f"Path is outside the repository: {path}") from error


def _resolve_source(relative_path: str) -> Path:
    """Resolve a source path and prevent absolute or parent-escaping inputs."""

    supplied = Path(relative_path)
    if supplied.is_absolute():
        raise SequenceBuildError(f"Source path must be relative: {relative_path}")
    resolved = (REPOSITORY_ROOT / supplied).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise SequenceBuildError(
            f"Source path escapes the repository: {relative_path}"
        ) from error
    if not resolved.is_file():
        raise SequenceBuildError(f"Required source does not exist: {resolved}")
    return resolved


def _source_label(
    specification: dict[str, Any],
    plan_entry: dict[str, str | None],
) -> tuple[str, int | None]:
    """Resolve one source label and its optional expected row count."""

    artifact_key = plan_entry["artifact_key"]
    fixed_path = plan_entry["source_path"]
    if artifact_key is None:
        if not isinstance(fixed_path, str):
            raise SequenceBuildError("Fixed copy-plan entry has no source path")
        return fixed_path, None

    try:
        artifact = specification["settings"]["phase_1"]["artifacts"][artifact_key]
        source_path = artifact["path"]
        expected_rows = artifact.get("rows")
    except (KeyError, TypeError) as error:
        raise SequenceBuildError(
            f"Experiment specification does not define artifact {artifact_key!r}"
        ) from error
    if not isinstance(source_path, str):
        raise SequenceBuildError(f"Artifact {artifact_key!r} has no valid path")
    return source_path, expected_rows


def _copy_inputs(
    specification: dict[str, Any],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Refresh the eight local input copies and prepare manifest entries."""

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, Any]] = {}

    for logical_name, plan_entry in COPY_PLAN.items():
        source_label, expected_rows = _source_label(specification, plan_entry)
        source_path = _resolve_source(source_label)
        destination_name = plan_entry["destination"]
        if not isinstance(destination_name, str):
            raise SequenceBuildError("Copy-plan destination must be text")
        copied_path = data_dir / destination_name

        try:
            # copy2 replaces only the generated destination and preserves the
            # source modification time used by the integrity report.
            shutil.copy2(source_path, copied_path)
        except OSError as error:
            raise SequenceBuildError(
                f"Cannot copy {source_path} to {copied_path}: {error}"
            ) from error

        entry: dict[str, Any] = {
            "source_path": source_label,
            "copied_path": f"data/{destination_name}",
            "expected_rows": expected_rows,
        }
        artifact_key = plan_entry["artifact_key"]
        if isinstance(artifact_key, str):
            entry["source_artifact_key"] = artifact_key

        if logical_name in ENDPOINT_INTERFACES:
            entry["category"] = "prediction_endpoints"
            entry.update(ENDPOINT_INTERFACES[logical_name])
        elif logical_name in {"raw_train", "raw_test"}:
            entry.update(
                {
                    "category": "telemetry_history",
                    "id_column": "uav_id",
                    "cycle_column": "flight_cycle",
                    "contains_target": logical_name == "raw_train",
                }
            )
        else:
            entry.update(
                {
                    "category": "fold_assignments",
                    "uav_id_column": "uav_id",
                    "fold_column": (
                        "outer_fold" if logical_name == "outer_folds" else "inner_fold"
                    ),
                }
            )
        entries[logical_name] = entry
    return entries


def _build_manifest(
    specification: dict[str, Any],
    specification_path: Path,
    files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Create the runtime sequence interface without materializing tensors."""

    settings = specification["settings"]
    representations = settings["representations"]
    return {
        "adapter_version": 1,
        "settings_version": settings["settings_version"],
        "experiment_specification": _repository_relative(specification_path),
        "channels": representations["sequence_channels"],
        "channel_count": representations["sequence_channel_count"],
        "lookbacks": representations["sequence_lookbacks"],
        "padding": {
            # "left" is the only padding side the settings schema ever allowed;
            # the setting itself was removed as dead configuration.
            "side": "left",
            "value": 0.0,
            "mask_dtype": "bool",
            "mask_true_means": "padding",
        },
        "side_features": representations["sequence_side_features"],
        "scaling": {
            # These were also single-option settings; the values are now fixed
            # here directly instead of being read back from the settings file.
            "method": "training_fold_channel_median_iqr",
            "fit_scope": "training_uavs_only",
            "center": "median",
            "scale": "IQR divided by 1.349",
            "fallback": "standard deviation, then 1.0",
            "relative_variation_tolerance": 1e-12,
            "padding_after_transform": 0.0,
            "side_features_scaled": False,
        },
        "files": files,
    }


def _artifact_relative_path(output_dir: Path, relative_path: str) -> Path:
    """Resolve one generated copy inside the selected artifact directory."""

    resolved = (output_dir.resolve() / relative_path).resolve()
    try:
        resolved.relative_to(output_dir.resolve())
    except ValueError as error:
        raise SequenceBuildError(
            f"Copied path escapes the output directory: {relative_path}"
        ) from error
    return resolved


def _readable_utc_timestamp(value: float) -> str:
    """Convert a filesystem modification time to readable UTC text."""

    return datetime.fromtimestamp(value, tz=UTC).isoformat()


def _verify_copies(
    files: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Compare every generated copy with its source and return a full report."""

    observations: dict[str, dict[str, Any]] = {}
    for logical_name, entry in files.items():
        source_label = entry["source_path"]
        copied_label = entry["copied_path"]
        source_path = _resolve_source(source_label)
        copied_path = _artifact_relative_path(output_dir, copied_label)

        source_exists = source_path.is_file()
        copied_exists = copied_path.is_file()
        source_stat = source_path.stat() if source_exists else None
        copied_stat = copied_path.stat() if copied_exists else None

        checks = {
            "source_exists": source_exists,
            "copy_exists": copied_exists,
            "size_matches": bool(
                source_stat
                and copied_stat
                and source_stat.st_size == copied_stat.st_size
            ),
            "timestamp_matches": bool(
                source_stat
                and copied_stat
                and source_stat.st_mtime_ns == copied_stat.st_mtime_ns
            ),
            "content_matches": bool(
                source_exists
                and copied_exists
                and filecmp.cmp(source_path, copied_path, shallow=False)
            ),
        }
        passed = all(checks.values())
        observations[logical_name] = {
            "source_path": source_label,
            "copied_path": copied_label,
            "source_size_bytes": source_stat.st_size if source_stat else None,
            "copied_size_bytes": copied_stat.st_size if copied_stat else None,
            "source_modified_time_ns": (
                source_stat.st_mtime_ns if source_stat else None
            ),
            "copied_modified_time_ns": (
                copied_stat.st_mtime_ns if copied_stat else None
            ),
            "source_modified_time_utc": (
                _readable_utc_timestamp(source_stat.st_mtime)
                if source_stat
                else None
            ),
            "copied_modified_time_utc": (
                _readable_utc_timestamp(copied_stat.st_mtime)
                if copied_stat
                else None
            ),
            "checks": checks,
            "passed": passed,
        }

    failed = sorted(
        name for name, observation in observations.items() if not observation["passed"]
    )
    return {
        "status": "failed" if failed else "passed",
        "checked_files": len(observations),
        "failed_files": failed,
        "files": observations,
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Write deterministic and human-readable JSON metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_sequence_data_adapter(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Refresh all Step 3 artifacts and fail if any copy differs."""

    specification = _load_experiment_specification(specification_path)
    files = _copy_inputs(specification, output_dir)
    manifest = _build_manifest(specification, specification_path, files)
    manifest_path = output_dir / MANIFEST_FILENAME
    report_path = output_dir / REPORT_FILENAME
    _write_json(manifest, manifest_path)

    report = _verify_copies(files, output_dir)
    _write_json(report, report_path)
    if report["status"] != "passed":
        raise SequenceBuildError(
            "Copied files differ from their sources: "
            + ", ".join(report["failed_files"])
        )
    return manifest_path, report_path


def main() -> None:
    """Build the sequence data adapter artifacts from the command line."""

    # The CLI changes locations only. Channels, lookbacks, padding, and scaling
    # are always taken from the resolved experiment specification.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specification",
        type=Path,
        default=DEFAULT_SPECIFICATION_PATH,
        help="Location of Step 1's generated experiment specification.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for copied inputs and generated metadata.",
    )
    args = parser.parse_args()

    try:
        manifest_path, report_path = build_sequence_data_adapter(
            args.specification,
            args.output_dir,
        )
    except SequenceBuildError as error:
        print(f"Sequence data adapter build failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("Sequence data adapter built successfully")
    print(f"Copied files: {len(COPY_PLAN)}")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Verification report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
