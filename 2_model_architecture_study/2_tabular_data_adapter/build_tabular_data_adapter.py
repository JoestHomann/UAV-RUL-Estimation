"""Build the copied inputs and compact manifest for the tabular data adapter.

Step 1's generated experiment specification is the sole configuration input.
This builder uses the Phase 1 paths recorded there, refreshes seven local data
copies with preserved filesystem metadata, writes a compact manifest, and then
runs the shared copy-integrity checker.

The source Phase 1 files are always read-only. Only generated files below this
step's artifact directory are created or replaced.
"""

from __future__ import annotations

import argparse
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
MANIFEST_FILENAME = "tabular_dataset_manifest.json"
REPORT_FILENAME = "copy_verification.json"

# Direct execution places this step directory on Python's import path. The
# explicit guard also supports importing this builder from another Phase 2 tool.
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from verify_copied_files import (  # noqa: E402
    CopyVerificationError,
    verify_copied_files,
)


class AdapterBuildError(ValueError):
    """Represent a readable specification, path, or copy failure."""


# Each logical name maps to the Phase 1 artifact key stored in the experiment
# specification. Destination names remain unchanged so provenance is obvious.
COPY_PLAN: dict[str, tuple[str, str]] = {
    "training": ("training_features", "training_features.csv.gz"),
    "development": (
        "development_features",
        "development_validation_features.csv.gz",
    ),
    "locked": ("locked_features", "locked_validation_features.csv.gz"),
    "test": ("test_features", "test_features.csv.gz"),
    "feature_catalog": ("feature_catalog", "feature_catalog.csv"),
    "outer_folds": ("outer_folds", "outer_folds.csv"),
    "inner_folds": ("inner_folds", "inner_folds.csv"),
}


# Metadata columns remain separate from model features in every returned
# TabularDataset. Target and weight columns receive their own fields as well.
DATASET_INTERFACES: dict[str, dict[str, Any]] = {
    "training": {
        "role": "model_fitting_prefixes",
        "metadata_columns": [
            "sample_id",
            "uav_id",
            "prefix_number",
            "cutoff",
        ],
        "target_column": "RUL",
        "sample_weight_column": "sample_weight",
    },
    "development": {
        "role": "inner_selection_scenarios",
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
    "locked": {
        "role": "locked_outer_evaluation_scenarios",
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
    "test": {
        "role": "final_unlabelled_test_endpoints",
        "metadata_columns": ["sample_id", "uav_id", "cutoff"],
        "target_column": None,
        "sample_weight_column": None,
    },
}


def _load_experiment_specification(path: Path) -> dict[str, Any]:
    """Read Step 1's resolved JSON and require the sections Step 2 consumes."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterBuildError(
            "Cannot read the Step 1 experiment specification. Run the Step 1 "
            f"builder first. Details: {error}"
        ) from error

    try:
        settings = payload["settings"]
        phase_1 = settings["phase_1"]
        artifacts = phase_1["artifacts"]
        feature_sets = phase_1["expected_feature_sets"]
        settings_version = settings["settings_version"]
    except (KeyError, TypeError) as error:
        raise AdapterBuildError(
            f"Experiment specification is missing required field {error}"
        ) from error

    if not isinstance(artifacts, dict) or not isinstance(feature_sets, dict):
        raise AdapterBuildError(
            "Experiment specification has invalid Phase 1 artifact metadata"
        )
    if not isinstance(settings_version, int):
        raise AdapterBuildError("Experiment settings version must be an integer")
    return payload


def _repository_relative(path: Path) -> str:
    """Return a portable repository-relative path."""

    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise AdapterBuildError(f"Path is outside the repository: {path}") from error


def _resolve_source(relative_path: str) -> Path:
    """Resolve a declared source and prevent absolute or escaping paths."""

    supplied = Path(relative_path)
    if supplied.is_absolute():
        raise AdapterBuildError(f"Phase 1 source must be relative: {relative_path}")

    source = (REPOSITORY_ROOT / supplied).resolve()
    try:
        source.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise AdapterBuildError(
            f"Phase 1 source escapes the repository: {relative_path}"
        ) from error
    if not source.is_file():
        raise AdapterBuildError(f"Required Phase 1 source does not exist: {source}")
    return source


def _artifact_specification(
    specification: dict[str, Any],
    artifact_key: str,
) -> dict[str, Any]:
    """Return one Phase 1 artifact entry with a readable missing-key error."""

    try:
        artifact = specification["settings"]["phase_1"]["artifacts"][artifact_key]
    except (KeyError, TypeError) as error:
        raise AdapterBuildError(
            f"Experiment specification does not define artifact {artifact_key!r}"
        ) from error
    if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
        raise AdapterBuildError(
            f"Experiment artifact {artifact_key!r} does not define a valid path"
        )
    return artifact


def _copy_inputs(
    specification: dict[str, Any],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Refresh all seven local copies and return their manifest entries.

    shutil.copy2 preserves source modification times and other common metadata.
    Existing generated copies are replaced deliberately, while Phase 1 sources
    are never opened for writing.
    """

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_entries: dict[str, dict[str, Any]] = {}

    for logical_name, (artifact_key, destination_name) in COPY_PLAN.items():
        artifact = _artifact_specification(specification, artifact_key)
        source_label = artifact["path"]
        source_path = _resolve_source(source_label)
        copied_path = data_dir / destination_name

        try:
            shutil.copy2(source_path, copied_path)
        except OSError as error:
            raise AdapterBuildError(
                f"Cannot copy {source_path} to {copied_path}: {error}"
            ) from error

        entry: dict[str, Any] = {
            "source_artifact_key": artifact_key,
            "source_path": source_label,
            "copied_path": f"data/{destination_name}",
            "expected_rows": artifact.get("rows"),
        }

        # Feature datasets need a declared interface so the runtime adapter can
        # separate model inputs from identifiers, targets, and sample weights.
        if logical_name in DATASET_INTERFACES:
            entry["category"] = "feature_dataset"
            entry.update(DATASET_INTERFACES[logical_name])
        elif logical_name == "feature_catalog":
            entry.update(
                {
                    "category": "feature_catalog",
                    "feature_name_column": "feature_name",
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
        file_entries[logical_name] = entry

    return file_entries


def _build_manifest(
    specification: dict[str, Any],
    specification_path: Path,
    files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Create the compact machine-readable interface for the copied data."""

    settings = specification["settings"]
    feature_sets = settings["phase_1"]["expected_feature_sets"]
    files["feature_catalog"]["membership_columns"] = list(feature_sets)

    return {
        "adapter_version": 1,
        "settings_version": settings["settings_version"],
        "experiment_specification": _repository_relative(specification_path),
        "feature_sets": {
            name: {"feature_count": count}
            for name, count in feature_sets.items()
        },
        "files": files,
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Write generated metadata with stable ordering and formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_tabular_data_adapter(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Build all Step 2 artifacts and return manifest and report paths."""

    specification = _load_experiment_specification(specification_path)
    copied_files = _copy_inputs(specification, output_dir)
    manifest = _build_manifest(
        specification,
        specification_path,
        copied_files,
    )

    manifest_path = output_dir / MANIFEST_FILENAME
    report_path = output_dir / REPORT_FILENAME
    _write_json(manifest, manifest_path)

    # Reuse the independently runnable checker. There is only one definition of
    # copy integrity, so automatic and manual verification cannot drift apart.
    verify_copied_files(manifest_path, report_path)
    return manifest_path, report_path


def main() -> None:
    """Build Step 2 from the command line."""

    # Only artifact locations are configurable. Dataset choices and interfaces
    # come from the resolved experiment specification and this adapter design.
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
        help="Directory for copied data, manifest, and verification report.",
    )
    args = parser.parse_args()

    try:
        manifest_path, report_path = build_tabular_data_adapter(
            args.specification,
            args.output_dir,
        )
    except (AdapterBuildError, CopyVerificationError) as error:
        print(f"Tabular data adapter build failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("Tabular data adapter built successfully")
    print(f"Copied files: {len(COPY_PLAN)}")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Verification report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
