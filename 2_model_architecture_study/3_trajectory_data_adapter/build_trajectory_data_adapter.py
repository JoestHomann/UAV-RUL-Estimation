"""Build the reusable trajectory interface over verified Step 3 inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from verify_trajectory_data_adapter import (
    TrajectoryVerificationError,
    verify_trajectory_data_adapter,
)


STEP_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STEP_DIR.parent.parent
DEFAULT_SPECIFICATION_PATH = (
    STEP_DIR.parent
    / "1_architecture_study_settings"
    / "artifacts"
    / "experiment_specification.json"
)
DEFAULT_SEQUENCE_MANIFEST_PATH = (
    STEP_DIR.parent
    / "3_sequence_data_adapter"
    / "artifacts"
    / "sequence_dataset_manifest.json"
)
DEFAULT_SEQUENCE_REPORT_PATH = (
    STEP_DIR.parent
    / "3_sequence_data_adapter"
    / "artifacts"
    / "copy_verification.json"
)
DEFAULT_OUTPUT_DIR = STEP_DIR / "artifacts"
MANIFEST_FILENAME = "trajectory_dataset_manifest.json"
REPORT_FILENAME = "trajectory_verification.json"


class TrajectoryBuildError(ValueError):
    """Represent a readable trajectory manifest or prerequisite failure."""


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrajectoryBuildError(f"Cannot read {description} at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TrajectoryBuildError(f"{description} must be a JSON object")
    return payload


def _repository_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise TrajectoryBuildError(f"Path is outside the repository: {path}") from error


def build_trajectory_data_adapter(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    sequence_manifest_path: Path = DEFAULT_SEQUENCE_MANIFEST_PATH,
    sequence_report_path: Path = DEFAULT_SEQUENCE_REPORT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Write the trajectory manifest and exercise its fold-safe runtime API."""

    specification = _read_json(specification_path, "experiment specification")
    sequence_manifest = _read_json(sequence_manifest_path, "sequence manifest")
    sequence_report = _read_json(sequence_report_path, "sequence copy report")
    try:
        settings = specification["settings"]
        settings_version = int(settings["settings_version"])
        representations = settings["representations"]
    except (KeyError, TypeError, ValueError) as error:
        raise TrajectoryBuildError(
            f"Experiment specification is missing trajectory inputs: {error}"
        ) from error
    if sequence_manifest.get("settings_version") != settings_version:
        raise TrajectoryBuildError("Sequence manifest uses another settings version")
    if sequence_report.get("status") != "passed":
        raise TrajectoryBuildError("Sequence copied inputs have not passed verification")

    manifest = {
        "adapter_version": 1,
        "settings_version": settings_version,
        "experiment_specification": _repository_relative(specification_path),
        "source_sequence_manifest": _repository_relative(sequence_manifest_path),
        "channels": representations["sequence_channels"],
        "channel_count": representations["sequence_channel_count"],
        "side_features": representations["sequence_side_features"],
        "query_interface": {
            "history": "all observed cycles through endpoint cutoff",
            "variable_length": True,
            "post_cutoff_rows_allowed": False,
        },
        "reference_interface": {
            "history": "complete run-to-failure training-UAV trajectories",
            "remaining_life": "terminal cycle minus current cycle",
            "fit_scope": "active_training_uavs_only",
        },
        "scaling": sequence_manifest.get("scaling"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    report_path = output_dir / REPORT_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        report = verify_trajectory_data_adapter(manifest_path)
    except TrajectoryVerificationError as error:
        raise TrajectoryBuildError(str(error)) from error
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH)
    parser.add_argument(
        "--sequence-manifest",
        type=Path,
        default=DEFAULT_SEQUENCE_MANIFEST_PATH,
    )
    parser.add_argument(
        "--sequence-report",
        type=Path,
        default=DEFAULT_SEQUENCE_REPORT_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest_path, report_path = build_trajectory_data_adapter(
            args.specification,
            args.sequence_manifest,
            args.sequence_report,
            args.output_dir,
        )
    except TrajectoryBuildError as error:
        print(f"Trajectory data adapter build failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("Trajectory data adapter built successfully")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Verification report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
