"""Build Phase 3's resolved settings and manual architecture selection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


STEP_DIR = Path(__file__).resolve().parent
PHASE_DIR = STEP_DIR.parent
for dependency_dir in (PHASE_DIR, STEP_DIR):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from phase_3_common import (  # noqa: E402
    Phase3Error,
    artifacts_directory,
    manifest_path,
    read_optional_json,
    repository_relative,
    selected_architecture_path,
    step_directory,
    write_json,
)
from verify_phase_3_settings import (  # noqa: E402
    DEFAULT_SETTINGS_PATH,
    SettingsError,
    load_and_verify_settings,
)


def build_selection(settings_path: Path = DEFAULT_SETTINGS_PATH) -> dict[str, object]:
    settings, verification = load_and_verify_settings(settings_path)
    run_number = settings.run_number
    selection_path = selected_architecture_path(run_number)

    phase_2_settings = verification.phase_2_specification["settings"]
    architecture = phase_2_settings["architectures"][settings.selected_model_family]
    selection = {
        "selection_version": 1,
        "settings_version": settings.settings_version,
        "phase_2_run_number": settings.phase_2_run_number,
        "phase_2_settings_version": verification.settings_version,
        "phase_3_run_number": run_number,
        "selected_model_family": settings.selected_model_family,
        "representation": architecture["representation"],
        "feature_sets": architecture["feature_sets"],
        "lookbacks": architecture["lookbacks"],
        "primary_metric": "mean_fold_rmse",
        "prediction_minimum": phase_2_settings["evaluation"]["prediction_minimum"],
        "locked_results_used_for_architecture_selection": True,
        "locked_results_used_for_configuration_tuning": False,
        "test_data_loaded": False,
        "selection_status": "approved",
    }
    resolved = {
        "settings_source": repository_relative(settings_path),
        "settings": settings.model_dump(mode="json"),
        "phase_2_verification": verification.to_dict(),
        "phase_2_settings": phase_2_settings,
    }
    artifact_dir = artifacts_directory(1, run_number)
    existing_selection = read_optional_json(selection_path, "selected architecture")
    existing_resolved = read_optional_json(
        artifact_dir / "resolved_phase_3_settings.json",
        "resolved Phase 3 settings",
    )
    step_2_dir = step_directory(2, run_number=run_number)
    step_2_started = step_2_dir.is_dir() and any(
        path.is_file() for path in step_2_dir.rglob("*")
    )
    if step_2_started and (
        existing_selection != selection or existing_resolved != resolved
    ):
        raise Phase3Error(
            "Phase 3 settings cannot change after Step 2 has written output; "
            "start a new run"
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(resolved, artifact_dir / "resolved_phase_3_settings.json")
    write_json(selection, selection_path)
    manifest = {
        "manifest_version": 1,
        "settings_version": settings.settings_version,
        "phase_2_run_number": settings.phase_2_run_number,
        "phase_3_run_number": run_number,
        "status": "complete",
        "locked_results_used_for_architecture_selection": True,
        "locked_results_used_for_configuration_tuning": False,
        "test_data_loaded": False,
        "artifacts": {
            "resolved_settings": "artifacts/resolved_phase_3_settings.json",
            "selected_architecture": "artifacts/selected_architecture.json",
        },
    }
    write_json(manifest, manifest_path(1, run_number))
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH)
    args = parser.parse_args()
    try:
        selection = build_selection(args.settings)
    except (SettingsError, Phase3Error) as error:
        print(f"Architecture selection build failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("Phase 3 architecture selection built")
    print(f"Selected family: {selection['selected_model_family']}")
    print(f"Phase 3 run: {selection['phase_3_run_number']}")


if __name__ == "__main__":
    main()
