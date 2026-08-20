"""Validate Phase 3 settings and the referenced completed Phase 2 run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import tomllib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


STEP_DIR = Path(__file__).resolve().parent
PHASE_DIR = STEP_DIR.parent
for dependency_dir in (PHASE_DIR,):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from phase_3_common import (  # noqa: E402
    PHASE_2_MODEL_REGISTRY_PATH,
    PHASE_2_SPECIFICATION_PATH,
    Phase3Error,
    phase_2_manifest_paths,
    read_json,
    repository_relative,
)


DEFAULT_SETTINGS_PATH = STEP_DIR / "phase_3_settings.toml"
MAX_SEED = 2**32 - 1


class SettingsError(Phase3Error):
    """Explain invalid TOML or an unusable Phase 2 reference."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FinalSearchSettings(StrictModel):
    candidate_budget: int = Field(gt=0)
    search_seed: int = Field(ge=0, le=MAX_SEED)
    model_seed: int = Field(ge=0, le=MAX_SEED)


class Phase3Settings(StrictModel):
    settings_version: int = Field(gt=0)
    run_number: int = Field(gt=0)
    phase_2_run_number: int = Field(gt=0)
    selected_model_family: str = Field(min_length=1)
    final_search: FinalSearchSettings

    @model_validator(mode="after")
    def family_name_is_normalized(self) -> "Phase3Settings":
        if self.selected_model_family != self.selected_model_family.strip().lower():
            raise ValueError(
                "selected_model_family must use the normalized registry name"
            )
        return self


@dataclass(frozen=True)
class Phase2Verification:
    settings_version: int
    run_number: int
    selected_family: str
    representation: str
    phase_2_specification: dict[str, Any]
    model_registry: dict[str, Any]
    manifests: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "passed",
            "settings_version": self.settings_version,
            "run_number": self.run_number,
            "selected_family": self.selected_family,
            "representation": self.representation,
            "manifests": {
                name: repository_relative(path)
                for name, path in phase_2_manifest_paths(self.run_number).items()
            },
        }


def _format_validation_error(error: ValidationError) -> str:
    messages = []
    for detail in error.errors(include_url=False):
        location = ".".join(str(part) for part in detail["loc"])
        messages.append(f"{location}: {detail['msg']}")
    return "\n".join(messages)


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> Phase3Settings:
    """Read and strictly validate the human-authored Phase 3 TOML."""

    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SettingsError(f"Cannot read Phase 3 settings {path}: {error}") from error
    try:
        return Phase3Settings.model_validate(payload)
    except ValidationError as error:
        raise SettingsError(
            "Phase 3 settings schema validation failed:\n"
            f"{_format_validation_error(error)}"
        ) from error


def _manifest_settings_version(
    manifests: dict[str, dict[str, Any]],
    run_number: int,
) -> int:
    versions: set[int] = set()
    for name, manifest in manifests.items():
        if manifest.get("status") != "complete":
            raise SettingsError(f"Referenced Phase 2 {name} manifest is not complete")
        if manifest.get("run_number") != run_number:
            raise SettingsError(
                f"Referenced Phase 2 {name} manifest identifies another run"
            )
        value = manifest.get("settings_version")
        if isinstance(value, bool) or not isinstance(value, int):
            raise SettingsError(
                f"Referenced Phase 2 {name} manifest has no settings version"
            )
        versions.add(value)
    if len(versions) != 1:
        raise SettingsError("Referenced Phase 2 manifests disagree on settings version")
    return versions.pop()


def verify_phase_2_reference(settings: Phase3Settings) -> Phase2Verification:
    """Require a complete selected family without opening locked predictions."""

    manifest_files = phase_2_manifest_paths(settings.phase_2_run_number)
    manifests = {
        name: read_json(path, f"Phase 2 {name} manifest")
        for name, path in manifest_files.items()
    }
    phase_2_settings_version = _manifest_settings_version(
        manifests,
        settings.phase_2_run_number,
    )

    specification = read_json(
        PHASE_2_SPECIFICATION_PATH,
        "Phase 2 experiment specification",
    )
    phase_2_settings = specification.get("settings")
    if not isinstance(phase_2_settings, dict):
        raise SettingsError("Phase 2 experiment specification has no settings object")
    if phase_2_settings.get("run_number") != settings.phase_2_run_number:
        raise SettingsError(
            "Current Phase 2 experiment specification does not identify the "
            "referenced run"
        )
    if phase_2_settings.get("settings_version") != phase_2_settings_version:
        raise SettingsError(
            "Current Phase 2 experiment specification does not match the "
            "referenced run's settings version"
        )

    family = settings.selected_model_family
    enabled = manifests["comparison"].get("enabled_families", [])
    if family not in enabled:
        raise SettingsError(
            f"Selected family {family!r} is absent from the complete comparison"
        )
    outer_folds = manifests["locked_evaluation"].get("outer_fold_labels", [])
    required_studies = {
        f"{family}__outer_{int(fold):02d}" for fold in outer_folds
    }
    complete_studies = set(manifests["selection"].get("completed_studies", []))
    missing_studies = sorted(required_studies - complete_studies)
    if missing_studies:
        raise SettingsError(
            f"Selected family has incomplete Phase 2 tuning: {missing_studies}"
        )

    registry = read_json(PHASE_2_MODEL_REGISTRY_PATH, "Phase 2 model registry")
    if registry.get("settings_version") != phase_2_settings_version:
        raise SettingsError("Phase 2 model registry uses another settings version")
    families = registry.get("families")
    family_registry = families.get(family) if isinstance(families, dict) else None
    if not isinstance(family_registry, dict):
        raise SettingsError(f"Model registry has no family {family!r}")
    if family_registry.get("implemented") is not True:
        raise SettingsError(f"Selected family {family!r} is not implemented")
    if family_registry.get("enabled") is not True:
        raise SettingsError(f"Selected family {family!r} is disabled")

    retraining_seeds = phase_2_settings["tuning"]["retraining_seeds"]
    required_seeds = (
        retraining_seeds
        if family_registry.get("stochastic") is True
        else retraining_seeds[:1]
    )
    required_runs = {
        f"{family}__outer_{int(fold):02d}__seed_{int(seed):03d}"
        for fold in outer_folds
        for seed in required_seeds
    }
    complete_runs = set(manifests["locked_evaluation"].get("completed_runs", []))
    missing_runs = sorted(required_runs - complete_runs)
    if missing_runs:
        raise SettingsError(
            f"Selected family has incomplete Phase 2 evaluation: {missing_runs}"
        )

    architecture = phase_2_settings.get("architectures", {}).get(family)
    if not isinstance(architecture, dict):
        raise SettingsError(f"Phase 2 settings have no architecture {family!r}")
    return Phase2Verification(
        settings_version=phase_2_settings_version,
        run_number=settings.phase_2_run_number,
        selected_family=family,
        representation=str(architecture["representation"]),
        phase_2_specification=specification,
        model_registry=registry,
        manifests=manifests,
    )


def load_and_verify_settings(
    path: Path = DEFAULT_SETTINGS_PATH,
) -> tuple[Phase3Settings, Phase2Verification]:
    settings = load_settings(path)
    return settings, verify_phase_2_reference(settings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH)
    args = parser.parse_args()
    try:
        settings, verification = load_and_verify_settings(args.settings)
    except (SettingsError, Phase3Error) as error:
        print(f"Phase 3 settings verification failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("Phase 3 settings verification passed")
    print(f"Phase 3 run: {settings.run_number}")
    print(f"Phase 2 source run: {verification.run_number}")
    print(f"Selected family: {settings.selected_model_family}")


if __name__ == "__main__":
    main()
