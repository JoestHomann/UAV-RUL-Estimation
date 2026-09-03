"""Validate Phase 3 settings and the referenced completed Phase 2 run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import tomllib
from typing import Any, Literal

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
    phase_2_run_root,
    read_json,
    repository_relative,
    configured_repository_path,
)


DEFAULT_SETTINGS_PATH = STEP_DIR / "phase_3_settings.toml"
MAX_SEED = 2**32 - 1
PROMOTED_ENSEMBLE_FAMILY = "calibrated_tree_blend"
PROMOTED_STACK_FAMILY = "heterogeneous_oof_stack"
RESIDUAL_ENSEMBLE_FAMILY = "residual_corrected_tree_ensemble"


class SettingsError(Phase3Error):
    """Explain invalid TOML or an unusable Phase 2 reference."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FinalSearchSettings(StrictModel):
    candidate_budget: int = Field(gt=0)
    search_seed: int = Field(ge=0, le=MAX_SEED)
    model_seed: int = Field(ge=0, le=MAX_SEED)


class PredictionCalibrationSettings(StrictModel):
    """Override post-model calibration for the new Phase 3 run."""

    calibration: Literal["none", "conditional_quantile"]
    safety_offset: float = Field(default=0.0, ge=0.0)
    non_overprediction_coverage: float = Field(gt=0.0, lt=1.0)
    calibration_prediction_bin_edges: list[float] = Field(min_length=3)
    calibration_minimum_bin_rows: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_conditional_policy(self) -> "PredictionCalibrationSettings":
        if self.calibration == "conditional_quantile":
            if self.safety_offset != 0.0:
                raise ValueError("conditional calibration requires safety_offset = 0")
            if self.non_overprediction_coverage < 0.5:
                raise ValueError("conditional calibration coverage must be at least 0.5")
        edges = self.calibration_prediction_bin_edges
        if any(upper <= lower for lower, upper in zip(edges[:-1], edges[1:], strict=True)):
            raise ValueError("calibration prediction bin edges must be strictly increasing")
        return self


class SubmissionPolicySettings(PredictionCalibrationSettings):
    """Name one frozen post-model policy produced from the same base model."""

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")


class ResidualCorrectedEnsembleSettings(StrictModel):
    """Freeze the component choices used to deploy the PE_11 winner."""

    experiment_definition: str = Field(min_length=1)
    extra_trees_configuration_index: int = Field(ge=0, le=4)
    xgboost_configuration_index: int = Field(ge=0, le=4)
    internal_folds: int = Field(ge=2, le=10)


class Phase3Settings(StrictModel):
    settings_version: int = Field(gt=0)
    run_number: int = Field(gt=0)
    phase_2_run_number: int = Field(gt=0)
    selected_model_family: str = Field(min_length=1)
    phase_2_run_root: str | None = None
    phase_2_specification: str | None = None
    phase_2_model_registry: str | None = None
    tabular_manifest: str | None = None
    sequence_manifest: str | None = None
    trajectory_manifest: str | None = None
    promotion_contract: str | None = None
    promotion_manifest: str | None = None
    final_search: FinalSearchSettings
    prediction_calibration: PredictionCalibrationSettings | None = None
    submission_policies: list[SubmissionPolicySettings] | None = None
    canonical_submission_policy: str | None = None
    residual_corrected_ensemble: ResidualCorrectedEnsembleSettings | None = None

    @model_validator(mode="after")
    def family_name_is_normalized(self) -> "Phase3Settings":
        if self.selected_model_family != self.selected_model_family.strip().lower():
            raise ValueError(
                "selected_model_family must use the normalized registry name"
            )
        if self.selected_model_family in {
            PROMOTED_ENSEMBLE_FAMILY,
            PROMOTED_STACK_FAMILY,
            RESIDUAL_ENSEMBLE_FAMILY,
        }:
            if self.promotion_contract is None or self.promotion_manifest is None:
                raise ValueError(
                    f"{self.selected_model_family} requires promotion_contract "
                    "and promotion_manifest"
                )
        if (
            self.selected_model_family == RESIDUAL_ENSEMBLE_FAMILY
            and self.residual_corrected_ensemble is None
        ):
            raise ValueError(
                "residual_corrected_tree_ensemble requires "
                "residual_corrected_ensemble settings"
            )
        names = [policy.name for policy in (self.submission_policies or [])]
        if len(names) != len(set(names)):
            raise ValueError("submission policy names must be unique")
        if names:
            if self.canonical_submission_policy not in names:
                raise ValueError(
                    "canonical_submission_policy must name one submission policy"
                )
        elif self.canonical_submission_policy is not None:
            raise ValueError(
                "canonical_submission_policy requires submission_policies"
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
    manifest_paths: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "passed",
            "settings_version": self.settings_version,
            "run_number": self.run_number,
            "selected_family": self.selected_family,
            "representation": self.representation,
            "manifests": {
                name: repository_relative(path)
                for name, path in self.manifest_paths.items()
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
        if path.suffix.lower() == ".json":
            payload = read_json(path, "Phase 3 settings")
        else:
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

    settings_payload = settings.model_dump(mode="json")
    phase_2_root = configured_repository_path(
        settings_payload,
        "phase_2_run_root",
        phase_2_run_root(settings.phase_2_run_number),
    )
    if settings.selected_model_family == PROMOTED_ENSEMBLE_FAMILY:
        return _verify_promoted_ensemble_reference(settings, phase_2_root)
    if settings.selected_model_family == PROMOTED_STACK_FAMILY:
        return _verify_promoted_stack_reference(settings, phase_2_root)
    if settings.selected_model_family == RESIDUAL_ENSEMBLE_FAMILY:
        return _verify_residual_ensemble_reference(settings, phase_2_root)

    manifest_files = phase_2_manifest_paths(
        settings.phase_2_run_number,
        run_root=phase_2_root,
    )
    manifests = {
        name: read_json(path, f"Phase 2 {name} manifest")
        for name, path in manifest_files.items()
    }
    phase_2_settings_version = _manifest_settings_version(
        manifests,
        settings.phase_2_run_number,
    )

    specification_path = configured_repository_path(
        settings_payload,
        "phase_2_specification",
        PHASE_2_SPECIFICATION_PATH,
    )
    specification = read_json(
        specification_path,
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

    registry_path = configured_repository_path(
        settings_payload,
        "phase_2_model_registry",
        PHASE_2_MODEL_REGISTRY_PATH,
    )
    registry = read_json(registry_path, "Phase 2 model registry")
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
        manifest_paths=manifest_files,
    )


def _verify_promoted_ensemble_reference(
    settings: Phase3Settings,
    phase_2_root: Path,
) -> Phase2Verification:
    """Validate the frozen PE_3 run_1 policy without requiring a Step 7 ranking."""

    payload = settings.model_dump(mode="json")
    selection_path = (
        phase_2_root / "5_inner_model_selection" / "selection_manifest.json"
    )


    promotion_contract_path = configured_repository_path(
        payload,
        "promotion_contract",
        Path("promotion_contract.json"),
    )
    promotion_manifest_path = configured_repository_path(
        payload,
        "promotion_manifest",
        Path("locked_confirmation_manifest.json"),
    )
    selection = read_json(selection_path, "Phase 2 selection manifest")
    contract = read_json(promotion_contract_path, "promoted ensemble contract")
    promotion = read_json(promotion_manifest_path, "promoted ensemble manifest")
    if selection.get("status") != "complete":
        raise SettingsError("Promoted ensemble source selection is incomplete")
    if promotion.get("status") != "complete" or promotion.get("phase_3_created") is not False:
        raise SettingsError("Promoted ensemble locked confirmation is not complete")
    if promotion.get("policy") != "ensemble:blend_xgb_0.50__calibrated":
        raise SettingsError("Promotion manifest identifies another policy")
    if contract.get("selected_candidate") != promotion.get("policy"):
        raise SettingsError("Promotion contract and confirmation policy disagree")
    if contract.get("component_families") != ["extra_trees", "xgboost"]:
        raise SettingsError("Promoted ensemble components are not frozen as expected")
    if contract.get("locked_results_used_for_selection") is not False:
        raise SettingsError("Promoted ensemble contract used locked results for selection")

    component_manifest_value = promotion.get("component_locked_manifest")
    if not isinstance(component_manifest_value, str) or not component_manifest_value:
        raise SettingsError("Promotion manifest has no component locked manifest")
    component_manifest_path = configured_repository_path(
        {"component_manifest": component_manifest_value},
        "component_manifest",
        Path("locked_evaluation_manifest.json"),
    )
    component_manifest = read_json(
        component_manifest_path,
        "promoted component locked manifest",
    )
    if component_manifest.get("status") != "complete":
        raise SettingsError("Promoted component locked evaluation is incomplete")

    specification_path = configured_repository_path(
        payload,
        "phase_2_specification",
        PHASE_2_SPECIFICATION_PATH,
    )
    specification = read_json(specification_path, "Phase 2 experiment specification")
    phase_2_settings = specification.get("settings")
    if not isinstance(phase_2_settings, dict):
        raise SettingsError("Phase 2 experiment specification has no settings object")
    if phase_2_settings.get("run_number") != settings.phase_2_run_number:
        raise SettingsError("Promoted source specification identifies another run")
    settings_version = int(phase_2_settings["settings_version"])
    if selection.get("settings_version") != settings_version:
        raise SettingsError("Promoted source selection uses another settings version")
    if component_manifest.get("settings_version") != settings_version:
        raise SettingsError("Promoted component evaluation uses another settings version")

    registry_path = configured_repository_path(
        payload,
        "phase_2_model_registry",
        PHASE_2_MODEL_REGISTRY_PATH,
    )
    registry = read_json(registry_path, "Phase 2 model registry")
    families = registry.get("families")
    if registry.get("settings_version") != settings_version or not isinstance(families, dict):
        raise SettingsError("Promoted source model registry is incompatible")
    for family in ("extra_trees", "xgboost"):
        details = families.get(family)
        if not isinstance(details, dict) or details.get("enabled") is not True:
            raise SettingsError(f"Promoted component {family!r} is unavailable")

    return Phase2Verification(
        settings_version=settings_version,
        run_number=settings.phase_2_run_number,
        selected_family=PROMOTED_ENSEMBLE_FAMILY,
        representation="tabular",
        phase_2_specification=specification,
        model_registry=registry,
        manifests={
            "selection": selection,
            "promotion_contract": contract,
            "promotion": promotion,
            "component_locked_evaluation": component_manifest,
        },
        manifest_paths={
            "selection": selection_path,
            "promotion_contract": promotion_contract_path,
            "promotion": promotion_manifest_path,
            "component_locked_evaluation": component_manifest_path,
        },
    )


def _verify_promoted_stack_reference(
    settings: Phase3Settings,
    phase_2_root: Path,
) -> Phase2Verification:
    """Require a locked-confirmed PE_7 stack without reopening its search."""

    del phase_2_root
    payload = settings.model_dump(mode="json")
    contract_path = configured_repository_path(
        payload, "promotion_contract", Path("promotion_contract.json")
    )
    confirmation_path = configured_repository_path(
        payload, "promotion_manifest", Path("locked_confirmation_manifest.json")
    )
    contract = read_json(contract_path, "promoted stack contract")
    confirmation = read_json(confirmation_path, "promoted stack confirmation")
    if contract.get("family") != PROMOTED_STACK_FAMILY:
        raise SettingsError("Promotion contract identifies another stack family")
    if contract.get("meta_model_fitted_from_oof_only") is not True:
        raise SettingsError("Promoted stack meta-model is not proven OOF-only")
    if not isinstance(contract.get("adapter_hyperparameters"), dict):
        raise SettingsError("Promoted stack has no adapter hyperparameters")
    if (
        confirmation.get("status") != "complete"
        or confirmation.get("uses_locked_evaluation") is not True
        or confirmation.get("locked_results_used_for_tuning") is not False
        or confirmation.get("development_gate_passed") is not True
    ):
        raise SettingsError("Promoted stack has no valid locked confirmation")

    specification_path = configured_repository_path(
        payload, "phase_2_specification", PHASE_2_SPECIFICATION_PATH
    )
    specification = read_json(specification_path, "stack source specification")
    phase_2_settings = specification.get("settings")
    if not isinstance(phase_2_settings, dict):
        raise SettingsError("Stack source specification has no settings")
    settings_version = int(phase_2_settings["settings_version"])
    registry_path = configured_repository_path(
        payload, "phase_2_model_registry", PHASE_2_MODEL_REGISTRY_PATH
    )
    registry = read_json(registry_path, "stack source registry")
    if registry.get("settings_version") != settings_version:
        raise SettingsError("Stack source registry uses another settings version")
    return Phase2Verification(
        settings_version=settings_version,
        run_number=settings.phase_2_run_number,
        selected_family=PROMOTED_STACK_FAMILY,
        representation="heterogeneous",
        phase_2_specification=specification,
        model_registry=registry,
        manifests={
            "promotion_contract": contract,
            "promotion": confirmation,
        },
        manifest_paths={
            "promotion_contract": contract_path,
            "promotion": confirmation_path,
        },
    )


def _verify_residual_ensemble_reference(
    settings: Phase3Settings,
    phase_2_root: Path,
) -> Phase2Verification:
    """Require PE_11 promotion and PE_12's independent test-like ranking."""

    del phase_2_root
    payload = settings.model_dump(mode="json")
    pe11_path = configured_repository_path(
        payload,
        "promotion_contract",
        Path("winner_manifest.json"),
    )
    pe12_path = configured_repository_path(
        payload,
        "promotion_manifest",
        Path("winner_manifest.json"),
    )
    pe11 = read_json(pe11_path, "PE_11 winner manifest")
    pe12 = read_json(pe12_path, "PE_12 winner manifest")
    if (
        pe11.get("status") != "promoted"
        or pe11.get("promoted") is not True
        or pe11.get("winner") != "residual_corrected"
        or pe11.get("uses_locked_evaluation") is not False
        or pe11.get("member_count") != 6
    ):
        raise SettingsError("PE_11 does not contain the promoted six-member residual policy")
    if (
        pe12.get("status") != "complete"
        or pe12.get("winner") != "PE11::residual_corrected"
        or pe12.get("uses_locked_evaluation") is not False
        or pe12.get("uses_test_labels") is not False
    ):
        raise SettingsError("PE_12 did not confirm PE_11 without test labels")

    specification_path = configured_repository_path(
        payload,
        "phase_2_specification",
        PHASE_2_SPECIFICATION_PATH,
    )
    specification = read_json(specification_path, "PE_11 source specification")
    phase_2_settings = specification.get("settings")
    if not isinstance(phase_2_settings, dict):
        raise SettingsError("PE_11 source specification has no settings object")
    if phase_2_settings.get("run_number") != settings.phase_2_run_number:
        raise SettingsError("PE_11 source specification identifies another run")
    settings_version = int(phase_2_settings["settings_version"])

    registry_path = configured_repository_path(
        payload,
        "phase_2_model_registry",
        PHASE_2_MODEL_REGISTRY_PATH,
    )
    registry = read_json(registry_path, "PE_11 source model registry")
    families = registry.get("families")
    if registry.get("settings_version") != settings_version or not isinstance(
        families, dict
    ):
        raise SettingsError("PE_11 source model registry is incompatible")
    for family in ("extra_trees", "xgboost"):
        details = families.get(family)
        if not isinstance(details, dict) or details.get("enabled") is not True:
            raise SettingsError(f"PE_11 component {family!r} is unavailable")

    ensemble_settings = settings.residual_corrected_ensemble
    assert ensemble_settings is not None
    definition_path = configured_repository_path(
        {"definition": ensemble_settings.experiment_definition},
        "definition",
        Path("settings.toml"),
    )
    if not definition_path.is_file():
        raise SettingsError("PE_11 experiment definition is unavailable")

    return Phase2Verification(
        settings_version=settings_version,
        run_number=settings.phase_2_run_number,
        selected_family=RESIDUAL_ENSEMBLE_FAMILY,
        representation="tabular",
        phase_2_specification=specification,
        model_registry=registry,
        manifests={"pe11_promotion": pe11, "pe12_confirmation": pe12},
        manifest_paths={
            "pe11_promotion": pe11_path,
            "pe12_confirmation": pe12_path,
            "experiment_definition": definition_path,
        },
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
