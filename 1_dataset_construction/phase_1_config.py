"""Load and validate the versioned Phase 1 experiment settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS_PATH = SCRIPT_DIR / "phase_1_settings.toml"
SUPPORTED_FEATURE_PROFILES = {"legacy", "extended"}
SUPPORTED_PREFIX_STRATEGIES = {
    "empirical",
    "stratified_empirical",
    "dense_all",
    "dense_stride",
}
SUPPORTED_SCENARIO_ASSIGNMENTS = {"eligible_random", "bipartite"}


@dataclass(frozen=True)
class PrefixVariant:
    name: str
    strategy: str
    cutoffs_per_uav: int | None
    seed: int
    stride: int = 1
    minimum_cutoff: int = 1


@dataclass(frozen=True)
class ScenarioProfile:
    name: str
    assignment: str
    development_scenarios: int
    locked_scenarios: int
    seed: int
    minimum_rul: int | None = None
    maximum_rul: int | None = None


@dataclass(frozen=True)
class PhaseOneProfile:
    name: str
    feature_profile: str
    feature_sets: tuple[str, ...]
    prefix_variants: tuple[PrefixVariant, ...]
    scenario_profile: ScenarioProfile


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _require_unique_strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{name} must be a non-empty list of strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(value)


def load_phase_one_profile(
    profile_name: str,
    settings_path: Path = DEFAULT_SETTINGS_PATH,
    scenario_profile_name: str | None = None,
) -> PhaseOneProfile:
    """Return one validated profile and its referenced prefix variants."""

    with settings_path.open("rb") as stream:
        payload = tomllib.load(stream)
    profiles = _require_mapping(payload.get("profiles"), "profiles")
    prefix_tables = _require_mapping(
        payload.get("prefix_variants"), "prefix_variants"
    )
    scenario_tables = _require_mapping(
        payload.get(
            "scenario_profiles",
            {
                "current": {
                    "assignment": "eligible_random",
                    "development_scenarios": 5,
                    "locked_scenarios": 20,
                    "seed": 20260814,
                }
            },
        ),
        "scenario_profiles",
    )
    try:
        profile = _require_mapping(profiles[profile_name], f"profiles.{profile_name}")
    except KeyError as error:
        available = ", ".join(sorted(profiles))
        raise ValueError(
            f"Unknown Phase 1 profile {profile_name!r}; available: {available}"
        ) from error

    feature_profile = profile.get("feature_profile")
    if feature_profile not in SUPPORTED_FEATURE_PROFILES:
        raise ValueError(
            f"profiles.{profile_name}.feature_profile must be one of "
            f"{sorted(SUPPORTED_FEATURE_PROFILES)}"
        )
    feature_sets = _require_unique_strings(
        profile.get("feature_sets"), f"profiles.{profile_name}.feature_sets"
    )
    variant_names = _require_unique_strings(
        profile.get("prefix_variants"),
        f"profiles.{profile_name}.prefix_variants",
    )
    scenario_name = scenario_profile_name or profile.get("scenario_profile", "current")
    if not isinstance(scenario_name, str) or not scenario_name:
        raise ValueError(
            f"profiles.{profile_name}.scenario_profile must be a non-empty string"
        )
    try:
        scenario_table = _require_mapping(
            scenario_tables[scenario_name],
            f"scenario_profiles.{scenario_name}",
        )
    except KeyError as error:
        raise ValueError(
            f"Profile {profile_name!r} references unknown scenario profile "
            f"{scenario_name!r}"
        ) from error
    assignment = scenario_table.get("assignment")
    if assignment not in SUPPORTED_SCENARIO_ASSIGNMENTS:
        raise ValueError(
            f"scenario_profiles.{scenario_name}.assignment must be one of "
            f"{sorted(SUPPORTED_SCENARIO_ASSIGNMENTS)}"
        )

    def positive_integer(table: dict[str, Any], field: str) -> int:
        value = table.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"scenario_profiles.{scenario_name}.{field} must be positive"
            )
        return value

    scenario_seed = scenario_table.get("seed")
    if not isinstance(scenario_seed, int) or isinstance(scenario_seed, bool):
        raise ValueError(
            f"scenario_profiles.{scenario_name}.seed must be an integer"
        )
    rul_bounds: dict[str, int | None] = {}
    for field in ("minimum_rul", "maximum_rul"):
        value = scenario_table.get(field)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ValueError(
                f"scenario_profiles.{scenario_name}.{field} must be positive"
            )
        rul_bounds[field] = value
    if (
        rul_bounds["minimum_rul"] is not None
        and rul_bounds["maximum_rul"] is not None
        and rul_bounds["minimum_rul"] > rul_bounds["maximum_rul"]
    ):
        raise ValueError(
            f"scenario_profiles.{scenario_name} has minimum_rul above maximum_rul"
        )
    if assignment == "eligible_random" and any(
        value is not None for value in rul_bounds.values()
    ):
        raise ValueError(
            f"scenario_profiles.{scenario_name} must use bipartite assignment "
            "when RUL bounds are configured"
        )
    scenario_profile = ScenarioProfile(
        name=scenario_name,
        assignment=assignment,
        development_scenarios=positive_integer(
            scenario_table, "development_scenarios"
        ),
        locked_scenarios=positive_integer(scenario_table, "locked_scenarios"),
        seed=scenario_seed,
        minimum_rul=rul_bounds["minimum_rul"],
        maximum_rul=rul_bounds["maximum_rul"],
    )

    variants: list[PrefixVariant] = []
    for variant_name in variant_names:
        try:
            table = _require_mapping(
                prefix_tables[variant_name],
                f"prefix_variants.{variant_name}",
            )
        except KeyError as error:
            raise ValueError(
                f"Profile {profile_name!r} references unknown prefix variant "
                f"{variant_name!r}"
            ) from error
        strategy = table.get("strategy")
        if strategy not in SUPPORTED_PREFIX_STRATEGIES:
            raise ValueError(
                f"prefix_variants.{variant_name}.strategy must be one of "
                f"{sorted(SUPPORTED_PREFIX_STRATEGIES)}"
            )
        cutoffs = table.get("cutoffs_per_uav")
        seed = table.get("seed")
        if strategy in {"empirical", "stratified_empirical"} and (
            not isinstance(cutoffs, int)
            or isinstance(cutoffs, bool)
            or cutoffs <= 0
        ):
            raise ValueError(
                f"prefix_variants.{variant_name}.cutoffs_per_uav must be positive"
            )
        if strategy in {"dense_all", "dense_stride"} and cutoffs is not None:
            raise ValueError(
                f"prefix_variants.{variant_name}.cutoffs_per_uav does not apply "
                f"to {strategy!r}"
            )
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"prefix_variants.{variant_name}.seed must be an integer")
        stride = table.get("stride", 1)
        minimum_cutoff = table.get("minimum_cutoff", 1)
        for field, value in (("stride", stride), ("minimum_cutoff", minimum_cutoff)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    f"prefix_variants.{variant_name}.{field} must be positive"
                )
        if strategy != "dense_stride" and "stride" in table:
            raise ValueError(
                f"prefix_variants.{variant_name}.stride only applies to dense_stride"
            )
        variants.append(
            PrefixVariant(
                variant_name,
                strategy,
                cutoffs,
                seed,
                stride=stride,
                minimum_cutoff=minimum_cutoff,
            )
        )

    return PhaseOneProfile(
        profile_name,
        str(feature_profile),
        feature_sets,
        tuple(variants),
        scenario_profile,
    )
