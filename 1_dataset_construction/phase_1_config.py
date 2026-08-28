"""Load and validate the versioned Phase 1 experiment settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS_PATH = SCRIPT_DIR / "phase_1_settings.toml"
SUPPORTED_FEATURE_PROFILES = {"legacy", "extended"}
SUPPORTED_PREFIX_STRATEGIES = {"empirical", "stratified_empirical"}


@dataclass(frozen=True)
class PrefixVariant:
    name: str
    strategy: str
    cutoffs_per_uav: int
    seed: int


@dataclass(frozen=True)
class PhaseOneProfile:
    name: str
    feature_profile: str
    feature_sets: tuple[str, ...]
    prefix_variants: tuple[PrefixVariant, ...]


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
) -> PhaseOneProfile:
    """Return one validated profile and its referenced prefix variants."""

    with settings_path.open("rb") as stream:
        payload = tomllib.load(stream)
    profiles = _require_mapping(payload.get("profiles"), "profiles")
    prefix_tables = _require_mapping(
        payload.get("prefix_variants"), "prefix_variants"
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
        if not isinstance(cutoffs, int) or isinstance(cutoffs, bool) or cutoffs <= 0:
            raise ValueError(
                f"prefix_variants.{variant_name}.cutoffs_per_uav must be positive"
            )
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"prefix_variants.{variant_name}.seed must be an integer")
        variants.append(PrefixVariant(variant_name, strategy, cutoffs, seed))

    return PhaseOneProfile(
        profile_name,
        str(feature_profile),
        feature_sets,
        tuple(variants),
    )
