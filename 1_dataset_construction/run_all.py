"""Build and verify all dataset-construction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from common import DEFAULT_TEST_CSV, DEFAULT_TRAIN_CSV, SCRIPT_DIR
from common import save_json
from phase_1_config import DEFAULT_SETTINGS_PATH, load_phase_one_profile


RUN_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def run(command: list[str]) -> None:
    print(f"Running {' '.join(command[1:])}...", flush=True)
    subprocess.run(command, check=True)


def run_shared_foundation(
    python: str,
    dataset_arguments: list[str],
) -> None:
    """Refresh deterministic artifacts shared by every Phase 1 profile."""

    run(
        [
            python,
            str(SCRIPT_DIR / "1_structural_data_audit" / "structural_data_audit.py"),
            *dataset_arguments,
        ]
    )
    run(
        [
            python,
            str(
                SCRIPT_DIR
                / "2_UAV_grouped_validation_folds"
                / "create_uav_grouped_folds.py"
            ),
        ]
    )
    run(
        [
            python,
            str(SCRIPT_DIR / "8_validation_metrics" / "validation_metrics.py"),
        ]
    )


def write_phase_two_interface(
    variant_root: Path,
    scenario_dir: Path | None = None,
) -> Path:
    """Describe one verified Phase 1 variant for direct Phase 2 consumption."""

    step_4 = variant_root / "4_training_prefixes" / "artifacts"
    step_5 = variant_root / "5_prefix_feature_engineering" / "artifacts"
    step_6 = variant_root / "6_feature_sets" / "artifacts"
    step_7 = variant_root / "7_fold_fitted_preprocessing" / "artifacts"
    step_10 = variant_root / "10_automated_leakage_checks" / "artifacts"
    scenario_dir = scenario_dir or (
        variant_root.parent / "3_test_like_validation_scenarios" / "artifacts"
    )
    prefix_config = json.loads(
        (step_4 / "training_prefix_config.json").read_text(encoding="utf-8")
    )
    feature_config = json.loads(
        (step_6 / "feature_set_config.json").read_text(encoding="utf-8")
    )
    verification = json.loads(
        (step_10 / "verification_report.json").read_text(encoding="utf-8")
    )

    def portable(path: Path) -> str:
        return path.resolve().relative_to(SCRIPT_DIR.parent.resolve()).as_posix()

    minimum_prefixes = int(prefix_config["actual_prefixes_per_uav_minimum"])
    maximum_prefixes = int(prefix_config["actual_prefixes_per_uav_maximum"])
    prefix_expectation = (
        {"expected_prefixes_per_training_uav": minimum_prefixes}
        if minimum_prefixes == maximum_prefixes
        else {
            "minimum_prefixes_per_training_uav": minimum_prefixes,
            "maximum_prefixes_per_training_uav": maximum_prefixes,
        }
    )
    return save_json(
        {
            **prefix_expectation,
            "expected_generated_features": int(feature_config["generated_features"]),
            "expected_feature_sets": feature_config["feature_sets"],
            "training_feature_rows": int(
                verification["feature_files"]["training_features.csv.gz"]["rows"]
            ),
            "artifacts": {
                "verification_report": portable(
                    step_10 / "verification_report.json"
                ),
                "fold_config": portable(
                    SCRIPT_DIR
                    / "2_UAV_grouped_validation_folds"
                    / "artifacts"
                    / "fold_config.json"
                ),
                "outer_folds": portable(
                    SCRIPT_DIR
                    / "2_UAV_grouped_validation_folds"
                    / "artifacts"
                    / "outer_folds.csv"
                ),
                "inner_folds": portable(
                    SCRIPT_DIR
                    / "2_UAV_grouped_validation_folds"
                    / "artifacts"
                    / "inner_folds.csv"
                ),
                "scenario_config": portable(
                    scenario_dir / "scenario_config.json"
                ),
                "development_scenarios": portable(
                    scenario_dir / "development_validation_scenarios.csv"
                ),
                "locked_scenarios": portable(
                    scenario_dir / "locked_validation_scenarios.csv"
                ),
                "training_prefix_config": portable(
                    step_4 / "training_prefix_config.json"
                ),
                "training_prefixes": portable(step_4 / "training_prefixes.csv"),
                "training_features": portable(step_5 / "training_features.csv.gz"),
                "development_features": portable(
                    step_5 / "development_validation_features.csv.gz"
                ),
                "locked_features": portable(
                    step_5 / "locked_validation_features.csv.gz"
                ),
                "test_features": portable(step_5 / "test_features.csv.gz"),
                "feature_catalog": portable(step_6 / "feature_catalog.csv"),
                "preprocessing_config": portable(
                    step_7 / "preprocessing_config.json"
                ),
                "metric_specification": portable(
                    SCRIPT_DIR
                    / "8_validation_metrics"
                    / "artifacts"
                    / "metric_specification.json"
                ),
            },
        },
        variant_root / "phase_2_interface.json",
    )


def run_versioned_profile(
    *,
    profile_name: str,
    settings_path: Path,
    run_name: str,
    requested_variants: list[str] | None,
    requested_scenario_profile: str | None,
    train_csv: Path,
    test_csv: Path,
) -> None:
    """Build one or more prefix variants below a versioned Phase 1 run."""

    profile = load_phase_one_profile(
        profile_name,
        settings_path,
        scenario_profile_name=requested_scenario_profile,
    )
    variants = list(profile.prefix_variants)
    if requested_variants:
        unknown = sorted(
            set(requested_variants) - {variant.name for variant in variants}
        )
        if unknown:
            raise ValueError(
                f"Profile {profile_name!r} has no prefix variants {unknown}"
            )
        variants = [
            variant for variant in variants if variant.name in requested_variants
        ]

    dataset_arguments = [
        "--train-csv",
        str(train_csv),
        "--test-csv",
        str(test_csv),
    ]
    run_shared_foundation(sys.executable, dataset_arguments)

    run_root = SCRIPT_DIR / "runs" / run_name
    scenario = profile.scenario_profile
    scenario_dir = run_root / "3_test_like_validation_scenarios" / "artifacts"
    scenario_command = [
        sys.executable,
        str(
            SCRIPT_DIR
            / "3_test_like_validation_scenarios"
            / "create_test_like_scenarios.py"
        ),
        *dataset_arguments,
        "--output-dir",
        str(scenario_dir),
        "--assignment",
        scenario.assignment,
        "--development-scenarios",
        str(scenario.development_scenarios),
        "--locked-scenarios",
        str(scenario.locked_scenarios),
        "--seed",
        str(scenario.seed),
    ]
    if scenario.minimum_rul is not None:
        scenario_command.extend(["--minimum-rul", str(scenario.minimum_rul)])
    if scenario.maximum_rul is not None:
        scenario_command.extend(["--maximum-rul", str(scenario.maximum_rul)])
    run(scenario_command)

    manifests: list[dict[str, object]] = []
    for variant in variants:
        variant_root = run_root / variant.name
        step_4 = variant_root / "4_training_prefixes" / "artifacts"
        step_5 = variant_root / "5_prefix_feature_engineering" / "artifacts"
        step_6 = variant_root / "6_feature_sets" / "artifacts"
        step_7 = variant_root / "7_fold_fitted_preprocessing" / "artifacts"
        step_9 = variant_root / "9_cycle_only_baseline" / "artifacts"
        step_10 = variant_root / "10_automated_leakage_checks" / "artifacts"

        prefix_command = [
            sys.executable,
            str(
                SCRIPT_DIR
                / "4_training_prefixes"
                / "create_training_prefixes.py"
            ),
            "--train-csv",
            str(train_csv),
            "--output-dir",
            str(step_4),
            "--seed",
            str(variant.seed),
            "--strategy",
            variant.strategy,
            "--stride",
            str(variant.stride),
            "--minimum-cutoff",
            str(variant.minimum_cutoff),
        ]
        if variant.cutoffs_per_uav is not None:
            prefix_command.extend(
                ["--cutoffs-per-uav", str(variant.cutoffs_per_uav)]
            )
        run(prefix_command)
        run(
            [
                sys.executable,
                str(
                    SCRIPT_DIR
                    / "5_prefix_feature_engineering"
                    / "build_prefix_features.py"
                ),
                *dataset_arguments,
                "--prefix-dir",
                str(step_4),
                "--scenario-dir",
                str(scenario_dir),
                "--output-dir",
                str(step_5),
                "--feature-profile",
                profile.feature_profile,
            ]
        )
        run(
            [
                sys.executable,
                str(SCRIPT_DIR / "6_feature_sets" / "define_feature_sets.py"),
                "--feature-dir",
                str(step_5),
                "--output-dir",
                str(step_6),
                "--settings",
                str(settings_path),
                "--profile",
                profile_name,
            ]
        )
        run(
            [
                sys.executable,
                str(
                    SCRIPT_DIR
                    / "7_fold_fitted_preprocessing"
                    / "preprocessing.py"
                ),
                "--feature-dir",
                str(step_5),
                "--feature-set-dir",
                str(step_6),
                "--output-dir",
                str(step_7),
            ]
        )
        run(
            [
                sys.executable,
                str(SCRIPT_DIR / "9_cycle_only_baseline" / "cycle_baseline.py"),
                "--feature-dir",
                str(step_5),
                "--output-dir",
                str(step_9),
            ]
        )
        run(
            [
                sys.executable,
                str(
                    SCRIPT_DIR
                    / "10_automated_leakage_checks"
                    / "verify_phase1.py"
                ),
                "--train-csv",
                str(train_csv),
                "--prefix-dir",
                str(step_4),
                "--feature-dir",
                str(step_5),
                "--feature-set-dir",
                str(step_6),
                "--preprocessing-dir",
                str(step_7),
                "--baseline-dir",
                str(step_9),
                "--output-dir",
                str(step_10),
                "--feature-profile",
                profile.feature_profile,
                "--scenario-dir",
                str(scenario_dir),
            ]
        )
        phase_2_interface_path = write_phase_two_interface(
            variant_root,
            scenario_dir,
        )
        manifests.append(
            {
                "prefix_variant": variant.name,
                "strategy": variant.strategy,
                "configured_cutoffs_per_uav": variant.cutoffs_per_uav,
                "configured_stride": variant.stride,
                "configured_minimum_cutoff": variant.minimum_cutoff,
                "artifact_root": str(variant_root),
                "verification_report": str(
                    step_10 / "verification_report.json"
                ),
                "phase_2_interface": str(phase_2_interface_path),
            }
        )

    manifest_path = run_root / "phase_1_run_manifest.json"
    merged_variants: dict[str, dict[str, object]] = {}
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        previous_run_name = previous.get("run_name")
        if previous_run_name is None and isinstance(
            previous.get("run_number"), int
        ):
            previous_run_name = f"run_{previous['run_number']}"
        if previous_run_name == run_name and previous.get("profile") == profile.name:
            merged_variants.update(
                {
                    str(item["prefix_variant"]): item
                    for item in previous.get("variants", [])
                    if isinstance(item, dict) and "prefix_variant" in item
                }
            )
    merged_variants.update(
        {str(item["prefix_variant"]): item for item in manifests}
    )
    ordered_variants = [
        merged_variants[variant.name]
        for variant in profile.prefix_variants
        if variant.name in merged_variants
    ]
    path = save_json(
        {
            "settings_version": 1,
            "run_name": run_name,
            "profile": profile.name,
            "feature_profile": profile.feature_profile,
            "feature_sets": list(profile.feature_sets),
            "scenario_profile": {
                "name": scenario.name,
                "assignment": scenario.assignment,
                "development_scenarios": scenario.development_scenarios,
                "locked_scenarios": scenario.locked_scenarios,
                "seed": scenario.seed,
                "minimum_rul": scenario.minimum_rul,
                "maximum_rul": scenario.maximum_rul,
                "artifact_root": str(scenario_dir),
            },
            "variants": ordered_variants,
        },
        manifest_path,
    )
    print(f"Versioned Phase 1 artifacts verified: {path}")


def refresh_versioned_interfaces(
    *,
    profile_name: str,
    settings_path: Path,
    run_name: str,
    requested_variants: list[str] | None,
    requested_scenario_profile: str | None,
) -> None:
    """Refresh Phase 2 contracts from already-verified Phase 1 artifacts."""

    profile = load_phase_one_profile(
        profile_name,
        settings_path,
        scenario_profile_name=requested_scenario_profile,
    )
    selected = set(requested_variants or [item.name for item in profile.prefix_variants])
    known = {item.name for item in profile.prefix_variants}
    unknown = sorted(selected - known)
    if unknown:
        raise ValueError(f"Profile {profile_name!r} has no prefix variants {unknown}")

    run_root = SCRIPT_DIR / "runs" / run_name
    scenario_dir = run_root / "3_test_like_validation_scenarios" / "artifacts"
    if not scenario_dir.is_dir():
        scenario_dir = (
            SCRIPT_DIR / "3_test_like_validation_scenarios" / "artifacts"
        )
    manifest_path = run_root / "phase_1_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {
        str(item["prefix_variant"]): item
        for item in manifest.get("variants", [])
        if isinstance(item, dict) and "prefix_variant" in item
    }
    for variant in profile.prefix_variants:
        if variant.name not in selected:
            continue
        variant_root = run_root / variant.name
        interface_path = write_phase_two_interface(variant_root, scenario_dir)
        if variant.name not in entries:
            raise ValueError(
                f"Run manifest has no entry for prefix variant {variant.name!r}"
            )
        entries[variant.name].update(
            {
                "strategy": variant.strategy,
                "configured_cutoffs_per_uav": variant.cutoffs_per_uav,
                "configured_stride": variant.stride,
                "configured_minimum_cutoff": variant.minimum_cutoff,
                "artifact_root": str(variant_root),
                "verification_report": str(
                    variant_root
                    / "10_automated_leakage_checks"
                    / "artifacts"
                    / "verification_report.json"
                ),
                "phase_2_interface": str(interface_path),
            }
        )

    manifest["variants"] = [
        entries[item.name]
        for item in profile.prefix_variants
        if item.name in entries
    ]
    manifest.pop("run_number", None)
    manifest.update(
        {
            "run_name": run_name,
            "profile": profile.name,
            "feature_profile": profile.feature_profile,
            "feature_sets": list(profile.feature_sets),
            "scenario_profile": {
                "name": profile.scenario_profile.name,
                "assignment": profile.scenario_profile.assignment,
                "development_scenarios": (
                    profile.scenario_profile.development_scenarios
                ),
                "locked_scenarios": profile.scenario_profile.locked_scenarios,
                "seed": profile.scenario_profile.seed,
                "minimum_rul": profile.scenario_profile.minimum_rul,
                "maximum_rul": profile.scenario_profile.maximum_rul,
                "artifact_root": str(scenario_dir),
            },
        }
    )
    save_json(manifest, manifest_path)
    print(f"Phase 2 interfaces refreshed: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--profile", default="legacy")
    parser.add_argument("--run-number", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--prefix-variant", nargs="+")
    parser.add_argument(
        "--scenario-profile",
        help="Select a named scenario profile from the Phase 1 settings.",
    )
    parser.add_argument(
        "--refresh-interface",
        action="store_true",
        help="Refresh Phase 2 contracts without rebuilding Phase 1 artifacts",
    )
    args = parser.parse_args()

    if args.refresh_interface and args.profile == "legacy":
        parser.error("--refresh-interface is only available for versioned profiles")
    if args.profile != "legacy":
        if args.run_name is not None and args.run_number is not None:
            parser.error("declare either --run-name or --run-number, not both")
        run_name = args.run_name
        if run_name is None and args.run_number is not None:
            if args.run_number <= 0:
                parser.error("--run-number must be positive")
            run_name = f"run_{args.run_number}"
        if run_name is None:
            parser.error("--run-name or --run-number is required")
        if not RUN_NAME_PATTERN.fullmatch(run_name):
            parser.error(
                "--run-name must start with a letter and contain only "
                "letters, digits, underscores, or hyphens"
            )
        if args.refresh_interface:
            refresh_versioned_interfaces(
                profile_name=args.profile,
                settings_path=args.settings,
                run_name=run_name,
                requested_variants=args.prefix_variant,
                requested_scenario_profile=args.scenario_profile,
            )
            return
        run_versioned_profile(
            profile_name=args.profile,
            settings_path=args.settings,
            run_name=run_name,
            requested_variants=args.prefix_variant,
            requested_scenario_profile=args.scenario_profile,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
        )
        return

    dataset_arguments = [
        "--train-csv",
        str(args.train_csv),
        "--test-csv",
        str(args.test_csv),
    ]
    run(
        [
            sys.executable,
            str(
                SCRIPT_DIR
                / "1_structural_data_audit"
                / "structural_data_audit.py"
            ),
            *dataset_arguments,
        ]
    )
    run(
        [
            sys.executable,
            str(
                SCRIPT_DIR
                / "2_UAV_grouped_validation_folds"
                / "create_uav_grouped_folds.py"
            ),
        ]
    )
    run(
        [
            sys.executable,
            str(
                SCRIPT_DIR
                / "3_test_like_validation_scenarios"
                / "create_test_like_scenarios.py"
            ),
            *dataset_arguments,
        ]
    )
    run(
        [
            sys.executable,
            str(
                SCRIPT_DIR
                / "4_training_prefixes"
                / "create_training_prefixes.py"
            ),
            "--train-csv",
            str(args.train_csv),
        ]
    )
    run(
        [
            sys.executable,
            str(
                SCRIPT_DIR
                / "5_prefix_feature_engineering"
                / "build_prefix_features.py"
            ),
            *dataset_arguments,
        ]
    )
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "6_feature_sets" / "define_feature_sets.py"),
        ]
    )
    run(
        [
            sys.executable,
            str(
                SCRIPT_DIR
                / "7_fold_fitted_preprocessing"
                / "preprocessing.py"
            ),
        ]
    )
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "8_validation_metrics" / "validation_metrics.py"),
        ]
    )
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "9_cycle_only_baseline" / "cycle_baseline.py"),
        ]
    )
    run(
        [
            sys.executable,
            str(
                SCRIPT_DIR
                / "10_automated_leakage_checks"
                / "verify_phase1.py"
            ),
            "--train-csv",
            str(args.train_csv),
        ]
    )
    print(f"Dataset-construction artifacts verified under {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
