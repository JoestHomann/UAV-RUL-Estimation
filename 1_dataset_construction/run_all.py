"""Build and verify all dataset-construction artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from common import DEFAULT_TEST_CSV, DEFAULT_TRAIN_CSV, SCRIPT_DIR


def run(command: list[str]) -> None:
    print(f"Running {' '.join(command[1:])}...", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    args = parser.parse_args()

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
