"""Run the complete core data-analysis suite in dependency order."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from core_common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
    SCRIPT_DIR,
    TELEMETRY_COLUMNS,
)


ANALYSES = [
    ("temporal_rul_analysis.py", "temporal_rul", False),
    ("representative_trajectories.py", "representative_trajectories", True),
    ("feature_redundancy.py", "feature_redundancy", False),
    ("train_test_drift.py", "train_test_drift", True),
    ("anomaly_analysis.py", "anomalies", True),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=TELEMETRY_COLUMNS,
        default=TELEMETRY_COLUMNS,
    )
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    for script_name, output_name, needs_test in ANALYSES:
        command = [
            sys.executable,
            str(SCRIPT_DIR / script_name),
            "--train-csv",
            str(args.train_csv),
        ]
        if needs_test:
            command.extend(["--test-csv", str(args.test_csv)])
        command.extend(
            [
            "--output-dir",
            str(args.output_root / output_name),
            "--dpi",
            str(args.dpi),
            "--channels",
            *args.channels,
            ]
        )
        print(f"Running {script_name}...", flush=True)
        subprocess.run(command, check=True)

    classification_command = [
        sys.executable,
        str(SCRIPT_DIR / "channel_classification.py"),
        "--train-csv",
        str(args.train_csv),
        "--input-root",
        str(args.output_root),
        "--output-dir",
        str(args.output_root / "channel_classification"),
        "--dpi",
        str(args.dpi),
        "--channels",
        *args.channels,
    ]
    print("Running channel_classification.py...", flush=True)
    subprocess.run(classification_command, check=True)
    print(f"Core data analysis complete: {args.output_root}")


if __name__ == "__main__":
    main()
