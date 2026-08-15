"""Run Step 6 only after every Step 5 family/fold study is complete."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


STEP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = STEP_DIR / "artifacts"

if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from evaluation_gate import (  # noqa: E402
    DEFAULT_SELECTED_CONFIGURATIONS_PATH,
    DEFAULT_SELECTION_MANIFEST_PATH,
    DEFAULT_SPECIFICATION_PATH,
)
from locked_outer_evaluation import LockedOuterEvaluationRunner  # noqa: E402


def main() -> None:
    """Parse optional run filters and execute locked evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specification",
        type=Path,
        default=DEFAULT_SPECIFICATION_PATH,
        help="Location of Step 1's generated experiment specification.",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST_PATH,
        help="Location of Step 5's complete selection manifest.",
    )
    parser.add_argument(
        "--selected-configurations",
        type=Path,
        default=DEFAULT_SELECTED_CONFIGURATIONS_PATH,
        help="Location of Step 5's selected configuration table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for models, predictions, run facts, and checkpoints.",
    )
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        help=(
            "Evaluate one enabled family. Repeat for several families; omit "
            "the option to evaluate every family."
        ),
    )
    parser.add_argument(
        "--outer-fold",
        action="append",
        type=int,
        dest="outer_folds",
        help=(
            "Evaluate one outer fold. Repeat for several folds; omit the "
            "option to evaluate every outer fold."
        ),
    )
    args = parser.parse_args()

    try:
        runner = LockedOuterEvaluationRunner(
            specification_path=args.specification,
            selection_manifest_path=args.selection_manifest,
            selected_configurations_path=args.selected_configurations,
            output_dir=args.output_dir,
        )
        families, outer_folds = runner.validate_request(
            args.families,
            args.outer_folds,
        )
        run_count = sum(
            len(runner.plan.seeds_for(family)) * len(outer_folds)
            for family in families
        )
        print("Step 5 gate passed; starting locked outer evaluation")
        print(f"Families: {', '.join(families)}")
        print(f"Outer folds: {outer_folds}")
        print(f"Family/fold/seed runs requested: {run_count}")
        manifest = runner.run(families, outer_folds)
    except (ValueError, OSError) as error:
        print(f"Locked outer evaluation did not run:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("Locked outer evaluation request finished")
    print(f"Overall Step 6 status: {manifest['status']}")
    print(
        "Completed runs: "
        f"{manifest['completed_run_count']}/"
        f"{manifest['expected_run_count']}"
    )
    print(
        "Saved "
        f"{(args.output_dir / 'locked_predictions.csv.gz').resolve()}"
    )


if __name__ == "__main__":
    main()
