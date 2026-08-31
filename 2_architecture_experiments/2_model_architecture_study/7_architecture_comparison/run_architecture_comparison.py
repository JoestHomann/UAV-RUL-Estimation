"""Run the read-only Step 7 comparison after Step 6 is fully complete."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
PHASE_DIR = STEP_DIR.parent

for dependency_dir in (STEP_DIR, PHASE_DIR):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from architecture_comparison import (  # noqa: E402
    ArchitectureComparisonAnalyzer,
    save_comparison,
)
from comparison_gate import (  # noqa: E402
    DEFAULT_SPECIFICATION_PATH,
    build_architecture_comparison_plan,
)
from run_layout import (  # noqa: E402
    STEP_7_DIRECTORY_NAME,
    step_directory_for_specification,
)


def main() -> None:
    """Validate prerequisites, calculate all views, and write comparison files."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specification",
        type=Path,
        default=DEFAULT_SPECIFICATION_PATH,
        help="Location of Step 1's generated experiment specification.",
    )
    parser.add_argument(
        "--locked-manifest",
        type=Path,
        default=None,
        help=(
            "Location of Step 6's complete locked-evaluation manifest; "
            "defaults to the current run folder."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for comparison tables, figures, and the Step 7 "
            "manifest; defaults to "
            "runs/run_<run_number>/7_architecture_comparison."
        ),
    )
    args = parser.parse_args()

    try:
        # This call examines only the settings and Step 6 manifest. The two
        # locked result tables are not opened until the completion gate passes.
        plan = build_architecture_comparison_plan(
            specification_path=args.specification,
            locked_manifest_path=args.locked_manifest,
        )
        print("Step 6 completion gate passed; loading locked result tables")
        predictions = pd.read_csv(plan.predictions_path)
        model_runs = pd.read_csv(plan.model_runs_path)
        analyzer = ArchitectureComparisonAnalyzer(predictions, model_runs, plan)
        tables = analyzer.calculate()
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = step_directory_for_specification(
                STEP_7_DIRECTORY_NAME,
                specification_path=args.specification,
            )
        manifest = save_comparison(
            tables,
            plan,
            output_dir,
            predictions=predictions,
        )
    except (
        ValueError,
        OSError,
        pd.errors.ParserError,
    ) as error:
        print(f"Architecture comparison did not run:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("Architecture comparison finished")
    print(f"Compared families: {', '.join(manifest['enabled_families'])}")
    print(
        "Saved "
        f"{(output_dir / 'architecture_comparison.csv').resolve()}"
    )


if __name__ == "__main__":
    main()
