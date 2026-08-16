"""Run Step 6 only after every Step 5 family/fold study is complete."""

from __future__ import annotations

import argparse
import os

# CUDA's deterministic cuBLAS kernels only exist when this variable is set
# before PyTorch creates a CUDA context. It must therefore be set before
# `import torch`, not inside the adapter that later calls
# torch.use_deterministic_algorithms(True) -- by then a context may already
# exist, and the setting would silently have no effect. setdefault leaves
# an operator-supplied value untouched. This has no effect at all on
# CPU-only machines.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
import sys

import torch


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

    # This process may run alongside sibling Step 6 processes dispatched by
    # run_phase_2.py's parallel study fan-out. Without an explicit cap,
    # PyTorch defaults to one intra-op thread per core, so several such
    # processes running at once would oversubscribe the machine instead of
    # speeding it up. This mirrors the codebase's existing n_jobs=1 choice
    # for XGBoost and Random Forest, applied unconditionally so behavior does
    # not change based on how many studies happen to run at the same time.
    torch.set_num_threads(1)

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
        neural_device = "cuda" if torch.cuda.is_available() else "cpu"
        device_detail = (
            f" ({torch.cuda.get_device_name(0)})" if neural_device == "cuda" else ""
        )
        print(f"Neural training device: {neural_device}{device_detail}")
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
