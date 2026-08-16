"""Run Step 5 automatic tuning within enabled architecture families."""

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
DEFAULT_SPECIFICATION_PATH = (
    STEP_DIR.parent
    / "1_architecture_study_settings"
    / "artifacts"
    / "experiment_specification.json"
)
DEFAULT_OUTPUT_DIR = STEP_DIR / "artifacts"

if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from inner_model_selection import (  # noqa: E402
    InnerModelSelectionError,
    InnerModelSelectionRunner,
)


def main() -> None:
    """Parse optional study filters and run the requested Step 5 work."""

    # This process may run alongside sibling Step 5 processes dispatched by
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Step 5 checkpoints and consolidated results.",
    )
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        help=(
            "Run one enabled model family. Repeat the option for multiple "
            "families; omit it to run every enabled family."
        ),
    )
    parser.add_argument(
        "--outer-fold",
        action="append",
        type=int,
        dest="outer_folds",
        help=(
            "Run one outer fold. Repeat the option for multiple folds; omit "
            "it to run all configured outer folds."
        ),
    )
    args = parser.parse_args()

    try:
        runner = InnerModelSelectionRunner(
            specification_path=args.specification,
            output_dir=args.output_dir,
        )
        families, outer_folds = runner.validate_request(
            args.families,
            args.outer_folds,
        )
        neural_device = "cuda" if torch.cuda.is_available() else "cpu"
        device_detail = (
            f" ({torch.cuda.get_device_name(0)})" if neural_device == "cuda" else ""
        )
        print(f"Neural training device: {neural_device}{device_detail}")
        print("Starting leakage-safe inner model selection")
        print(f"Families: {', '.join(families)}")
        print(f"Outer folds: {outer_folds}")
        print(f"Independent studies requested: {len(families) * len(outer_folds)}")
        manifest = runner.run(families, outer_folds)
    except ValueError as error:
        print(f"Inner model selection failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("Inner model selection request finished")
    print(f"Overall Step 5 status: {manifest['status']}")
    print(
        "Completed studies: "
        f"{manifest['completed_study_count']}/"
        f"{manifest['expected_study_count']}"
    )
    print(
        "Saved "
        f"{(args.output_dir / 'selected_configurations.csv').resolve()}"
    )


if __name__ == "__main__":
    main()
