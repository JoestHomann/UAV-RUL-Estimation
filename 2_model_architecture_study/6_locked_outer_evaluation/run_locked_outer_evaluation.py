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

import ctypes
from pathlib import Path
import sys

import torch


STEP_DIR = Path(__file__).resolve().parent

if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from evaluation_gate import (  # noqa: E402
    DEFAULT_SPECIFICATION_PATH,
)
from locked_outer_evaluation import LockedOuterEvaluationRunner  # noqa: E402


def _exit_without_interpreter_shutdown(status: int) -> None:
    """Leave the process as soon as every artifact is safely on disk.

    PyTorch's Windows build intermittently fast-fails while its native
    libraries are torn down at the very end of a process, which the operating
    system reports as exit code 3221226505 (0xC0000409,
    STATUS_STACK_BUFFER_OVERRUN). That teardown happens after ``main`` has
    returned, so this request's checkpoints, predictions, run facts, and
    manifest are all already written -- but run_phase_2.py can only observe
    the exit code, so it records finished work as failed and stops the whole
    pipeline.

    ``os._exit`` alone is not enough to avoid this. It skips Python's
    interpreter finalization but still reaches the Win32 ``ExitProcess``,
    which abruptly terminates the remaining threads and then calls every
    loaded DLL's detach handler -- exactly the native teardown that
    fast-fails. ``TerminateProcess`` is documented not to notify attached
    DLLs, so calling it on this process is the one self-exit that leaves the
    reported status ours rather than the crash code. ``os._exit`` stays as
    the fallback for other platforms, and as an unreachable safety net if
    the call ever fails.

    Bypassing teardown is safe here because nothing in this step depends on
    it: there is no atexit handler and no ``__del__`` side effect, every
    artifact is written by an atomic replace long before this point, and the
    TensorBoard writer flushes each event group as it is written and is
    closed by its own context manager. The two standard streams are flushed
    explicitly below, since neither exit path performs the flush that normal
    shutdown would, which would otherwise drop this run's console output
    whenever stdout is a pipe rather than a console.
    """

    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), status)
    os._exit(status)


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
        default=None,
        help=(
            "Location of Step 5's complete selection manifest; defaults to "
            "the current run folder."
        ),
    )
    parser.add_argument(
        "--selected-configurations",
        type=Path,
        default=None,
        help=(
            "Location of Step 5's selected configuration table; defaults to "
            "the current run folder."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for models, predictions, run facts, and checkpoints; "
            "defaults to runs/run_<run_number>/6_locked_outer_evaluation."
        ),
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
    # A completed run and a deliberate SystemExit both leave through the hard
    # exit. An unexpected exception is deliberately left to propagate so its
    # traceback still reaches the operator; Python then reports a non-zero
    # status either way, so run_phase_2.py still treats that run as failed.
    try:
        main()
    except SystemExit as error:
        # Only main's SystemExit(1) and argparse's status reach this, so the
        # code is None or an integer.
        _exit_without_interpreter_shutdown(0 if error.code is None else int(error.code))
    _exit_without_interpreter_shutdown(0)
