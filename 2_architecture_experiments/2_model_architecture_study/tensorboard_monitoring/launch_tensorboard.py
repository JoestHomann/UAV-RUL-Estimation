"""Launch TensorBoard on one numbered run's event directory, or on all of them."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from threading import Event

PHASE_DIR = Path(__file__).resolve().parent.parent
if str(PHASE_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE_DIR))

from run_layout import (  # noqa: E402
    RUNS_DIR,
    RunLayoutError,
    read_run_number,
    tensorboard_log_root,
)
from tensorboard_monitoring import ensure_tensorboard_available  # noqa: E402


def _resolve_log_directory(
    run: int | None,
    all_runs: bool,
) -> tuple[Path, str]:
    """Choose which event directory the dashboard opens on.

    Defaulting to the run the settings currently select means the dashboard
    shows the run the pipeline is actually writing, without having to be told
    which one that is.
    """

    if all_runs:
        # Pointed at the shared runs directory, TensorBoard discovers every
        # run's events and labels each series with its run folder, which is
        # what makes two runs comparable in one view.
        return RUNS_DIR, "every run"
    if run is None:
        run = read_run_number()
    return tensorboard_log_root(run), f"run {run}"


def main() -> None:
    """Start the dashboard and keep it alive until the user presses Ctrl+C."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="Local TensorBoard port; defaults to 6006.",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help=(
            "Show one numbered run; defaults to the run_number in the current "
            "architecture study settings."
        ),
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Show every run at once so their curves can be compared.",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.all_runs and args.run is not None:
        parser.error("--all-runs and --run cannot be combined")

    ensure_tensorboard_available()
    try:
        log_directory, description = _resolve_log_directory(args.run, args.all_runs)
    except RunLayoutError as error:
        print(f"Cannot start TensorBoard:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    from tensorboard import program

    # Created if absent so the dashboard can be opened before, or alongside, the
    # first study that writes into it.
    log_directory.mkdir(parents=True, exist_ok=True)
    dashboard = program.TensorBoard()
    dashboard.configure(
        argv=[
            None,
            "--logdir",
            str(log_directory),
            "--port",
            str(args.port),
        ]
    )
    url = dashboard.launch()
    print(f"TensorBoard is monitoring {description}: {log_directory}")
    print(f"Open {url}")
    print("Press Ctrl+C to stop the dashboard")
    try:
        while True:
            Event().wait(1.0)
    except KeyboardInterrupt:
        print("\nTensorBoard stopped")


if __name__ == "__main__":
    main()
