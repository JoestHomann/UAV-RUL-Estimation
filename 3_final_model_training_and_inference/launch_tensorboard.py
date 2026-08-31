"""Launch TensorBoard for one Phase 3 run or the complete Phase 3 run tree."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from threading import Event


PHASE_DIR = Path(__file__).resolve().parent
PHASE_2_DIR = (
    PHASE_DIR.parent
    / "2_architecture_experiments"
    / "2_model_architecture_study"
)
for dependency_dir in (PHASE_DIR, PHASE_2_DIR):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from phase_3_run_layout import (  # noqa: E402
    RUNS_DIR,
    Phase3RunLayoutError,
    read_run_number,
    tensorboard_log_root,
)
from tensorboard_monitoring import ensure_tensorboard_available  # noqa: E402


def _resolve_log_directory(
    run: int | None,
    all_runs: bool,
) -> tuple[Path, str]:
    if all_runs:
        return RUNS_DIR, "every Phase 3 run"
    selected_run = read_run_number() if run is None else run
    return tensorboard_log_root(selected_run), f"Phase 3 run {selected_run}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--run", type=int, default=None)
    parser.add_argument("--all-runs", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.all_runs and args.run is not None:
        parser.error("--all-runs and --run cannot be combined")

    ensure_tensorboard_available()
    try:
        log_directory, description = _resolve_log_directory(args.run, args.all_runs)
    except Phase3RunLayoutError as error:
        print(f"Cannot start TensorBoard:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    from tensorboard import program

    log_directory.mkdir(parents=True, exist_ok=True)
    dashboard = program.TensorBoard()
    dashboard.configure(
        argv=[None, "--logdir", str(log_directory), "--port", str(args.port)]
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
