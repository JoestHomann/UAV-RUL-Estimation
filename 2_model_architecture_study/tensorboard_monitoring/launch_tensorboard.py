"""Launch TensorBoard for the stable Phase 2 event-log directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from threading import Event

PHASE_DIR = Path(__file__).resolve().parent.parent
if str(PHASE_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE_DIR))

from tensorboard_monitoring import DEFAULT_LOG_ROOT, ensure_tensorboard_available


def main() -> None:
    """Start the dashboard and keep it alive until the user presses Ctrl+C."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="Local TensorBoard port; defaults to 6006.",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    ensure_tensorboard_available()
    from tensorboard import program

    DEFAULT_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    dashboard = program.TensorBoard()
    dashboard.configure(
        argv=[
            None,
            "--logdir",
            str(DEFAULT_LOG_ROOT),
            "--port",
            str(args.port),
        ]
    )
    url = dashboard.launch()
    print(f"TensorBoard is monitoring {DEFAULT_LOG_ROOT}")
    print(f"Open {url}")
    print("Press Ctrl+C to stop the dashboard")
    try:
        Event().wait()
    except KeyboardInterrupt:
        print("\nTensorBoard stopped")


if __name__ == "__main__":
    main()
