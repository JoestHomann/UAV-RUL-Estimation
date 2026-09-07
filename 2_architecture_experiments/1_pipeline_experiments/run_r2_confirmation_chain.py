"""Run PE_20 through PE_23 in order while respecting scientific gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PIPELINE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PIPELINE_DIR.parents[1]
EXPERIMENTS = PIPELINE_DIR / "experiments"


def _command(number: int, forwarded: list[str]) -> list[str]:
    return [
        sys.executable,
        str(EXPERIMENTS / f"PE_{number}" / "run.py"),
        *forwarded,
    ]


def _run(number: int, forwarded: list[str]) -> None:
    command = _command(number, forwarded)
    print(f"Running PE_{number}: {subprocess.list2cmdline(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"PE_{number} failed with exit code {completed.returncode}")


def _winner(number: int) -> dict:
    path = EXPERIMENTS / f"PE_{number}" / "runs" / "run_1" / "reporting" / "winner_manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read PE_{number} winner manifest: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"PE_{number} winner manifest is invalid")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Print commands only")
    parser.add_argument("--status", action="store_true", help="Show all workflow states")
    parser.add_argument("--force", action="store_true", help="Restart every reached workflow")
    args = parser.parse_args()
    if args.list:
        for number in range(20, 24):
            print(subprocess.list2cmdline(_command(number, ["--list"])))
        return
    if args.status:
        for number in range(20, 24):
            _run(number, ["--status"])
        return
    forwarded = ["--force"] if args.force else []
    try:
        _run(20, forwarded)
        _run(21, forwarded)
        history = _winner(20)
        tabpfn = _winner(21)
        if not (
            history.get("promoted") is True
            and history.get("winner") == "prediction_history"
            and tabpfn.get("promoted") is True
            and tabpfn.get("winner") == "control_plus_tabpfn_full"
        ):
            print(
                "Confirmation chain stopped after PE_20 and PE_21 because both "
                "independent candidates did not pass their frozen gates.",
                flush=True,
            )
            return
        _run(22, forwarded)
        combined = _winner(22)
        if not (
            combined.get("promoted") is True
            and combined.get("winner") == "history_plus_tabpfn_full"
        ):
            print(
                "Confirmation chain stopped after PE_22 because the combined "
                "candidate did not pass its frozen gate.",
                flush=True,
            )
            return
        _run(23, forwarded)
    except RuntimeError as error:
        print(f"Confirmation chain stopped:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
