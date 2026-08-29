"""Record a manually entered competition score for one experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tomllib
from typing import Any


MANAGER_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MANAGER_DIR.parent
DEFAULT_CONFIG_PATH = MANAGER_DIR / "pipeline_experiments.toml"


def _read_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Cannot read experiment catalog: {error}") from error


def record(
    config_path: Path,
    run_name: str,
    public_score: float,
    metric: str,
    description: str | None,
    notes: str | None,
) -> Path:
    config = _read_config(config_path)
    experiments = config.get("experiments", {})
    if not isinstance(experiments, dict) or run_name not in experiments:
        raise ValueError(f"Unknown experiment {run_name!r}")
    if not math.isfinite(public_score):
        raise ValueError("public score must be finite")
    path = MANAGER_DIR / "runs" / run_name / "leaderboard_result.json"
    payload = {
        "record_version": 1,
        "experiment": run_name,
        "metric": metric,
        "public_score": public_score,
        "submission_description": description,
        "notes": notes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run", required=True)
    parser.add_argument("--public-score", type=float, required=True)
    parser.add_argument("--metric", default="r2")
    parser.add_argument("--submission-description")
    parser.add_argument("--notes")
    args = parser.parse_args()
    try:
        path = record(
            args.config.resolve(),
            args.run,
            args.public_score,
            args.metric,
            args.submission_description,
            args.notes,
        )
    except (ValueError, OSError) as error:
        print(f"Leaderboard recording failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"Saved {path.resolve()}")


if __name__ == "__main__":
    main()
