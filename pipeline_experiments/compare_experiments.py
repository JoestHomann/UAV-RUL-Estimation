"""Compare completed pipeline experiments without reopening locked data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib
from typing import Any

import pandas as pd


MANAGER_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MANAGER_DIR.parent
DEFAULT_CONFIG_PATH = MANAGER_DIR / "pipeline_experiments.toml"
OUTPUT_PATH = MANAGER_DIR / "experiment_comparison.csv"


def _read_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Cannot read experiment catalog: {error}") from error


def _repo_path(value: str) -> Path:
    path = (REPOSITORY_ROOT / value).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes the repository: {value}") from error
    return path


def compare(config_path: Path, names: list[str] | None = None) -> pd.DataFrame:
    config = _read_config(config_path)
    experiments = config.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ValueError("The catalog has no experiments table")
    selected = sorted(names or experiments)
    rows: list[pd.DataFrame] = []
    for name in selected:
        experiment = experiments.get(name)
        if not isinstance(experiment, dict):
            raise ValueError(f"Unknown experiment {name!r}")
        run_number = experiment.get("phase_2_run_number")
        path = (
            REPOSITORY_ROOT
            / "2_model_architecture_study"
            / "runs"
            / f"run_{run_number}"
            / "7_architecture_comparison"
            / "architecture_comparison.csv"
        )
        if not path.is_file():
            rows.append(
                pd.DataFrame(
                    [{"experiment": name, "status": "not_complete", "phase_2_run_number": run_number}]
                )
            )
            continue
        table = pd.read_csv(path)
        table.insert(0, "experiment", name)
        table.insert(1, "status", "complete")
        table.insert(2, "phase_2_run_number", run_number)
        rows.append(table)

        leaderboard_path = MANAGER_DIR / "runs" / name / "leaderboard_result.json"
        if leaderboard_path.is_file():
            record = json.loads(leaderboard_path.read_text(encoding="utf-8"))
            table["leaderboard_metric"] = record.get("metric")
            table["leaderboard_score"] = record.get("public_score")
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True, sort=False)
    result.to_csv(OUTPUT_PATH, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run", action="append", dest="names")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    try:
        result = compare(args.config.resolve(), args.names)
        if args.output.resolve() != OUTPUT_PATH.resolve():
            result.to_csv(args.output, index=False)
    except (ValueError, OSError, pd.errors.ParserError, json.JSONDecodeError) as error:
        print(f"Experiment comparison failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"Compared {len(result)} rows")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
