"""Run PE_6 sampling, propagate its winner, and compare lookbacks."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_config import read_experiment_config  # noqa: E402
from experiment_paths import run_directory  # noqa: E402
from report_temporal_sampling import report  # noqa: E402


EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]


class TemporalWorkflowError(ValueError):
    """Explain an invalid PE_6 definition or failed child execution."""


def _run_cell(config_path: Path, name: str) -> None:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_experiments.py"),
        "--config",
        str(config_path),
        "--run",
        name,
    ]
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if completed.returncode != 0:
        raise TemporalWorkflowError(f"Temporal sampling cell {name} failed")


def _workflow(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get("temporal_sampling_workflows", {}).get(name)
    if not isinstance(value, dict):
        raise TemporalWorkflowError(f"Unknown temporal workflow {name!r}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_6")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = read_experiment_config(config_path)
    workflow = _workflow(config, args.workflow)
    definition = config.get("run_definitions", {}).get(args.workflow)
    if not isinstance(definition, dict):
        raise TemporalWorkflowError("PE_6 run definition is missing")
    run_root = run_directory(EXPERIMENTS_DIR, args.workflow, definition)
    run_root.mkdir(parents=True, exist_ok=True)

    sampling_cells = workflow.get("sampling_cells")
    lookback_cells = workflow.get("lookback_cells")
    if not isinstance(sampling_cells, list) or not isinstance(lookback_cells, list):
        raise TemporalWorkflowError("PE_6 cell lists are invalid")
    for cell in sampling_cells:
        _run_cell(config_path, str(cell))

    sampling_manifest = report(
        config,
        args.workflow,
        "sampling",
        run_root / "sampling_comparison",
    )
    if not sampling_manifest["gate_passed"]:
        print("PE_6 sampling gate did not pass; lookback stage was not opened")
        return

    winner = str(sampling_manifest["winner"])
    winner_experiment = config.get("experiments", {}).get(winner)
    if not isinstance(winner_experiment, dict):
        raise TemporalWorkflowError("Sampling winner definition disappeared")
    prefix_variant = winner_experiment.get("prefix_variant")
    phase_1_run_name = winner_experiment.get("phase_1_run_name")
    if not isinstance(prefix_variant, str) or not isinstance(phase_1_run_name, str):
        raise TemporalWorkflowError("Sampling winner has no Phase 1 source")

    resolved = copy.deepcopy(config)
    for cell in lookback_cells:
        experiment = resolved.get("experiments", {}).get(str(cell))
        if not isinstance(experiment, dict):
            raise TemporalWorkflowError(f"Unknown lookback cell {cell!r}")
        experiment["prefix_variant"] = prefix_variant
        experiment["phase_1_run_name"] = phase_1_run_name
        experiment["phase_1_mode"] = "reuse"
    resolved_path = run_root / "resolved_lookback_config.json"
    resolved_path.write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for cell in lookback_cells:
        _run_cell(resolved_path, str(cell))
    report(
        resolved,
        args.workflow,
        "lookback",
        run_root / "lookback_comparison",
    )
    print("PE_6 temporal sampling workflow complete")


if __name__ == "__main__":
    main()
