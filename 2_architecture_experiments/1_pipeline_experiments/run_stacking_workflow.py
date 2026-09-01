"""Run PE_7 alignment, nested OOF stacking, and promotion gating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from experiment_config import read_experiment_config
from experiment_paths import run_directory


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]


def _run(command: list[str], label: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(label, flush=True)
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {completed.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_7")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_experiment_config(config_path)
    definition = config.get("run_definitions", {}).get(args.workflow)
    if not isinstance(definition, dict):
        raise ValueError(f"Unknown run definition {args.workflow!r}")
    root = run_directory(EXPERIMENTS_DIR, args.workflow, definition)
    workflow = config.get("stacking_workflows", {}).get(args.workflow)
    temporal_manifest_value = (
        workflow.get("temporal_winner_manifest")
        if isinstance(workflow, dict)
        else None
    )
    if not isinstance(temporal_manifest_value, str):
        raise ValueError("PE_7 has no temporal_winner_manifest setting")
    temporal_manifest = json.loads(
        (REPOSITORY_ROOT / temporal_manifest_value).read_text(encoding="utf-8")
    )
    if temporal_manifest.get("seed_stability_passed") is not True:
        raise ValueError(
            "PE_7 refuses to run until temporal Run 7 passes seed confirmation"
        )
    aligned = root / "aligned_oof_predictions"
    stacking = root / "stacking"
    reporting = root / "reporting"
    _run(
        [sys.executable, str(SCRIPT_DIR / "align_oof_predictions.py"), "--config", str(config_path), "--workflow", args.workflow, "--output-dir", str(aligned)],
        "PE_7 exact OOF alignment",
    )
    _run(
        [sys.executable, str(SCRIPT_DIR / "stack_oof_predictions.py"), "--config", str(config_path), "--workflow", args.workflow, "--aligned", str(aligned / "aligned_oof_predictions.csv.gz"), "--output-dir", str(stacking)],
        "PE_7 nested OOF stacking",
    )
    _run(
        [sys.executable, str(SCRIPT_DIR / "promote_stacked_model.py"), "--config", str(config_path), "--workflow", args.workflow, "--stack-dir", str(stacking), "--output", str(reporting / "promotion_contract.json")],
        "PE_7 promotion gate",
    )


if __name__ == "__main__":
    main()
