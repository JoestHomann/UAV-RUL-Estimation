"""Freeze deterministic PE_9 shift-pruning candidates from diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiment_paths import run_directory
from experiment_config import read_experiment_config


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_9")
    args = parser.parse_args()
    config = read_experiment_config(args.config.resolve())
    definition = config["run_definitions"][args.workflow]
    workflow = config["domain_workflows"][args.workflow]
    if workflow.get("competition_rules_allow_unlabelled_test_adaptation") is not True:
        raise ValueError(
            "PE_9 feature construction is disabled until competition rules "
            "permit unlabelled test-distribution adaptation"
        )
    root = run_directory(EXPERIMENTS_DIR, args.workflow, definition)
    shifts = pd.read_csv(root / "domain_diagnostic" / "feature_shift_statistics.csv")
    ranking = shifts["feature"].astype(str).tolist()
    manifest = {
        "control": [],
        "shift_pruned_5": ranking[:5],
        "shift_pruned_10": ranking[:10],
        "target_aware_candidates": ranking[:15],
        "selection_rule": "descending KS plus 0.25 absolute standardized mean difference; lexical tie break",
    }
    output = root / "feature_sets"
    output.mkdir(parents=True, exist_ok=True)
    (output / "domain_feature_sets.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
