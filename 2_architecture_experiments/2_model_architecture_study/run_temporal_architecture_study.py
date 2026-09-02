"""Materialize and run development-only temporal architecture Run 7."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any


STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
PIPELINE_DIR = STUDY_DIR.parent / "1_pipeline_experiments"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from experiment_config import read_experiment_config  # noqa: E402
from experiment_paths import repository_path  # noqa: E402
from run_experiments import (  # noqa: E402
    _load_interface,
    _paths,
    _phase2_settings,
)


class TemporalArchitectureError(ValueError):
    """Explain incomplete PE_6 inputs or a failed Run 7 step."""


def _run(command: list[str], label: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(label, flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if completed.returncode != 0:
        raise TemporalArchitectureError(f"{label} failed with exit code {completed.returncode}")


def _load_settings(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        settings = tomllib.load(stream)
    families = settings.get("families")
    if not isinstance(families, list) or set(families) != {"tcn", "multiscale_cnn", "gru", "lstm"}:
        raise TemporalArchitectureError(
            "Run 7 families must be exactly tcn, multiscale_cnn, gru, and lstm"
        )
    budget = settings.get("candidate_budget")
    if not isinstance(budget, int) or not 8 <= budget <= 12:
        raise TemporalArchitectureError("candidate_budget must be between 8 and 12")
    return settings


def _winner_experiment(settings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = settings.get("sources", {})
    pe6_settings = repository_path(REPOSITORY_ROOT, str(sources.get("pe6_settings")))
    pe6_root = repository_path(REPOSITORY_ROOT, str(sources.get("pe6_run_root")))
    lookback_manifest_path = pe6_root / "lookback_comparison" / "winner_manifest.json"
    if not lookback_manifest_path.is_file():
        raise TemporalArchitectureError("PE_6 lookback winner is missing; complete PE_6 first")
    manifest = json.loads(lookback_manifest_path.read_text(encoding="utf-8"))
    winner = manifest.get("winner")
    if not manifest.get("gate_passed") or not isinstance(winner, str):
        raise TemporalArchitectureError("PE_6 has no promoted lookback winner")
    resolved_path = pe6_root / "resolved_lookback_config.json"
    config = read_experiment_config(resolved_path if resolved_path.is_file() else pe6_settings)
    experiment = config.get("experiments", {}).get(winner)
    if not isinstance(experiment, dict):
        raise TemporalArchitectureError(f"PE_6 winner {winner!r} is not defined")
    return config, copy.deepcopy(experiment)


def _paths_for_run(run_root: Path) -> dict[str, Path]:
    return {
        "settings": run_root / "resolved_settings.json",
        "specification": run_root / "1_architecture_study_settings" / "artifacts" / "experiment_specification.json",
        "tabular": run_root / "2_tabular_data_adapter" / "artifacts" / "tabular_dataset_manifest.json",
        "sequence": run_root / "3_sequence_data_adapter" / "artifacts" / "sequence_dataset_manifest.json",
        "trajectory": run_root / "3_trajectory_data_adapter" / "artifacts" / "trajectory_dataset_manifest.json",
        "registry": run_root / "4_model_adapters" / "artifacts" / "model_registry.json",
    }


def _materialize(settings: dict[str, Any], run_root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    config, experiment = _winner_experiment(settings)
    # PE_6 fixes the multiscale CNN while isolating sampling density/lookback.
    # Run 7 is an architecture search, so all families must recover their
    # declared search spaces rather than inheriting that sampling-only freeze.
    experiment.pop("fixed_hyperparameters", None)
    experiment.update(
        {
            "architectures": list(settings["families"]),
            "candidate_budget": int(settings["candidate_budget"]),
            "search_seed": int(settings["search_seed"]),
            "retraining_seeds": list(settings["confirmation_seeds"]),
            "phase_2_run_number": int(settings["run_number"]),
            "phase_2_settings_version": int(settings["settings_version"]),
            "target_profile": "capped_125",
            "prediction_profile": "symmetric",
            "phase_2_scope": "selection_only",
            "neural_training": copy.deepcopy(settings["neural_training"]),
        }
    )
    config = copy.deepcopy(config)
    config["execution"] = {"max_workers": int(settings["max_workers"])}
    interface_path, interface = _load_interface(experiment)
    resolved = _phase2_settings(config, experiment, interface, interface_path)
    fixed_families = [
        family
        for family in settings["families"]
        if not any(
            definition.get("kind") != "fixed"
            for definition in resolved["architectures"][family]["search"].values()
        )
    ]
    if fixed_families:
        raise TemporalArchitectureError(
            "Run 7 families unexpectedly have no tunable hyperparameters: "
            + ", ".join(fixed_families)
        )
    paths = _paths_for_run(run_root)
    paths["settings"].parent.mkdir(parents=True, exist_ok=True)
    paths["settings"].write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=STUDY_DIR / "temporal_run_7_settings.toml")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an incompatible or interrupted Run 7 selection.",
    )
    args = parser.parse_args()
    settings = _load_settings(args.settings.resolve())
    run_root = STUDY_DIR / "runs" / f"run_{int(settings['run_number'])}"
    paths = _paths_for_run(run_root)
    if args.status:
        selection = run_root / "5_inner_model_selection" / "selection_manifest.json"
        print(f"Run root: {run_root}")
        print(f"Selection manifest: {'available' if selection.is_file() else 'not generated'}")
        return
    if args.list:
        print("1. Resolve the gate-passing PE_6 density/lookback winner")
        print("2. Build Run 7 settings and tabular/sequence/trajectory manifests")
        print("3. Search tcn, multiscale_cnn, gru, and lstm on development folds")
        print("4. Build temporal architecture report; locked Step 6 is never called")
        return
    paths, config = _materialize(settings, run_root)
    shared = _paths(config)
    _run([sys.executable, str(shared["phase_2_settings_builder"]), "--settings", str(paths["settings"]), "--output-dir", str(paths["specification"].parent)], "Run 7 Step 1 settings")
    _run([sys.executable, str(shared["tabular_adapter_builder"]), "--specification", str(paths["specification"]), "--output-dir", str(paths["tabular"].parent)], "Run 7 Step 2 tabular adapter")
    _run([sys.executable, str(shared["sequence_adapter_builder"]), "--specification", str(paths["specification"]), "--output-dir", str(paths["sequence"].parent)], "Run 7 Step 3 sequence adapter")
    _run([sys.executable, str(shared["trajectory_adapter_builder"]), "--specification", str(paths["specification"]), "--sequence-manifest", str(paths["sequence"]), "--sequence-report", str(paths["sequence"].parent / "copy_verification.json"), "--output-dir", str(paths["trajectory"].parent)], "Run 7 Step 3b trajectory adapter")
    _run([sys.executable, str(shared["model_registry_builder"]), "--specification", str(paths["specification"]), "--output-dir", str(paths["registry"].parent)], "Run 7 Step 4 model registry")
    selection_command = [sys.executable, str(shared["phase_2_orchestrator"]), "--specification", str(paths["specification"]), "--from-step", "5", "--through-step", "5", "--tabular-manifest", str(paths["tabular"]), "--sequence-manifest", str(paths["sequence"]), "--trajectory-manifest", str(paths["trajectory"]), "--model-registry", str(paths["registry"]), "--run-root", str(run_root), "--max-workers", str(settings["max_workers"])]
    if args.force:
        selection_command.append("--force")
    _run(selection_command, "Run 7 development selection")
    _run([sys.executable, str(STUDY_DIR / "7_architecture_comparison" / "report_temporal_run.py"), "--settings", str(args.settings.resolve()), "--run-root", str(run_root)], "Run 7 temporal report")
    _run([sys.executable, str(STUDY_DIR / "7_architecture_comparison" / "confirm_temporal_winner.py"), "--settings", str(args.settings.resolve()), "--run-root", str(run_root)], "Run 7 seed confirmation")


if __name__ == "__main__":
    main()
