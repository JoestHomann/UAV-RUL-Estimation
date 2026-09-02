"""Materialize and run development-only hybrid architecture Run 8."""

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
from run_experiments import _load_interface, _paths, _phase2_settings  # noqa: E402


class HybridArchitectureError(ValueError):
    """Explain incomplete PE_10 inputs or a failed Run 8 step."""


def _run(command: list[str], label: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(label, flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if completed.returncode != 0:
        raise HybridArchitectureError(f"{label} failed with exit code {completed.returncode}")


def _load_settings(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        settings = tomllib.load(stream)
    expected = {"xgboost", "hybrid_cnn", "hybrid_gru"}
    if set(settings.get("families", [])) != expected:
        raise HybridArchitectureError(f"Run 8 families must be exactly {sorted(expected)}")
    budget = settings.get("candidate_budget")
    if not isinstance(budget, int) or not 5 <= budget <= 15:
        raise HybridArchitectureError("candidate_budget must be between 5 and 15")
    return settings


def _paths_for_run(run_root: Path) -> dict[str, Path]:
    return {
        "settings": run_root / "resolved_settings.json",
        "specification": run_root / "1_architecture_study_settings" / "artifacts" / "experiment_specification.json",
        "tabular": run_root / "2_tabular_data_adapter" / "artifacts" / "tabular_dataset_manifest.json",
        "sequence": run_root / "3_sequence_data_adapter" / "artifacts" / "sequence_dataset_manifest.json",
        "trajectory": run_root / "3_trajectory_data_adapter" / "artifacts" / "trajectory_dataset_manifest.json",
        "registry": run_root / "4_model_adapters" / "artifacts" / "model_registry.json",
    }


def _pe10_winner(settings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    sources = settings.get("sources", {})
    pe10_settings = repository_path(REPOSITORY_ROOT, str(sources.get("pe10_settings")))
    pe10_root = repository_path(REPOSITORY_ROOT, str(sources.get("pe10_run_root")))
    manifest_path = pe10_root / "reporting" / "winner_manifest.json"
    if not manifest_path.is_file():
        raise HybridArchitectureError("PE_10 winner is missing; complete PE_10 first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    winner = manifest.get("winner")
    representation = manifest.get("representation")
    if manifest.get("status") != "complete" or not isinstance(winner, str):
        raise HybridArchitectureError("PE_10 has no frozen representation winner")
    if representation not in {"recent_only", "multiresolution"}:
        raise HybridArchitectureError("PE_10 winner has an invalid representation")
    config = read_experiment_config(pe10_settings)
    experiment = config.get("experiments", {}).get(winner)
    if not isinstance(experiment, dict):
        raise HybridArchitectureError(f"PE_10 winner {winner!r} is not defined")
    return config, copy.deepcopy(experiment), str(representation)


def _materialize(settings: dict[str, Any], run_root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    config, experiment, representation = _pe10_winner(settings)
    source_lookback = 20 if representation == "recent_only" else 100
    experiment.update(
        {
            "architectures": list(settings["families"]),
            "candidate_budget": int(settings["candidate_budget"]),
            "search_seed": int(settings["search_seed"]),
            "retraining_seeds": [13],
            "phase_2_run_number": int(settings["run_number"]),
            "phase_2_settings_version": int(settings["settings_version"]),
            "sequence_lookbacks": [source_lookback],
            "target_profile": "capped_125",
            "prediction_profile": "conditional_q55",
            "phase_2_scope": "selection_only",
            "neural_training": copy.deepcopy(settings["neural_training"]),
            "fixed_hyperparameters": {
                "hybrid_cnn": {
                    "history_mode": representation,
                    "recent_lookback": 20,
                    "history_bins": 20,
                },
                "hybrid_gru": {
                    "history_mode": representation,
                    "recent_lookback": 20,
                    "history_bins": 20,
                },
            },
        }
    )
    config = copy.deepcopy(config)
    config["execution"] = {"max_workers": int(settings["max_workers"])}
    interface_path, interface = _load_interface(experiment)
    resolved = _phase2_settings(config, experiment, interface, interface_path)
    paths = _paths_for_run(run_root)
    paths["settings"].parent.mkdir(parents=True, exist_ok=True)
    paths["settings"].write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        type=Path,
        default=STUDY_DIR / "hybrid_run_8_settings.toml",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    settings_path = args.settings.resolve()
    settings = _load_settings(settings_path)
    run_root = STUDY_DIR / "runs" / f"run_{int(settings['run_number'])}"
    paths = _paths_for_run(run_root)
    if args.status:
        selection = run_root / "5_inner_model_selection" / "selection_manifest.json"
        report = run_root / "7_architecture_comparison" / "hybrid_winner_manifest.json"
        print(f"Run root: {run_root}")
        print(f"Selection: {'complete' if selection.is_file() else 'pending'}")
        print(f"Report: {'complete' if report.is_file() else 'pending'}")
        return
    if args.list:
        print("1. Read the frozen PE_10 representation winner")
        print("2. Build aligned tabular and sequence adapter artifacts")
        print("3. Search XGBoost, hybrid CNN, and hybrid GRU on development folds")
        print("4. Apply paired accuracy and safety gates; locked Step 6 is never called")
        return

    paths, config = _materialize(settings, run_root)
    shared = _paths(config)
    _run([sys.executable, str(shared["phase_2_settings_builder"]), "--settings", str(paths["settings"]), "--output-dir", str(paths["specification"].parent)], "Run 8 Step 1 settings")
    _run([sys.executable, str(shared["tabular_adapter_builder"]), "--specification", str(paths["specification"]), "--output-dir", str(paths["tabular"].parent)], "Run 8 Step 2 tabular adapter")
    _run([sys.executable, str(shared["sequence_adapter_builder"]), "--specification", str(paths["specification"]), "--output-dir", str(paths["sequence"].parent)], "Run 8 Step 3 sequence adapter")
    _run([sys.executable, str(shared["trajectory_adapter_builder"]), "--specification", str(paths["specification"]), "--sequence-manifest", str(paths["sequence"]), "--sequence-report", str(paths["sequence"].parent / "copy_verification.json"), "--output-dir", str(paths["trajectory"].parent)], "Run 8 Step 3b trajectory adapter")
    _run([sys.executable, str(shared["model_registry_builder"]), "--specification", str(paths["specification"]), "--output-dir", str(paths["registry"].parent)], "Run 8 Step 4 model registry")
    command = [sys.executable, str(shared["phase_2_orchestrator"]), "--specification", str(paths["specification"]), "--from-step", "5", "--through-step", "5", "--tabular-manifest", str(paths["tabular"]), "--sequence-manifest", str(paths["sequence"]), "--trajectory-manifest", str(paths["trajectory"]), "--model-registry", str(paths["registry"]), "--run-root", str(run_root), "--max-workers", str(settings["max_workers"])]
    if args.force:
        command.append("--force")
    _run(command, "Run 8 development selection")
    _run([sys.executable, str(STUDY_DIR / "7_architecture_comparison" / "report_hybrid_run.py"), "--settings", str(settings_path), "--run-root", str(run_root)], "Run 8 hybrid report")


if __name__ == "__main__":
    main()
