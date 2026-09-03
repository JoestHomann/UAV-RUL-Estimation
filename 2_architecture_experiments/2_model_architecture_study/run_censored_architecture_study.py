"""Materialize and run development-only censored-target architecture Run 9."""

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


class CensoredArchitectureError(ValueError):
    """Explain an invalid Run 9 setting or failed execution step."""


def _run(command: list[str], label: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(label, flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if completed.returncode != 0:
        raise CensoredArchitectureError(
            f"{label} failed with exit code {completed.returncode}"
        )


def _load_settings(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        settings = tomllib.load(stream)
    expected = {"xgboost", "xgboost_aft", "horizon_xgboost"}
    if set(settings.get("families", [])) != expected:
        raise CensoredArchitectureError(
            f"Run 9 families must be exactly {sorted(expected)}"
        )
    budget = settings.get("candidate_budget")
    if isinstance(budget, bool) or not isinstance(budget, int) or not 3 <= budget <= 15:
        raise CensoredArchitectureError("candidate_budget must be between 3 and 15")
    return settings


def _run_paths(run_root: Path) -> dict[str, Path]:
    return {
        "settings": run_root / "resolved_settings.json",
        "specification": run_root / "1_architecture_study_settings" / "artifacts" / "experiment_specification.json",
        "tabular": run_root / "2_tabular_data_adapter" / "artifacts" / "tabular_dataset_manifest.json",
        "sequence": run_root / "3_sequence_data_adapter" / "artifacts" / "sequence_dataset_manifest.json",
        "trajectory": run_root / "3_trajectory_data_adapter" / "artifacts" / "trajectory_dataset_manifest.json",
        "registry": run_root / "4_model_adapters" / "artifacts" / "model_registry.json",
    }


def _materialize(settings: dict[str, Any], run_root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    source_path = repository_path(REPOSITORY_ROOT, settings["sources"]["pe3_settings"])
    config = read_experiment_config(source_path)
    source_name = str(settings["sources"].get("source_experiment", "PE3_features_drift"))
    source = config.get("experiments", {}).get(source_name)
    if not isinstance(source, dict):
        raise CensoredArchitectureError(f"PE_3 source {source_name!r} is missing")
    experiment = copy.deepcopy(source)
    experiment.update(
        {
            "architectures": list(settings["families"]),
            "candidate_budget": int(settings["candidate_budget"]),
            "search_seed": int(settings["search_seed"]),
            "retraining_seeds": [13],
            "phase_2_run_number": int(settings["run_number"]),
            "phase_2_settings_version": int(settings["settings_version"]),
            "target_profile": "raw",
            "prediction_profile": "symmetric",
            "phase_2_scope": "selection_only",
        }
    )
    config = copy.deepcopy(config)
    config["execution"] = {"max_workers": int(settings["max_workers"])}
    interface_path, interface = _load_interface(experiment)
    resolved = _phase2_settings(config, experiment, interface, interface_path)
    paths = _run_paths(run_root)
    paths["settings"].parent.mkdir(parents=True, exist_ok=True)
    paths["settings"].write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings", type=Path, default=STUDY_DIR / "censored_run_9_settings.toml"
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    settings_path = args.settings.resolve()
    settings = _load_settings(settings_path)
    run_root = STUDY_DIR / "runs" / f"run_{int(settings['run_number'])}"
    paths = _run_paths(run_root)
    if args.list:
        print("1. Materialize the frozen PE_3 data contract with raw evaluation targets")
        print("2. Search scalar XGBoost, right-censored XGBoost AFT, and horizon XGBoost")
        print("3. Compare development OOF predictions with the current calibrated tree blend")
        print("4. Stop before locked Step 6")
        return
    if args.status:
        print(f"Run root: {run_root}")
        print(f"Selection: {'complete' if (run_root / '5_inner_model_selection' / 'selection_manifest.json').is_file() else 'pending'}")
        print(f"Report: {'complete' if (run_root / '7_architecture_comparison' / 'censored_winner_manifest.json').is_file() else 'pending'}")
        return
    paths, config = _materialize(settings, run_root)
    shared = _paths(config)
    _run([sys.executable, str(shared["phase_2_settings_builder"]), "--settings", str(paths["settings"]), "--output-dir", str(paths["specification"].parent)], "Run 9 Step 1 settings")
    _run([sys.executable, str(shared["tabular_adapter_builder"]), "--specification", str(paths["specification"]), "--output-dir", str(paths["tabular"].parent)], "Run 9 Step 2 tabular adapter")
    _run([sys.executable, str(shared["sequence_adapter_builder"]), "--specification", str(paths["specification"]), "--output-dir", str(paths["sequence"].parent)], "Run 9 Step 3 sequence adapter")
    _run([sys.executable, str(shared["trajectory_adapter_builder"]), "--specification", str(paths["specification"]), "--sequence-manifest", str(paths["sequence"]), "--sequence-report", str(paths["sequence"].parent / "copy_verification.json"), "--output-dir", str(paths["trajectory"].parent)], "Run 9 Step 3b trajectory adapter")
    _run([sys.executable, str(shared["model_registry_builder"]), "--specification", str(paths["specification"]), "--output-dir", str(paths["registry"].parent)], "Run 9 Step 4 model registry")
    command = [sys.executable, str(shared["phase_2_orchestrator"]), "--specification", str(paths["specification"]), "--from-step", "5", "--through-step", "5", "--tabular-manifest", str(paths["tabular"]), "--sequence-manifest", str(paths["sequence"]), "--trajectory-manifest", str(paths["trajectory"]), "--model-registry", str(paths["registry"]), "--run-root", str(run_root), "--max-workers", str(settings["max_workers"])]
    if args.force:
        command.append("--force")
    _run(command, "Run 9 development selection")
    _run([sys.executable, str(STUDY_DIR / "7_architecture_comparison" / "report_censored_run.py"), "--settings", str(settings_path), "--run-root", str(run_root)], "Run 9 censored-target report")


if __name__ == "__main__":
    main()

