"""Compare PE_10 cells and freeze the representation supplied to Run 8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_config import read_experiment_config  # noqa: E402
from experiment_paths import run_directory  # noqa: E402


def _selected(root: Path, cell: str) -> pd.DataFrame:
    path = root / cell / "phase2" / "5_inner_model_selection" / "selected_configurations.csv"
    table = pd.read_csv(path)
    rows = table.loc[table["model_family"].eq("hybrid_cnn")].copy()
    if len(rows) != 5 or set(rows["outer_fold"].astype(int)) != set(range(5)):
        raise ValueError(f"{cell} does not contain five completed hybrid CNN folds")
    return rows[["outer_fold", "mean_inner_rmse"]].rename(
        columns={"mean_inner_rmse": f"rmse__{cell}"}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_10")
    args = parser.parse_args()
    config = read_experiment_config(args.config)
    workflow = config.get("hybrid_representation_workflows", {}).get(args.workflow)
    definition = config.get("run_definitions", {}).get(args.workflow)
    if not isinstance(workflow, dict) or not isinstance(definition, dict):
        raise ValueError(f"Unknown hybrid representation workflow {args.workflow!r}")
    root = run_directory(SCRIPT_DIR / "experiments", args.workflow, definition)
    control = str(workflow["control"])
    treatment = str(workflow["treatment"])
    paired = _selected(root, control).merge(
        _selected(root, treatment),
        on="outer_fold",
        validate="one_to_one",
    )
    control_column = f"rmse__{control}"
    treatment_column = f"rmse__{treatment}"
    paired["rmse_improvement"] = paired[control_column] - paired[treatment_column]
    fold_wins = int((paired["rmse_improvement"] > 0.0).sum())
    control_mean = float(paired[control_column].mean())
    treatment_mean = float(paired[treatment_column].mean())
    relative_improvement = (control_mean - treatment_mean) / control_mean
    treatment_promoted = (
        fold_wins >= int(workflow.get("minimum_fold_wins", 4))
        and relative_improvement
        >= float(workflow.get("minimum_relative_rmse_improvement", 0.01))
    )
    winner = treatment if treatment_promoted else control
    representation = "multiresolution" if treatment_promoted else "recent_only"

    reporting = root / "reporting"
    figures = root / "figures"
    reporting.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    paired.to_csv(reporting / "paired_fold_results.csv", index=False)
    summary = pd.DataFrame(
        [
            {"cell": control, "mean_rmse": control_mean, "selected": not treatment_promoted},
            {"cell": treatment, "mean_rmse": treatment_mean, "selected": treatment_promoted},
        ]
    )
    summary.to_csv(reporting / "representation_summary.csv", index=False)
    manifest = {
        "status": "complete",
        "winner": winner,
        "representation": representation,
        "control": control,
        "treatment": treatment,
        "treatment_promoted": treatment_promoted,
        "fold_wins": fold_wins,
        "relative_rmse_improvement": relative_improvement,
        "uses_locked_evaluation": False,
    }
    (reporting / "winner_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#2b6f6d" if selected else "#6f7f8f" for selected in summary["selected"]]
    axis.bar(summary["cell"], summary["mean_rmse"], color=colors)
    axis.set_ylabel("Mean inner-fold RMSE")
    axis.set_title("PE_10 hybrid temporal representation")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "hybrid_representation_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
