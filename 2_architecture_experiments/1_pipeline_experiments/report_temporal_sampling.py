"""Compare PE_6 temporal sampling cells and freeze one development winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_config import read_experiment_config  # noqa: E402
from experiment_paths import artifact_directory, gallery_directory  # noqa: E402


EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]


class TemporalSamplingError(ValueError):
    """Explain incomplete or incompatible temporal sampling artifacts."""


def _metrics(experiment: str, table: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for outer_fold, rows in table.groupby("outer_fold", sort=True):
        observed = rows["observed_rul"].to_numpy(dtype=np.float64)
        predicted = rows["predicted_rul"].to_numpy(dtype=np.float64)
        residual = predicted - observed
        denominator = float(np.sum(np.square(observed - np.mean(observed))))
        records.append(
            {
                "experiment": experiment,
                "outer_fold": int(outer_fold),
                "rows": int(len(rows)),
                "r2": 1.0 - float(np.sum(np.square(residual))) / denominator,
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "mae": float(np.mean(np.abs(residual))),
                "bias": float(np.mean(residual)),
            }
        )
    return pd.DataFrame.from_records(records)


def _load_cell(
    config: dict[str, Any],
    name: str,
    family: str,
) -> pd.DataFrame:
    experiment = config.get("experiments", {}).get(name)
    if not isinstance(experiment, dict):
        raise TemporalSamplingError(f"Unknown temporal sampling cell {name!r}")
    path = (
        artifact_directory(EXPERIMENTS_DIR, name, experiment)
        / "phase2"
        / "5_inner_model_selection"
        / "selected_inner_predictions.csv.gz"
    )
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise TemporalSamplingError(f"Cannot read {name} predictions: {error}") from error
    required = {
        "model_family",
        "outer_fold",
        "uav_id",
        "scenario",
        "cutoff",
        "observed_rul",
        "predicted_rul",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise TemporalSamplingError(f"{name} predictions are missing {missing}")
    table = table.loc[table["model_family"].astype(str).eq(family)].copy()
    if table.empty or table["outer_fold"].nunique() != 5:
        raise TemporalSamplingError(f"{name} has incomplete {family} predictions")
    return table


def _summary(folds: pd.DataFrame) -> pd.DataFrame:
    return (
        folds.groupby("experiment", as_index=False)
        .agg(
            folds=("outer_fold", "nunique"),
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            mean_bias=("bias", "mean"),
        )
        .sort_values(["mean_rmse", "mean_r2"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _plot(summary: pd.DataFrame, path: Path, title: str) -> None:
    ordered = summary.sort_values("mean_rmse", ascending=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(ordered["experiment"], ordered["mean_rmse"], color="#2878b5")
    axes[0].set_ylabel("Mean development RMSE")
    axes[1].bar(ordered["experiment"], ordered["mean_r2"], color="#d98b2b")
    axes[1].set_ylabel("Mean development R2")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def report(
    config: dict[str, Any],
    workflow_name: str,
    stage: str,
    output_dir: Path,
) -> dict[str, Any]:
    workflows = config.get("temporal_sampling_workflows", {})
    workflow = workflows.get(workflow_name) if isinstance(workflows, dict) else None
    if not isinstance(workflow, dict):
        raise TemporalSamplingError(f"Unknown workflow {workflow_name!r}")
    family = str(workflow.get("family", "multiscale_cnn"))
    if stage == "sampling":
        cells = workflow.get("sampling_cells")
        control = str(workflow.get("sampling_control"))
    elif stage == "lookback":
        cells = workflow.get("lookback_cells")
        control = None
    else:
        raise TemporalSamplingError("stage must be sampling or lookback")
    if not isinstance(cells, list) or not cells or not all(isinstance(x, str) for x in cells):
        raise TemporalSamplingError(f"{stage}_cells must be a non-empty list")

    folds = pd.concat(
        [_metrics(name, _load_cell(config, name, family)) for name in cells],
        ignore_index=True,
    )
    summary = _summary(folds)
    winner = str(summary.iloc[0]["experiment"])
    gate_passed = True
    gate: dict[str, Any] = {}
    if stage == "sampling":
        if control not in cells:
            raise TemporalSamplingError("sampling_control is absent from sampling_cells")
        dense = summary.loc[~summary["experiment"].eq(control)].copy()
        if dense.empty:
            raise TemporalSamplingError("Sampling stage has no dense treatment")
        dense = dense.sort_values(
            ["mean_rmse", "experiment"],
            ascending=[True, False],
        )
        winner = str(dense.iloc[0]["experiment"])
        control_folds = folds.loc[folds["experiment"].eq(control)].set_index("outer_fold")
        winner_folds = folds.loc[folds["experiment"].eq(winner)].set_index("outer_fold")
        aligned = control_folds[["rmse"]].join(
            winner_folds[["rmse"]],
            lsuffix="_control",
            rsuffix="_winner",
            how="inner",
        )
        control_rmse = float(control_folds["rmse"].mean())
        winner_rmse = float(winner_folds["rmse"].mean())
        fold_wins = int((aligned["rmse_winner"] < aligned["rmse_control"]).sum())
        improvement = (control_rmse - winner_rmse) / control_rmse
        worst_delta = float(
            (aligned["rmse_winner"] - aligned["rmse_control"]).max()
        )
        gate = {
            "fold_wins": fold_wins,
            "relative_rmse_improvement": improvement,
            "worst_fold_rmse_delta": worst_delta,
        }
        gate_passed = fold_wins >= 4 and improvement >= 0.03 and worst_delta <= 1.0

    output_dir.mkdir(parents=True, exist_ok=True)
    folds_path = output_dir / f"{stage}_fold_results.csv"
    summary_path = output_dir / f"{stage}_summary.csv"
    figure_path = output_dir / f"temporal_{stage}_comparison.png"
    folds.to_csv(folds_path, index=False)
    summary.to_csv(summary_path, index=False)
    _plot(summary, figure_path, f"PE_6 temporal {stage} comparison")

    owner = config.get("run_definitions", {}).get(workflow_name)
    if isinstance(owner, dict):
        gallery = gallery_directory(EXPERIMENTS_DIR, workflow_name, owner)
        gallery.mkdir(parents=True, exist_ok=True)
        shutil.copy2(figure_path, gallery / figure_path.name)

    manifest = {
        "status": "complete" if gate_passed else "no_promotion",
        "workflow": workflow_name,
        "stage": stage,
        "family": family,
        "winner": winner,
        "gate_passed": gate_passed,
        "gate": gate,
        "uses_locked_evaluation": False,
        "artifacts": {
            "fold_results": folds_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "summary": summary_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "figure": figure_path.relative_to(REPOSITORY_ROOT).as_posix(),
        },
    }
    (output_dir / "winner_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_6")
    parser.add_argument("--stage", choices=("sampling", "lookback"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = report(
        read_experiment_config(args.config.resolve()),
        args.workflow,
        args.stage,
        args.output_dir.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
