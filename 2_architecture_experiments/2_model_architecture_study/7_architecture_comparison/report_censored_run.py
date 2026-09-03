"""Apply development gates to censored and horizon architecture Run 9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

import matplotlib.pyplot as plt
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_DIR = REPOSITORY_ROOT / "2_architecture_experiments" / "1_pipeline_experiments"
import sys
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
from oof_experiment_utils import regression_metrics, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    with args.settings.open("rb") as stream:
        settings = tomllib.load(stream)
    predictions = pd.read_csv(
        args.run_root / "5_inner_model_selection" / "selected_inner_predictions.csv.gz"
    )
    records = []
    for (family, fold), rows in predictions.groupby(["model_family", "outer_fold"]):
        records.append({"model_family": family, "outer_fold": int(fold), **regression_metrics(rows["observed_rul"], rows["predicted_rul"])})
    folds = pd.DataFrame(records)
    summary = folds.groupby("model_family", as_index=False).agg(mean_r2=("r2", "mean"), mean_rmse=("rmse", "mean"), mean_rms_overprediction=("rms_overprediction", "mean")).sort_values("mean_rmse")
    control_path = REPOSITORY_ROOT / settings["sources"]["tree_oof_predictions"]
    control = pd.read_csv(control_path)
    control = control.loc[control["method"].eq(settings["sources"]["tree_oof_method"])]
    control_records = []
    for fold, rows in control.groupby("outer_fold"):
        control_records.append({"outer_fold": int(fold), **regression_metrics(rows["observed_rul"], rows["predicted_rul"])})
    control_folds = pd.DataFrame(control_records)
    control_summary = {
        "model_family": "current_tree_blend",
        "mean_r2": float(control_folds["r2"].mean()),
        "mean_rmse": float(control_folds["rmse"].mean()),
        "mean_rms_overprediction": float(control_folds["rms_overprediction"].mean()),
    }
    gates = []
    for family in ("xgboost_aft", "horizon_xgboost"):
        candidate = folds.loc[folds["model_family"].eq(family)]
        paired = candidate.merge(control_folds, on="outer_fold", suffixes=("", "_control"))
        improvement = float((paired["rmse_control"].mean() - paired["rmse"].mean()) / paired["rmse_control"].mean())
        wins = int((paired["rmse"] < paired["rmse_control"]).sum())
        gates.append({"model_family": family, "fold_wins_vs_current_tree": wins, "relative_rmse_improvement_vs_current_tree": improvement, "passes_gate": wins >= int(settings["minimum_fold_wins"]) and improvement >= float(settings["minimum_relative_rmse_improvement"])})
    gate_table = pd.DataFrame(gates)
    promoted = gate_table.loc[gate_table["passes_gate"]]
    winner = "current_tree_blend"
    if not promoted.empty:
        winner = str(summary.loc[summary["model_family"].isin(promoted["model_family"])].iloc[0]["model_family"])
    summary = pd.concat([summary, pd.DataFrame([control_summary])], ignore_index=True).sort_values("mean_rmse")
    output = args.run_root / "7_architecture_comparison"
    figures = args.run_root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "censored_fold_metrics.csv", index=False)
    summary.merge(gate_table, on="model_family", how="left").to_csv(output / "censored_architecture_summary.csv", index=False)
    manifest = {"status": "complete", "winner": winner, "treatment_promoted": winner != "current_tree_blend", "uses_locked_evaluation": False, "gate": {"minimum_fold_wins": int(settings["minimum_fold_wins"]), "minimum_relative_rmse_improvement": float(settings["minimum_relative_rmse_improvement"])}}
    write_json(output / "censored_winner_manifest.json", manifest)
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.bar(summary["model_family"], summary["mean_rmse"], color=["#287271" if name == winner else "#68768a" for name in summary["model_family"]])
    axis.set_ylabel("Mean development RMSE")
    axis.set_title("Run 9 censored and horizon targets")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "censored_architecture_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
