"""Apply PE_9 fold-win, RMSE, and high-propensity gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_paths import gallery_directory, run_directory
from experiment_config import read_experiment_config


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"


def _rmse(rows: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean(np.square(rows["predicted_rul"] - rows["observed_rul"]))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_9")
    args = parser.parse_args()
    config = read_experiment_config(args.config.resolve())
    definition = config["run_definitions"][args.workflow]
    root = run_directory(EXPERIMENTS_DIR, args.workflow, definition)
    predictions = pd.read_csv(root / "evaluation" / "domain_oof_predictions.csv.gz")
    propensity = pd.read_csv(root / "domain_diagnostic" / "domain_propensity.csv")
    propensity = propensity.loc[propensity["domain"].eq("development"), ["uav_id", "scenario", "cutoff", "propensity"]]
    predictions = predictions.merge(propensity, on=["uav_id", "scenario", "cutoff"], how="left", validate="many_to_one")
    if predictions["propensity"].isna().any():
        raise ValueError("PE_9 development predictions do not align with propensity")
    threshold = float(propensity["propensity"].quantile(0.8))
    fold_records = []
    for (cell, family, fold), rows in predictions.groupby(["cell", "model_family", "outer_fold"]):
        high = rows.loc[rows["propensity"] >= threshold]
        fold_records.append({"cell": cell, "model_family": family, "outer_fold": int(fold), "rmse": _rmse(rows), "high_propensity_rmse": _rmse(high) if not high.empty else np.nan})
    folds = pd.DataFrame(fold_records)
    control = folds.loc[folds["cell"].eq("control")].rename(columns={"rmse": "control_rmse", "high_propensity_rmse": "control_high_propensity_rmse"})[["model_family", "outer_fold", "control_rmse", "control_high_propensity_rmse"]]
    records = []
    for (cell, family), rows in folds.groupby(["cell", "model_family"]):
        paired = rows.merge(control, on=["model_family", "outer_fold"], validate="one_to_one")
        improvement = float((paired["control_rmse"].mean() - paired["rmse"].mean()) / paired["control_rmse"].mean())
        high_delta = float((paired["high_propensity_rmse"] - paired["control_high_propensity_rmse"]).mean())
        records.append({"cell": cell, "model_family": family, "mean_rmse": float(rows["rmse"].mean()), "fold_wins": int((paired["rmse"] < paired["control_rmse"]).sum()), "relative_rmse_improvement": improvement, "high_propensity_rmse_delta": high_delta})
    summary = pd.DataFrame(records)
    summary["passes_gate"] = summary["cell"].ne("control") & (summary["fold_wins"] >= 4) & (summary["relative_rmse_improvement"] >= 0.02) & (summary["high_propensity_rmse_delta"] <= 0)
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    folds.to_csv(reporting / "domain_fold_metrics.csv", index=False)
    summary.to_csv(reporting / "domain_summary.csv", index=False)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(summary["model_family"] + " / " + summary["cell"], summary["mean_rmse"], color=["#27806b" if value else "#386cb0" for value in summary["passes_gate"]])
    axis.set_ylabel("Mean development RMSE")
    axis.tick_params(axis="x", rotation=30)
    axis.set_title("PE_9 domain-robust feature pruning")
    figure.tight_layout()
    figure_path = reporting / "domain_feature_comparison.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    gallery = gallery_directory(EXPERIMENTS_DIR, args.workflow, definition)
    gallery.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure_path, gallery / figure_path.name)
    winners = summary.loc[summary["passes_gate"]].sort_values("mean_rmse")
    manifest = {"status": "promoted" if not winners.empty else "no_promotion", "winner": None if winners.empty else winners.iloc[0][["cell", "model_family"]].to_dict(), "uses_locked_evaluation": False, "highest_propensity_threshold": threshold}
    (reporting / "domain_winner_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
