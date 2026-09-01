"""Summarize PE_8 overall, banded, safety, and plausibility gates."""

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
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]


def _rmse(rows: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean(np.square(rows["predicted_rul"] - rows["observed_rul"]))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_8")
    args = parser.parse_args()
    config = read_experiment_config(args.config.resolve())
    definition = config["run_definitions"][args.workflow]
    root = run_directory(EXPERIMENTS_DIR, args.workflow, definition)
    predictions = pd.read_csv(root / "onset" / "onset_oof_predictions.csv.gz")
    predictions["rul_band"] = pd.cut(predictions["observed_rul"], [-np.inf, 50, 75, 125, np.inf], labels=["0-50", "51-75", "76-125", "126+"])
    fold_records = []
    band_records = []
    for (cell, family, fold), rows in predictions.groupby(["cell", "model_family", "outer_fold"]):
        error = rows["predicted_rul"] - rows["observed_rul"]
        fold_records.append({"cell": cell, "model_family": family, "outer_fold": int(fold), "rmse": _rmse(rows), "rms_overprediction": float(np.sqrt(np.mean(np.square(np.maximum(error, 0.0)))))})
        for band, band_rows in rows.groupby("rul_band", observed=True):
            band_records.append({"cell": cell, "model_family": family, "outer_fold": int(fold), "rul_band": str(band), "rmse": _rmse(band_rows)})
    folds = pd.DataFrame(fold_records)
    bands = pd.DataFrame(band_records)
    controls = folds.loc[folds["cell"].eq("cap125"), ["model_family", "outer_fold", "rmse", "rms_overprediction"]].rename(columns={"rmse": "control_rmse", "rms_overprediction": "control_rms_overprediction"})
    control_bands = bands.loc[bands["cell"].eq("cap125")].rename(columns={"rmse": "control_rmse"})
    summaries = []
    for (cell, family), rows in folds.groupby(["cell", "model_family"]):
        paired = rows.merge(controls, on=["model_family", "outer_fold"], validate="one_to_one")
        treatment_bands = bands.loc[(bands["cell"].eq(cell)) & (bands["model_family"].eq(family))]
        paired_bands = treatment_bands.merge(control_bands, on=["model_family", "outer_fold", "rul_band"], validate="one_to_one")
        changes = paired_bands.groupby("rul_band").apply(lambda group: float((group["rmse"] - group["control_rmse"]).mean()), include_groups=False).to_dict()
        summaries.append({"cell": cell, "model_family": family, "mean_rmse": float(rows["rmse"].mean()), "fold_wins": int((paired["rmse"] < paired["control_rmse"]).sum()), "delta_rmse_0_50": changes.get("0-50", np.nan), "delta_rmse_51_75": changes.get("51-75", np.nan), "delta_rmse_76_125": changes.get("76-125", np.nan), "delta_rms_overprediction": float((paired["rms_overprediction"] - paired["control_rms_overprediction"]).mean())})
    summary = pd.DataFrame(summaries)
    summary["passes_gate"] = (summary["cell"].ne("cap125") & (summary["fold_wins"] >= 4) & (summary["delta_rmse_51_75"] < 0) & (summary["delta_rmse_76_125"] < 0) & (summary["delta_rmse_0_50"] <= 0.25) & (summary["delta_rms_overprediction"] <= 0))
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    folds.to_csv(reporting / "onset_fold_metrics.csv", index=False)
    bands.to_csv(reporting / "onset_band_metrics.csv", index=False)
    summary.to_csv(reporting / "onset_summary.csv", index=False)
    figure, axis = plt.subplots(figsize=(10, 5))
    labels = summary["model_family"] + " / " + summary["cell"]
    axis.bar(labels, summary["mean_rmse"], color=["#27806b" if value else "#386cb0" for value in summary["passes_gate"]])
    axis.set_ylabel("Mean development RMSE")
    axis.tick_params(axis="x", rotation=30)
    axis.set_title("PE_8 personalized degradation onset")
    figure.tight_layout()
    figure_path = reporting / "onset_target_comparison.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    gallery = gallery_directory(EXPERIMENTS_DIR, args.workflow, definition)
    gallery.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure_path, gallery / figure_path.name)
    winners = summary.loc[summary["passes_gate"]].sort_values("mean_rmse")
    manifest = {"status": "promoted" if not winners.empty else "no_promotion", "winner": None if winners.empty else winners.iloc[0][["cell", "model_family"]].to_dict(), "uses_locked_evaluation": False}
    (reporting / "onset_winner_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
