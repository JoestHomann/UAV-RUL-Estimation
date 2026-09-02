"""Apply paired development gates to hybrid architecture Run 8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ALIGNMENT_KEYS = ["outer_fold", "inner_fold", "uav_id", "scenario", "cutoff"]


def _metrics(rows: pd.DataFrame) -> dict[str, float]:
    residual = rows["predicted_rul"].to_numpy(float) - rows["observed_rul"].to_numpy(float)
    observed = rows["observed_rul"].to_numpy(float)
    predictions = rows["predicted_rul"].to_numpy(float)
    denominator = float(np.square(observed - observed.mean()).sum())
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "r2": float(1.0 - np.square(residual).sum() / denominator),
        "rms_overprediction": float(np.sqrt(np.mean(np.square(np.maximum(residual, 0.0))))),
    }


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
    expected = set(settings["families"])
    if set(predictions["model_family"]) != expected:
        raise ValueError("Run 8 selected predictions do not contain every family")
    source = settings["sources"]
    tree_path = (REPOSITORY_ROOT / source["tree_oof_predictions"]).resolve()
    tree_predictions = pd.read_csv(tree_path)
    tree_method = str(source["tree_oof_method"])
    tree_predictions = tree_predictions.loc[
        tree_predictions["method"].eq(tree_method),
        [*ALIGNMENT_KEYS, "observed_rul", "predicted_rul"],
    ].copy()
    if tree_predictions.empty:
        raise ValueError(f"Current tree OOF method {tree_method!r} is missing")
    records = []
    for (family, outer_fold), rows in predictions.groupby(["model_family", "outer_fold"]):
        records.append({"model_family": family, "outer_fold": int(outer_fold), **_metrics(rows)})
    folds = pd.DataFrame(records)
    current_tree_records = []
    for outer_fold, rows in tree_predictions.groupby("outer_fold"):
        current_tree_records.append(
            {
                "model_family": "current_tree_blend",
                "outer_fold": int(outer_fold),
                **_metrics(rows),
            }
        )
    current_tree_folds = pd.DataFrame(current_tree_records)
    control = current_tree_folds.rename(
        columns={
            "rmse": "control_rmse",
            "r2": "control_r2",
            "rms_overprediction": "control_rms_overprediction",
        }
    )
    summaries = []
    for family, rows in folds.groupby("model_family"):
        family_predictions = predictions.loc[
            predictions["model_family"].eq(family),
            [*ALIGNMENT_KEYS, "observed_rul"],
        ]
        aligned = family_predictions.merge(
            tree_predictions[[*ALIGNMENT_KEYS, "observed_rul"]],
            on=ALIGNMENT_KEYS,
            suffixes=("__hybrid", "__tree"),
            validate="one_to_one",
        )
        if len(aligned) != len(family_predictions) or not np.allclose(
            aligned["observed_rul__hybrid"],
            aligned["observed_rul__tree"],
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(
                f"{family} predictions do not align with the current tree OOF baseline"
            )
        paired = rows.merge(
            control[["outer_fold", "control_rmse", "control_r2", "control_rms_overprediction"]],
            on="outer_fold",
            validate="one_to_one",
        )
        relative = float(
            (paired["control_rmse"].mean() - paired["rmse"].mean())
            / paired["control_rmse"].mean()
        )
        fold_wins = int((paired["rmse"] < paired["control_rmse"]).sum())
        safety_delta = float(
            (paired["rms_overprediction"] - paired["control_rms_overprediction"]).mean()
        )
        passes = (
            family.startswith("hybrid_")
            and fold_wins >= int(settings["minimum_fold_wins"])
            and relative >= float(settings["minimum_relative_rmse_improvement"])
            and safety_delta <= float(settings["maximum_rms_overprediction_increase"])
        )
        summaries.append(
            {
                "model_family": family,
                "mean_rmse": float(rows["rmse"].mean()),
                "mean_r2": float(rows["r2"].mean()),
                "mean_rms_overprediction": float(rows["rms_overprediction"].mean()),
                "fold_wins_vs_current_tree": fold_wins,
                "relative_rmse_improvement_vs_current_tree": relative,
                "rms_overprediction_delta_vs_current_tree": safety_delta,
                "passes_gate": passes,
            }
        )
    summaries.append(
        {
            "model_family": "current_tree_blend",
            "mean_rmse": float(current_tree_folds["rmse"].mean()),
            "mean_r2": float(current_tree_folds["r2"].mean()),
            "mean_rms_overprediction": float(
                current_tree_folds["rms_overprediction"].mean()
            ),
            "fold_wins_vs_current_tree": 0,
            "relative_rmse_improvement_vs_current_tree": 0.0,
            "rms_overprediction_delta_vs_current_tree": 0.0,
            "passes_gate": False,
        }
    )
    summary = pd.DataFrame(summaries).sort_values("mean_rmse")
    promoted = summary.loc[summary["passes_gate"]]
    winner = (
        "current_tree_blend"
        if promoted.empty
        else str(promoted.iloc[0]["model_family"])
    )
    manifest = {
        "status": "complete",
        "winner": winner,
        "hybrid_promoted": not promoted.empty,
        "uses_locked_evaluation": False,
        "control": {
            "method": tree_method,
            "predictions": str(tree_path.relative_to(REPOSITORY_ROOT)),
        },
        "gate": {
            "minimum_fold_wins": int(settings["minimum_fold_wins"]),
            "minimum_relative_rmse_improvement": float(settings["minimum_relative_rmse_improvement"]),
            "maximum_rms_overprediction_increase": float(settings["maximum_rms_overprediction_increase"]),
        },
    }
    output = args.run_root / "7_architecture_comparison"
    figures = args.run_root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "hybrid_fold_metrics.csv", index=False)
    summary.to_csv(output / "hybrid_architecture_summary.csv", index=False)
    (output / "hybrid_winner_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    colors = ["#287271" if family == winner else "#718096" for family in summary["model_family"]]
    axis.bar(summary["model_family"], summary["mean_rmse"], color=colors)
    axis.set_ylabel("Mean grouped development RMSE")
    axis.set_title("Run 8 hybrid architecture comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "hybrid_architecture_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
