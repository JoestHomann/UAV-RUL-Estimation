"""Report development metrics and tree residual complementarity for Run 7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class TemporalReportError(ValueError):
    """Explain incomplete or unalignable temporal predictions."""


def _fold_metrics(table: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (family, fold), rows in table.groupby(["model_family", "outer_fold"], sort=True):
        y = rows["observed_rul"].to_numpy(float)
        prediction = rows["predicted_rul"].to_numpy(float)
        error = prediction - y
        records.append(
            {
                "model_family": family,
                "outer_fold": int(fold),
                "r2": 1.0 - float(np.square(error).sum()) / float(np.square(y - y.mean()).sum()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    with args.settings.resolve().open("rb") as stream:
        settings = tomllib.load(stream)
    run_root = args.run_root.resolve()
    predictions = pd.read_csv(run_root / "5_inner_model_selection" / "selected_inner_predictions.csv.gz")
    folds = _fold_metrics(predictions)
    summary = folds.groupby("model_family", as_index=False).agg(
        folds=("outer_fold", "nunique"),
        mean_r2=("r2", "mean"),
        mean_rmse=("rmse", "mean"),
        worst_fold_rmse=("rmse", "max"),
        sd_rmse=("rmse", "std"),
    )
    source = REPOSITORY_ROOT / str(settings["sources"]["tree_oof_predictions"])
    tree = pd.read_csv(source)
    tree = tree.loc[tree["method"].astype(str).eq(str(settings["sources"]["tree_oof_method"]))]
    keys = ["outer_fold", "inner_fold", "validation_row", "uav_id", "scenario", "cutoff", "observed_rul"]
    correlations = []
    for family, rows in predictions.groupby("model_family", sort=True):
        paired = rows[keys + ["predicted_rul"]].merge(
            tree[keys + ["predicted_rul"]], on=keys, suffixes=("_temporal", "_tree"), validate="one_to_one"
        )
        if len(paired) != len(rows):
            raise TemporalReportError(f"Tree OOF rows do not align with {family}")
        temporal_residual = paired["predicted_rul_temporal"] - paired["observed_rul"]
        tree_residual = paired["predicted_rul_tree"] - paired["observed_rul"]
        correlations.append({"model_family": family, "tree_residual_correlation": float(np.corrcoef(temporal_residual, tree_residual)[0, 1])})
    summary = summary.merge(pd.DataFrame(correlations), on="model_family", validate="one_to_one")
    summary["passes_accuracy_gate"] = (
        (summary["mean_r2"] >= 0.89)
        & (summary["mean_rmse"] <= 10.7)
        & (summary["tree_residual_correlation"].abs() < 0.90)
    )
    summary = summary.sort_values(["passes_accuracy_gate", "mean_rmse"], ascending=[False, True])
    winner_rows = summary.loc[summary["passes_accuracy_gate"]]
    winner = None if winner_rows.empty else str(winner_rows.iloc[0]["model_family"])
    output = run_root / "7_architecture_comparison"
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "temporal_fold_metrics.csv", index=False)
    summary.to_csv(output / "temporal_architecture_summary.csv", index=False)
    if winner is not None:
        predictions.loc[predictions["model_family"].eq(winner)].to_csv(
            output / "winner_oof_predictions.csv.gz", index=False, compression="gzip"
        )
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(summary["model_family"], summary["mean_rmse"], color="#386cb0")
    axis.axhline(10.7, color="#b33a3a", linestyle="--")
    axis.set_ylabel("Mean development RMSE")
    axis.set_title("Temporal architecture Run 7")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output / "temporal_architecture_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    manifest = {
        "status": "development_candidate" if winner else "no_promotion",
        "winner": winner,
        "accuracy_and_complementarity_gate_passed": winner is not None,
        "seed_stability_pending": winner is not None,
        "locked_evaluation_opened": False,
    }
    (output / "temporal_winner_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
