"""Retrain the Run 7 winner across declared seeds on development splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib

import numpy as np
import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
STUDY_DIR = STEP_DIR.parent
for path in (STUDY_DIR / "3_sequence_data_adapter", STUDY_DIR / "4_model_adapters"):
    sys.path.insert(0, str(path))
from model_registry import ModelAdapterFactory  # noqa: E402
from no_op_training_monitor import NoOpTrainingMonitor  # noqa: E402
from sequence_data_adapter import SequenceDataAdapter  # noqa: E402


def _rmse(rows: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean(np.square(rows["predicted_rul"] - rows["observed_rul"]))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    with args.settings.resolve().open("rb") as stream:
        settings = tomllib.load(stream)
    root = args.run_root.resolve()
    report_dir = root / "7_architecture_comparison"
    manifest_path = report_dir / "temporal_winner_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    winner = manifest.get("winner")
    if not isinstance(winner, str):
        print("Run 7 has no accuracy-gate winner; seed confirmation skipped")
        return
    specification = root / "1_architecture_study_settings" / "artifacts" / "experiment_specification.json"
    sequence_manifest = root / "3_sequence_data_adapter" / "artifacts" / "sequence_dataset_manifest.json"
    selected = pd.read_csv(root / "5_inner_model_selection" / "selected_configurations.csv")
    selected = selected.loc[selected["model_family"].eq(winner)]
    adapter = SequenceDataAdapter(sequence_manifest)
    factory = ModelAdapterFactory(specification)
    records = []
    for outer_fold, selected_rows in selected.groupby("outer_fold", sort=True):
        selected_row = selected_rows.iloc[0]
        hyperparameters = json.loads(selected_row["hyperparameters_json"])
        lookback = int(selected_row["lookback"])
        iterations = selected_row["outer_retraining_iterations"]
        fixed_iterations = None if pd.isna(iterations) else int(iterations)
        for inner_fold in adapter.inner_fold_labels(int(outer_fold)):
            split = adapter.get_inner_selection_split(int(outer_fold), int(inner_fold), lookback)
            for seed in settings["confirmation_seeds"]:
                model = factory.create(
                    winner,
                    hyperparameters,
                    seed=int(seed),
                    training_iterations=fixed_iterations,
                    training_monitor=NoOpTrainingMonitor(),
                )
                model.fit(split.training, split.validation)
                predicted = model.predict(split.validation)
                for index, metadata in split.validation.metadata.reset_index(drop=True).iterrows():
                    records.append({"model_family": winner, "outer_fold": int(outer_fold), "inner_fold": int(inner_fold), "seed": int(seed), "uav_id": metadata["uav_id"], "scenario": metadata["scenario"], "cutoff": metadata["cutoff"], "observed_rul": float(split.validation.target.iloc[index]), "predicted_rul": float(predicted[index])})
    predictions = pd.DataFrame(records)
    predictions.to_csv(report_dir / "seed_confirmation_predictions.csv.gz", index=False, compression="gzip")
    seed_rows = []
    for seed, rows in predictions.groupby("seed", sort=True):
        fold_rmse = rows.groupby("outer_fold").apply(_rmse, include_groups=False)
        seed_rows.append({"seed": int(seed), "mean_rmse": float(fold_rmse.mean()), "sd_fold_rmse": float(fold_rmse.std()), "worst_fold_rmse": float(fold_rmse.max())})
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(report_dir / "temporal_seed_stability.csv", index=False)
    spread = float(seed_summary["mean_rmse"].max() - seed_summary["mean_rmse"].min())
    passed = len(seed_summary) == len(settings["confirmation_seeds"]) and spread <= 0.50 and float(seed_summary["worst_fold_rmse"].max()) <= 15.0
    manifest.update({"status": "promoted" if passed else "no_promotion", "seed_stability_pending": False, "seed_stability_passed": passed, "seed_mean_rmse_spread": spread, "confirmation_seeds": list(settings["confirmation_seeds"])})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
