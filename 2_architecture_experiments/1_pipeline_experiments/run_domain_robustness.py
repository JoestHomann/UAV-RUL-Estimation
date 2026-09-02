"""Evaluate PE_9 shift-pruned features with fixed tree configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from experiment_paths import repository_path, run_directory
from experiment_config import read_experiment_config


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
STUDY_DIR = SCRIPT_DIR.parent / "2_model_architecture_study"
for path in (STUDY_DIR / "2_tabular_data_adapter", STUDY_DIR / "4_model_adapters"):
    sys.path.insert(0, str(path))
from model_registry import ModelAdapterFactory  # noqa: E402
from no_op_training_monitor import NoOpTrainingMonitor  # noqa: E402
from tabular_data_adapter import TabularDataAdapter, TabularDataset  # noqa: E402


def _subset(dataset: TabularDataset, names: list[str]) -> TabularDataset:
    return TabularDataset(dataset.features[names].copy(), dataset.metadata, dataset.target, dataset.sample_weights, dataset.fitting_target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_9")
    args = parser.parse_args()
    config = read_experiment_config(args.config.resolve())
    workflow = config["domain_workflows"][args.workflow]
    definition = config["run_definitions"][args.workflow]
    root = run_directory(EXPERIMENTS_DIR, args.workflow, definition)
    feature_sets = json.loads((root / "feature_sets" / "domain_feature_sets.json").read_text(encoding="utf-8"))
    adapter = TabularDataAdapter(repository_path(REPOSITORY_ROOT, workflow["tabular_manifest"]))
    factory = ModelAdapterFactory(repository_path(REPOSITORY_ROOT, workflow["source_specification"]))
    selected = pd.read_csv(repository_path(REPOSITORY_ROOT, workflow["selected_configurations"]))
    predictions = []
    selection_records = []
    for family in workflow.get("families", ["xgboost", "extra_trees"]):
        for outer_fold, selected_rows in selected.loc[selected["model_family"].eq(family)].groupby("outer_fold"):
            selected_row = selected_rows.iloc[0]
            base_names = adapter.feature_names(str(selected_row["feature_set"]))
            for inner_fold in adapter.inner_fold_labels(int(outer_fold)):
                split = adapter.get_inner_selection_split(int(outer_fold), int(inner_fold), str(selected_row["feature_set"]))
                importance_model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=13, n_jobs=1)
                importance_model.fit(split.training.features, split.training.target)
                importance = dict(zip(base_names, importance_model.feature_importances_, strict=True))
                candidates = [name for name in feature_sets["target_aware_candidates"] if name in importance]
                low_cutoff = float(np.median([importance[name] for name in candidates])) if candidates else -1.0
                target_aware_remove = [name for name in candidates if importance[name] <= low_cutoff]
                removals = {"control": [], "shift_pruned_5": feature_sets["shift_pruned_5"], "shift_pruned_10": feature_sets["shift_pruned_10"], "target_aware_pruned": target_aware_remove}
                for cell, removed in removals.items():
                    names = [name for name in base_names if name not in set(removed)]
                    if not names:
                        raise ValueError(f"PE_9 {cell} removed every feature")
                    training = _subset(split.training, names)
                    validation = _subset(split.validation, names)
                    model = factory.create(family, json.loads(selected_row["hyperparameters_json"]), seed=int(selected_row["model_seed"]), training_monitor=NoOpTrainingMonitor())
                    model.fit(training, validation)
                    predicted = model.predict(validation)
                    selection_records.append({"cell": cell, "model_family": family, "outer_fold": int(outer_fold), "inner_fold": int(inner_fold), "feature_count": len(names), "removed_features": json.dumps(sorted(set(base_names) - set(names)))})
                    for index, metadata in validation.metadata.reset_index(drop=True).iterrows():
                        predictions.append({"cell": cell, "model_family": family, "outer_fold": int(outer_fold), "inner_fold": int(inner_fold), "uav_id": metadata["uav_id"], "scenario": metadata["scenario"], "cutoff": metadata["cutoff"], "observed_rul": float(validation.target.iloc[index]), "predicted_rul": float(predicted[index])})
    output = root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(output / "domain_oof_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(selection_records).to_csv(output / "fold_feature_selections.csv", index=False)
    print(f"PE_9 predictions saved to {output}")


if __name__ == "__main__":
    main()
