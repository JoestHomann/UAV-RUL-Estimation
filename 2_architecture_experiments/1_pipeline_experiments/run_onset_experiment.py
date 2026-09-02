"""Run PE_8 fixed-model onset-target cells on development folds only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from degradation_onset import build_personalized_targets
from experiment_config import read_experiment_config
from experiment_paths import repository_path, run_directory


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
STUDY_DIR = SCRIPT_DIR.parent / "2_model_architecture_study"
for path in (STUDY_DIR / "2_tabular_data_adapter", STUDY_DIR / "4_model_adapters"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_registry import ModelAdapterFactory  # noqa: E402
from no_op_training_monitor import NoOpTrainingMonitor  # noqa: E402
from tabular_data_adapter import TabularDataAdapter, TabularDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_8")
    args = parser.parse_args()
    config = read_experiment_config(args.config.resolve())
    workflow = config.get("onset_workflows", {}).get(args.workflow)
    definition = config.get("run_definitions", {}).get(args.workflow)
    if not isinstance(workflow, dict) or not isinstance(definition, dict):
        raise ValueError(f"Unknown onset workflow {args.workflow!r}")
    specification = repository_path(REPOSITORY_ROOT, workflow["source_specification"])
    manifest = repository_path(REPOSITORY_ROOT, workflow["tabular_manifest"])
    selected = pd.read_csv(repository_path(REPOSITORY_ROOT, workflow["selected_configurations"]))
    adapter = TabularDataAdapter(manifest)
    factory = ModelAdapterFactory(specification)
    output = run_directory(EXPERIMENTS_DIR, args.workflow, definition) / "onset"
    output.mkdir(parents=True, exist_ok=True)
    cells = ["cap125", "temporal_correlation", "monotonic_health_index"]
    prediction_records = []
    onset_records = []
    provenance = []
    for family in workflow.get("families", ["extra_trees", "xgboost"]):
        family_configs = selected.loc[selected["model_family"].eq(family)]
        for outer_fold, selected_row in family_configs.groupby("outer_fold", sort=True):
            configuration = selected_row.iloc[0]
            hyperparameters = json.loads(configuration["hyperparameters_json"])
            feature_set = str(configuration["feature_set"])
            for inner_fold in adapter.inner_fold_labels(int(outer_fold)):
                split = adapter.get_inner_selection_split(int(outer_fold), int(inner_fold), feature_set)
                training_uavs = set(split.training.metadata["uav_id"].astype(str))
                validation_uavs = set(split.validation.metadata["uav_id"].astype(str))
                if training_uavs & validation_uavs:
                    raise ValueError("PE_8 fold contains overlapping UAVs")
                for cell in cells:
                    training = split.training
                    if cell != "cap125":
                        result = build_personalized_targets(
                            training.features,
                            training.metadata,
                            training.target,
                            detector=cell,
                        )
                        training = TabularDataset(
                            features=training.features,
                            metadata=training.metadata,
                            target=training.target,
                            sample_weights=training.sample_weights,
                            fitting_target=result.fitting_target,
                        )
                        for record in result.uav_onsets.to_dict("records"):
                            onset_records.append({"cell": cell, "model_family": family, "outer_fold": int(outer_fold), "inner_fold": int(inner_fold), **record})
                    model = factory.create(
                        family,
                        hyperparameters,
                        seed=int(configuration["model_seed"]),
                        training_monitor=NoOpTrainingMonitor(),
                    )
                    model.fit(training, split.validation)
                    predicted = model.predict(split.validation)
                    for row_index, row in split.validation.metadata.reset_index(drop=True).iterrows():
                        prediction_records.append(
                            {"cell": cell, "model_family": family, "outer_fold": int(outer_fold), "inner_fold": int(inner_fold), "uav_id": row["uav_id"], "scenario": row["scenario"], "cutoff": row["cutoff"], "observed_rul": float(split.validation.target.iloc[row_index]), "predicted_rul": float(predicted[row_index])}
                        )
                    provenance.append({"cell": cell, "model_family": family, "outer_fold": int(outer_fold), "inner_fold": int(inner_fold), "training_uavs": len(training_uavs), "validation_uavs": len(validation_uavs), "uav_overlap": 0})
    pd.DataFrame(prediction_records).to_csv(output / "onset_oof_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(onset_records).to_csv(output / "onset_distributions.csv", index=False)
    pd.DataFrame(provenance).to_csv(output / "fold_provenance.csv", index=False)
    print(f"PE_8 predictions saved to {output}")


if __name__ == "__main__":
    main()
