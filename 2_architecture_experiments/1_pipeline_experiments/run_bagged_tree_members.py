"""Generate seed-resolved, leakage-safe development predictions for PE_11."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from experiment_config import read_experiment_config
from experiment_paths import repository_path, run_directory
from oof_experiment_utils import write_json


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
STUDY_DIR = SCRIPT_DIR.parent / "2_model_architecture_study"
for module_dir in (STUDY_DIR / "2_tabular_data_adapter", STUDY_DIR / "4_model_adapters"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from model_registry import ModelAdapterFactory  # noqa: E402
from no_op_training_monitor import NoOpTrainingMonitor  # noqa: E402
from tabular_data_adapter import TabularDataAdapter  # noqa: E402


def _workflow(config: dict, name: str) -> tuple[dict, dict]:
    workflow = config.get("bagging_residual_workflows", {}).get(name)
    definition = config.get("run_definitions", {}).get(name)
    if not isinstance(workflow, dict) or not isinstance(definition, dict):
        raise ValueError(f"Unknown bagging/residual workflow {name!r}")
    return workflow, definition


def _write_csv_atomic(table: pd.DataFrame, path: Path, *, gzip: bool = False) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    table.to_csv(
        temporary,
        index=False,
        compression="gzip" if gzip else None,
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_11")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = read_experiment_config(args.config.resolve())
    workflow, definition = _workflow(config, args.workflow)
    specification = repository_path(REPOSITORY_ROOT, workflow["source_specification"])
    manifest = repository_path(REPOSITORY_ROOT, workflow["tabular_manifest"])
    selected_path = repository_path(
        REPOSITORY_ROOT, workflow["selected_configurations"]
    )
    adapter = TabularDataAdapter(manifest)
    factory = ModelAdapterFactory(specification)
    selected = pd.read_csv(selected_path)
    families = [str(value) for value in workflow["families"]]
    seeds = [int(value) for value in workflow["seeds"]]
    residual_features = [str(value) for value in workflow["residual_features"]]
    selected = selected.loc[selected["model_family"].isin(families)].copy()
    expected = {(family, fold) for family in families for fold in adapter.outer_fold_labels()}
    observed = set(zip(selected["model_family"], selected["outer_fold"], strict=False))
    if observed != expected:
        raise ValueError("PE_11 selected configurations do not cover every family/fold")

    root = run_directory(EXPERIMENTS_DIR, args.workflow, definition)
    output = root / "members"
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "member_predictions.csv.gz"
    completed_path = output / "completed_fits.csv"
    manifest_path = output / "member_manifest.json"
    if args.force:
        for path in (prediction_path, completed_path, manifest_path):
            path.unlink(missing_ok=True)
    existing = pd.read_csv(prediction_path) if prediction_path.is_file() else pd.DataFrame()
    completed = (
        pd.read_csv(completed_path)
        if completed_path.is_file()
        else pd.DataFrame(columns=["model_family", "outer_fold", "inner_fold", "seed"])
    )
    completed_keys = {
        (str(row.model_family), int(row.outer_fold), int(row.inner_fold), int(row.seed))
        for row in completed.itertuples(index=False)
    }
    records: list[dict] = []
    completion_records: list[dict] = completed.to_dict("records")
    total = len(expected) * len(adapter.inner_fold_labels(0)) * len(seeds)
    for configuration in selected.sort_values(["model_family", "outer_fold"]).itertuples(index=False):
        family = str(configuration.model_family)
        outer_fold = int(configuration.outer_fold)
        hyperparameters = json.loads(configuration.hyperparameters_json)
        feature_set = str(configuration.feature_set)
        for inner_fold in adapter.inner_fold_labels(outer_fold):
            split = adapter.get_inner_selection_split(
                outer_fold, int(inner_fold), feature_set
            )
            missing_features = sorted(
                set(residual_features) - set(split.validation.features.columns)
            )
            if missing_features:
                raise ValueError(
                    f"PE_11 residual features are missing: {missing_features}"
                )
            training_uavs = set(split.training.metadata["uav_id"].astype(str))
            validation_uavs = set(split.validation.metadata["uav_id"].astype(str))
            if training_uavs & validation_uavs:
                raise ValueError("PE_11 encountered overlapping training/validation UAVs")
            for seed in seeds:
                fit_key = (family, outer_fold, int(inner_fold), seed)
                if existing.empty:
                    existing_rows = 0
                else:
                    mask = (
                        existing["model_family"].astype(str).eq(family)
                        & existing["outer_fold"].astype(int).eq(outer_fold)
                        & existing["inner_fold"].astype(int).eq(int(inner_fold))
                        & existing["seed"].astype(int).eq(seed)
                    )
                    existing_rows = int(mask.sum())
                if fit_key in completed_keys and existing_rows != len(split.validation):
                    raise ValueError(
                        f"PE_11 completed checkpoint {fit_key} has "
                        f"{existing_rows}/{len(split.validation)} predictions"
                    )
                if fit_key not in completed_keys and existing_rows == len(split.validation):
                    completion_records.append(
                        {
                            "model_family": family,
                            "outer_fold": outer_fold,
                            "inner_fold": int(inner_fold),
                            "seed": seed,
                            "training_uavs": len(training_uavs),
                            "validation_uavs": len(validation_uavs),
                            "uav_overlap": 0,
                            "training_seconds": np.nan,
                            "recovered_from_predictions": True,
                        }
                    )
                    completed_keys.add(fit_key)
                    _write_csv_atomic(pd.DataFrame(completion_records), completed_path)
                elif fit_key not in completed_keys and existing_rows:
                    raise ValueError(
                        f"PE_11 incomplete prediction checkpoint {fit_key}: "
                        f"{existing_rows}/{len(split.validation)} rows"
                    )
                if fit_key in completed_keys:
                    continue
                model = factory.create(
                    family,
                    hyperparameters,
                    seed=seed,
                    training_monitor=NoOpTrainingMonitor(),
                )
                summary = model.fit(split.training, split.validation)
                prediction = model.predict(split.validation)
                metadata = split.validation.metadata.reset_index(drop=True)
                for validation_row, row in metadata.iterrows():
                    record = {
                            "outer_fold": outer_fold,
                            "inner_fold": int(inner_fold),
                            "validation_row": int(validation_row),
                            "uav_id": str(row["uav_id"]),
                            "scenario": str(row["scenario"]),
                            "cutoff": float(row["cutoff"]),
                            "observed_rul": float(split.validation.target.iloc[validation_row]),
                            "model_family": family,
                            "seed": seed,
                            "member": f"{family}__seed_{seed:03d}",
                            "predicted_rul": float(prediction[validation_row]),
                    }
                    for feature in residual_features:
                        record[feature] = float(
                            split.validation.features.iloc[validation_row][feature]
                        )
                    records.append(record)
                completion_records.append(
                    {
                        "model_family": family,
                        "outer_fold": outer_fold,
                        "inner_fold": int(inner_fold),
                        "seed": seed,
                        "training_uavs": len(training_uavs),
                        "validation_uavs": len(validation_uavs),
                        "uav_overlap": 0,
                        "training_seconds": summary.training_seconds,
                    }
                )
                new_rows = pd.DataFrame(records)
                combined = pd.concat([existing, new_rows], ignore_index=True)
                _write_csv_atomic(combined, prediction_path, gzip=True)
                _write_csv_atomic(pd.DataFrame(completion_records), completed_path)
                existing = combined
                records.clear()
                completed_keys.add(fit_key)
                print(
                    f"PE_11 members: {len(completed_keys)}/{total} fits complete",
                    flush=True,
                )
    if len(completed_keys) != total:
        raise ValueError(f"PE_11 member generation is incomplete: {len(completed_keys)}/{total}")
    if existing.duplicated(
        ["outer_fold", "inner_fold", "validation_row", "model_family", "seed"]
    ).any():
        raise ValueError("PE_11 member predictions contain duplicate fit rows")
    write_json(
        manifest_path,
        {
            "status": "complete",
            "completed_fits": len(completed_keys),
            "expected_fits": total,
            "prediction_rows": len(existing),
            "uses_locked_evaluation": False,
        },
    )
    print(f"PE_11 member predictions saved to {prediction_path}")


if __name__ == "__main__":
    main()
