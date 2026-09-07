"""Run complete nested outer confirmation of fixed-lag forecast history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from advanced_r2_utils import REPOSITORY_ROOT, load_workflow, method_report
from confirmation_utils import (
    cell_complete,
    evaluation_jobs,
    fixed_partitions,
    prediction_records,
    select_uavs,
    validate_partitions,
)


PHASE1 = REPOSITORY_ROOT / "1_dataset_construction"
PHASE2 = REPOSITORY_ROOT / "2_architecture_experiments" / "2_model_architecture_study"
for directory in (
    PHASE1 / "5_prefix_feature_engineering",
    PHASE2 / "2_tabular_data_adapter",
    PHASE2 / "4_model_adapters",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from history_corrected_tree_system import HistoryCorrectedTreeSystem  # noqa: E402
from model_registry import ModelAdapterFactory  # noqa: E402
from no_op_training_monitor import NoOpTrainingMonitor  # noqa: E402
from tabular_data_adapter import TabularDataAdapter  # noqa: E402


METHODS = {"control", "prediction_history"}


def _new_system(workflow: dict) -> HistoryCorrectedTreeSystem:
    factory = ModelAdapterFactory(REPOSITORY_ROOT / workflow["specification"])
    adapter = factory.create(
        "residual_corrected_tree_ensemble",
        {"ensemble_contract_path": str(workflow["source_contract"])},
        seed=int(workflow["model_seed"]),
        allow_disabled=True,
        training_monitor=NoOpTrainingMonitor(),
    )
    return HistoryCorrectedTreeSystem(
        adapter,
        history_lags=[int(value) for value in workflow["history_lags"]],
        feature_profile=str(workflow["feature_profile"]),
        near_cap_threshold=float(workflow["near_cap_threshold"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_20")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workflow, _, root = load_workflow(
        args.config, "history_confirmation_workflows", args.workflow
    )
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    checkpoint = reporting / "fold_predictions.csv"
    provenance_path = reporting / "fit_provenance.csv"
    if args.force:
        for path in (checkpoint, provenance_path, reporting / "winner_manifest.json"):
            path.unlink(missing_ok=True)

    outer, inner = fixed_partitions(
        outer_path=str(workflow["outer_folds"]),
        inner_path=str(workflow["inner_folds"]),
        split_seed=int(workflow["split_seed"]),
    )
    validate_partitions(
        outer,
        inner,
        expected_outer_folds=int(workflow["outer_fold_count"]),
        expected_inner_folds=int(workflow["inner_fold_count"]),
    )
    outer.to_csv(reporting / "outer_folds.csv", index=False)
    inner.to_csv(reporting / "inner_folds.csv", index=False)
    jobs = evaluation_jobs(outer, inner, include_inner=True)

    adapter = TabularDataAdapter(REPOSITORY_ROOT / workflow["tabular_manifest"])
    feature_set = str(workflow["feature_set"])
    full_training = adapter.load_training(feature_set)
    development = adapter.load_development(feature_set)
    raw = pd.read_csv(REPOSITORY_ROOT / workflow["train_csv"])
    completed = pd.read_csv(checkpoint) if checkpoint.is_file() else pd.DataFrame()
    provenance = (
        pd.read_csv(provenance_path) if provenance_path.is_file() else pd.DataFrame()
    )

    for job_number, job in enumerate(jobs, start=1):
        if cell_complete(completed, job, METHODS):
            print(
                f"PE_20 cell {job_number}/{len(jobs)} already complete: "
                f"{job.evaluation_level} seed={job.split_seed} "
                f"outer={job.outer_fold} inner={job.inner_fold}",
                flush=True,
            )
            continue
        print(
            f"PE_20 cell {job_number}/{len(jobs)}: {job.evaluation_level} "
            f"seed={job.split_seed} outer={job.outer_fold} inner={job.inner_fold}",
            flush=True,
        )
        training = select_uavs(full_training, job.training_uavs)
        validation = select_uavs(development, job.validation_uavs)
        system = _new_system(workflow)
        summary = system.fit(training, raw)
        predictions = system.predict_methods(validation, raw)
        cell_mask = pd.Series(False, index=completed.index)
        if not completed.empty:
            cell_mask = (
                completed.split_seed.astype(int).eq(job.split_seed)
                & completed.source_outer_fold.astype(int).eq(job.outer_fold)
                & completed.evaluation_level.astype(str).eq(job.evaluation_level)
                & completed.inner_fold.astype(int).eq(job.inner_fold)
            )
            completed = completed.loc[~cell_mask].copy()
        completed = pd.concat(
            [completed, pd.DataFrame(prediction_records(job, validation, predictions))],
            ignore_index=True,
        )
        completed.to_csv(checkpoint, index=False)
        summary_row = {
            "split_seed": job.split_seed,
            "outer_fold": job.report_fold,
            "source_outer_fold": job.outer_fold,
            "evaluation_level": job.evaluation_level,
            "inner_fold": job.inner_fold,
            "training_validation_uav_overlap": 0,
            **summary.to_dict(),
        }
        if not provenance.empty:
            prior = (
                provenance.split_seed.astype(int).eq(job.split_seed)
                & provenance.source_outer_fold.astype(int).eq(job.outer_fold)
                & provenance.evaluation_level.astype(str).eq(job.evaluation_level)
                & provenance.inner_fold.astype(int).eq(job.inner_fold)
            )
            provenance = provenance.loc[~prior].copy()
        provenance = pd.concat(
            [provenance, pd.DataFrame([summary_row])], ignore_index=True
        )
        provenance.to_csv(provenance_path, index=False)

    outer_rows = completed.loc[completed.evaluation_level.eq("outer")].copy()
    checkpoint_keys = [
        "split_seed",
        "source_outer_fold",
        "evaluation_level",
        "inner_fold",
        "method",
        "uav_id",
        "scenario",
        "cutoff",
    ]
    if completed.duplicated(checkpoint_keys).any():
        raise ValueError("PE_20 checkpoint contains duplicate prediction rows")
    order = ["outer_fold", "scenario", "uav_id", "cutoff"]
    identities = (
        outer_rows.loc[outer_rows.method.eq("control")]
        .sort_values(order)
        .reset_index(drop=True)
    )
    if identities.empty:
        raise ValueError("PE_20 has no completed outer control predictions")
    identity_columns = [*order, "observed_rul"]
    methods = {}
    for name, rows in outer_rows.groupby("method"):
        aligned = rows.sort_values(order).reset_index(drop=True)
        if not aligned[identity_columns].equals(identities[identity_columns]):
            raise ValueError(f"PE_20 method {name!r} does not align with the control")
        methods[str(name)] = aligned.predicted_rul.to_numpy(float)
    manifest = method_report(
        identities,
        methods,
        root=root,
        control="control",
        minimum_fold_wins=int(workflow["minimum_fold_wins"]),
        minimum_relative_rmse_improvement=float(
            workflow["minimum_relative_rmse_improvement"]
        ),
    )
    manifest.update(
        {
            "nested_inner_predictions_complete": True,
            "history_lags": [int(value) for value in workflow["history_lags"]],
            "fixed_split_seed": int(workflow["split_seed"]),
        }
    )
    (reporting / "winner_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
