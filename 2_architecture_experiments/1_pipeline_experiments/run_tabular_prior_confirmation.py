"""Confirm the PE_18 full-feature TabPFN blend on two new split seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from advanced_r2_utils import COMMON_KEYS, REPOSITORY_ROOT, load_workflow, method_report
from confirmation_utils import (
    EvaluationJob,
    evaluation_jobs,
    generated_partitions,
    prediction_records,
    select_uavs,
    validate_partitions,
)


PHASE1 = REPOSITORY_ROOT / "1_dataset_construction"
PHASE2 = REPOSITORY_ROOT / "2_architecture_experiments" / "2_model_architecture_study"
for directory in (
    PHASE1 / "2_UAV_grouped_validation_folds",
    PHASE2 / "2_tabular_data_adapter",
    PHASE2 / "4_model_adapters",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from model_registry import ModelAdapterFactory  # noqa: E402
from no_op_training_monitor import NoOpTrainingMonitor  # noqa: E402
from run_tabular_prior import (  # noqa: E402
    cross_fit_outer_blend,
    fit_tabpfn,
    fitting_target,
    tabpfn_dependency_status,
)
from tabular_data_adapter import TabularDataAdapter  # noqa: E402


def _method_complete(table: pd.DataFrame, job: EvaluationJob, method: str) -> bool:
    if table.empty:
        return False
    required = {
        "split_seed",
        "source_outer_fold",
        "evaluation_level",
        "inner_fold",
        "method",
    }
    if not required.issubset(table):
        raise ValueError("Existing PE_21 checkpoint lacks nesting metadata")
    rows = table.loc[
        table.split_seed.astype(int).eq(job.split_seed)
        & table.source_outer_fold.astype(int).eq(job.outer_fold)
        & table.evaluation_level.astype(str).eq(job.evaluation_level)
        & table.inner_fold.astype(int).eq(job.inner_fold)
        & table.method.astype(str).eq(method)
    ]
    keys = ["uav_id", "scenario", "cutoff"]
    return (
        not rows.empty
        and not rows.duplicated(keys).any()
        and set(rows.uav_id.astype(str)) == set(job.validation_uavs)
    )


def _fit_control(workflow: dict, training: object, validation: object) -> np.ndarray:
    factory = ModelAdapterFactory(REPOSITORY_ROOT / workflow["specification"])
    model = factory.create(
        "residual_corrected_tree_ensemble",
        {"ensemble_contract_path": str(workflow["source_contract"])},
        seed=int(workflow["model_seed"]),
        allow_disabled=True,
        training_monitor=NoOpTrainingMonitor(),
    )
    model.fit(training, None)
    return np.asarray(model.predict(validation), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_21")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workflow, _, root = load_workflow(
        args.config, "tabular_confirmation_workflows", args.workflow
    )
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    checkpoint = reporting / "fold_predictions.csv"
    if args.force:
        for path in (
            checkpoint,
            reporting / "winner_manifest.json",
            reporting / "blend_provenance.csv",
        ):
            path.unlink(missing_ok=True)

    dependency = tabpfn_dependency_status(workflow["tabpfn"])
    (reporting / "dependency_manifest.json").write_text(
        json.dumps(dependency, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not dependency["ready"]:
        raise RuntimeError(
            "PE_21 requires the pinned TabPFN package and local checkpoint; "
            "see dependency_manifest.json"
        )

    outer, inner = generated_partitions(
        history_summary_path=str(workflow["history_summary"]),
        split_seeds=[int(value) for value in workflow["split_seeds"]],
        outer_fold_count=int(workflow["outer_fold_count"]),
        inner_fold_count=int(workflow["inner_fold_count"]),
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
    completed = pd.read_csv(checkpoint) if checkpoint.is_file() else pd.DataFrame()

    for job_number, job in enumerate(jobs, start=1):
        training = select_uavs(full_training, job.training_uavs)
        validation = select_uavs(development, job.validation_uavs)
        for method in ("control", "tabpfn_full"):
            if _method_complete(completed, job, method):
                print(
                    f"PE_21 cell {job_number}/{len(jobs)} {method} already complete",
                    flush=True,
                )
                continue
            print(
                f"PE_21 cell {job_number}/{len(jobs)}: {method} "
                f"{job.evaluation_level} seed={job.split_seed} "
                f"outer={job.outer_fold} inner={job.inner_fold}",
                flush=True,
            )
            if method == "control":
                prediction = _fit_control(workflow, training, validation)
            else:
                prediction = fit_tabpfn(
                    training.features,
                    fitting_target(training, float(workflow["target_cap"])),
                    validation.features,
                    {
                        **workflow["tabpfn"],
                        "checkpoint": dependency["resolved_checkpoint"],
                    },
                )
            if not completed.empty:
                prior = (
                    completed.split_seed.astype(int).eq(job.split_seed)
                    & completed.source_outer_fold.astype(int).eq(job.outer_fold)
                    & completed.evaluation_level.astype(str).eq(job.evaluation_level)
                    & completed.inner_fold.astype(int).eq(job.inner_fold)
                    & completed.method.astype(str).eq(method)
                )
                completed = completed.loc[~prior].copy()
            completed = pd.concat(
                [
                    completed,
                    pd.DataFrame(
                        prediction_records(job, validation, {method: prediction})
                    ),
                ],
                ignore_index=True,
            )
            completed.to_csv(checkpoint, index=False)

    expected = len(jobs) * 2
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
        raise ValueError("PE_21 checkpoint contains duplicate prediction rows")
    observed = completed.groupby(
        [
            "split_seed",
            "source_outer_fold",
            "evaluation_level",
            "inner_fold",
            "method",
        ]
    ).ngroups
    if observed != expected:
        raise ValueError(f"PE_21 completed {observed}/{expected} prediction cells")

    outer_rows = completed.loc[completed.evaluation_level.eq("outer")].copy()
    inner_rows = completed.loc[completed.evaluation_level.eq("inner")].copy()
    order = ["outer_fold", "scenario", "uav_id", "cutoff"]
    identities = (
        outer_rows.loc[outer_rows.method.eq("control")]
        .sort_values(order)
        .reset_index(drop=True)
    )
    identity_columns = [*order, "observed_rul"]
    challenger = (
        outer_rows.loc[outer_rows.method.eq("tabpfn_full")]
        .sort_values(order)
        .reset_index(drop=True)
    )
    if not challenger[identity_columns].equals(identities[identity_columns]):
        raise ValueError("PE_21 outer TabPFN predictions do not align with control")

    selection_keys = [*COMMON_KEYS]
    inner_control = inner_rows.loc[inner_rows.method.eq("control"), selection_keys + ["predicted_rul"]].rename(
        columns={"predicted_rul": "control_prediction"}
    )
    inner_challenger = inner_rows.loc[
        inner_rows.method.eq("tabpfn_full"), selection_keys + ["predicted_rul"]
    ].rename(columns={"predicted_rul": "challenger_prediction"})
    selection = inner_control.merge(
        inner_challenger,
        on=selection_keys,
        validate="one_to_one",
    )
    blend_rows = identities[identity_columns].copy()
    blend_rows["control_prediction"] = identities.predicted_rul.to_numpy(float)
    blend_rows["challenger_prediction"] = challenger.predicted_rul.to_numpy(float)
    blend, provenance = cross_fit_outer_blend(
        blend_rows,
        selection,
        weights=[float(value) for value in workflow["challenger_weights"]],
    )
    for row in provenance:
        row["method"] = "control_plus_tabpfn_full"
    pd.DataFrame(provenance).to_csv(reporting / "blend_provenance.csv", index=False)
    methods = {
        "control": identities.predicted_rul.to_numpy(float),
        "tabpfn_full": challenger.predicted_rul.to_numpy(float),
        "control_plus_tabpfn_full": blend,
    }
    manifest = method_report(
        identities,
        methods,
        root=root,
        control="control",
        minimum_fold_wins=int(workflow["minimum_fold_wins"]),
        minimum_relative_rmse_improvement=float(
            workflow["minimum_relative_rmse_improvement"]
        ),
        method_improvement_thresholds={
            "tabpfn_full": float(workflow["minimum_relative_rmse_improvement"]),
            "control_plus_tabpfn_full": float(
                workflow["blend_minimum_relative_rmse_improvement"]
            ),
        },
    )
    dependency.update(
        {
            "all_tabpfn_cells_completed": True,
            "split_seeds": [int(value) for value in workflow["split_seeds"]],
            "nested_inner_predictions_complete": True,
            "uses_test_labels": False,
        }
    )
    manifest.update(dependency)
    (reporting / "winner_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reporting / "dependency_manifest.json").write_text(
        json.dumps(dependency, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
