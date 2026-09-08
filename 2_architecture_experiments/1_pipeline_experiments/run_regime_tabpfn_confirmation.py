"""Evaluate a frozen regime-aware TabPFN blend on fresh grouped split seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

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
from oof_experiment_utils import regression_metrics  # noqa: E402
from run_tabular_prior import (  # noqa: E402
    cross_fit_outer_blend,
    fit_tabpfn,
    fitting_target,
    tabpfn_dependency_status,
)
from tabular_data_adapter import TabularDataAdapter  # noqa: E402


CONTROL_DIAGNOSTICS = (
    "base_prediction",
    "uncertainty_std",
    "uncertainty_range",
    "family_disagreement",
)
FITTED_METHODS = {"control", "tabpfn_full"}


def pre_registration_contract(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return the scientific choices that must remain fixed across resumes."""

    return {
        "split_seeds": [int(value) for value in workflow["split_seeds"]],
        "outer_fold_count": int(workflow["outer_fold_count"]),
        "inner_fold_count": int(workflow["inner_fold_count"]),
        "feature_set": str(workflow["feature_set"]),
        "source_contract": str(workflow["source_contract"]),
        "model_seed": int(workflow["model_seed"]),
        "target_cap": float(workflow["target_cap"]),
        "gate_features": [str(value) for value in workflow["gate_features"]],
        "gate_maximum_depth": int(workflow["gate_maximum_depth"]),
        "gate_minimum_samples_leaf": int(workflow["gate_minimum_samples_leaf"]),
        "gate_maximum_tabpfn_weight": float(
            workflow["gate_maximum_tabpfn_weight"]
        ),
        "gate_minimum_challenger_gap": float(
            workflow["gate_minimum_challenger_gap"]
        ),
        "gate_random_state": int(workflow["gate_random_state"]),
        "global_anchor_weights": [
            float(value) for value in workflow["global_anchor_weights"]
        ],
        "minimum_fold_wins": int(workflow["minimum_fold_wins"]),
        "minimum_relative_rmse_improvement": float(
            workflow["minimum_relative_rmse_improvement"]
        ),
        "minimum_pooled_r2": float(workflow["minimum_pooled_r2"]),
        "require_bootstrap_improvement": bool(
            workflow["require_bootstrap_improvement"]
        ),
        "tabpfn": {
            "version": str(workflow["tabpfn"]["version"]),
            "checkpoint": str(workflow["tabpfn"]["checkpoint"]),
            "sha256": str(workflow["tabpfn"]["sha256"]),
            "device": str(workflow["tabpfn"]["device"]),
        },
    }


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
        raise ValueError("Existing PE_24 checkpoint lacks nesting metadata")
    rows = table.loc[
        table.split_seed.astype(int).eq(job.split_seed)
        & table.source_outer_fold.astype(int).eq(job.outer_fold)
        & table.evaluation_level.astype(str).eq(job.evaluation_level)
        & table.inner_fold.astype(int).eq(job.inner_fold)
        & table.method.astype(str).eq(method)
    ]
    keys = ["uav_id", "scenario", "cutoff"]
    complete = (
        not rows.empty
        and not rows.duplicated(keys).any()
        and set(rows.uav_id.astype(str)) == set(job.validation_uavs)
    )
    if complete and method == "control":
        if not set(CONTROL_DIAGNOSTICS).issubset(rows):
            raise ValueError("Existing PE_24 control checkpoint lacks diagnostics")
        complete = bool(
            np.isfinite(rows[list(CONTROL_DIAGNOSTICS)].to_numpy(float)).all()
        )
    return complete


def _fit_control(
    workflow: dict[str, Any],
    training: Any,
    validation: Any,
) -> pd.DataFrame:
    factory = ModelAdapterFactory(REPOSITORY_ROOT / workflow["specification"])
    model = factory.create(
        "residual_corrected_tree_ensemble",
        {"ensemble_contract_path": str(workflow["source_contract"])},
        seed=int(workflow["model_seed"]),
        allow_disabled=True,
        training_monitor=NoOpTrainingMonitor(),
    )
    model.fit(training, None)
    return model.predict_with_diagnostics(validation)


def _control_records(
    job: EvaluationJob,
    validation: Any,
    diagnostics: pd.DataFrame,
) -> list[dict[str, Any]]:
    records = prediction_records(
        job,
        validation,
        {"control": diagnostics["predicted_rul"].to_numpy(float)},
    )
    for index, record in enumerate(records):
        for name in CONTROL_DIAGNOSTICS:
            record[name] = float(diagnostics.iloc[index][name])
    return records


def _gate_matrix(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = sorted(set(columns) - set(rows.columns))
    if missing:
        raise ValueError(f"Regime gate features are missing: {missing}")
    matrix = rows[columns].astype(float)
    if not np.isfinite(matrix.to_numpy(float)).all():
        raise ValueError("Regime gate features contain non-finite values")
    return matrix


def _equal_uav_row_weights(rows: pd.DataFrame) -> NDArray[np.float64]:
    uavs = rows["uav_id"].astype(str)
    counts = uavs.value_counts()
    weights = np.asarray([1.0 / float(counts[value]) for value in uavs], dtype=float)
    return weights * len(weights) / weights.sum()


def cross_fit_regime_blend(
    outer_rows: pd.DataFrame,
    selection_rows: pd.DataFrame,
    *,
    feature_columns: list[str],
    maximum_depth: int,
    minimum_samples_leaf: int,
    maximum_tabpfn_weight: float,
    minimum_challenger_gap: float,
    random_state: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[dict[str, Any]]]:
    """Fit one bounded gate per outer fold using only inner OOF predictions."""

    if maximum_depth < 1 or minimum_samples_leaf < 1:
        raise ValueError("Regime tree depth and leaf size must be positive")
    if not 0.0 < maximum_tabpfn_weight <= 1.0:
        raise ValueError("Maximum TabPFN weight must be in (0, 1]")
    if minimum_challenger_gap < 0.0:
        raise ValueError("Minimum challenger gap must be nonnegative")

    prediction = np.full(len(outer_rows), np.nan, dtype=float)
    selected_weights = np.full(len(outer_rows), np.nan, dtype=float)
    provenance: list[dict[str, Any]] = []
    for outer_fold, held in outer_rows.groupby("outer_fold", sort=True):
        training = selection_rows.loc[selection_rows.outer_fold.eq(outer_fold)].copy()
        if training.empty:
            raise ValueError(f"PE_24 has no inner predictions for fold {outer_fold}")
        overlap = set(training.uav_id.astype(str)) & set(held.uav_id.astype(str))
        if overlap:
            raise ValueError("PE_24 gate fitting has outer-held UAV overlap")

        training_delta = (
            training.challenger_prediction.to_numpy(float)
            - training.control_prediction.to_numpy(float)
        )
        usable = np.abs(training_delta) >= minimum_challenger_gap
        target = np.zeros(len(training), dtype=float)
        target[usable] = (
            training.observed_rul.to_numpy(float)[usable]
            - training.control_prediction.to_numpy(float)[usable]
        ) / training_delta[usable]
        sample_weight = _equal_uav_row_weights(training) * np.square(training_delta)
        sample_weight[~usable] = 0.0
        if int(np.count_nonzero(sample_weight)) < 2 * minimum_samples_leaf:
            raise ValueError(
                f"PE_24 fold {outer_fold} has too few informative gate rows"
            )

        tree = DecisionTreeRegressor(
            max_depth=maximum_depth,
            min_samples_leaf=minimum_samples_leaf,
            random_state=random_state,
        )
        tree.fit(
            _gate_matrix(training, feature_columns),
            target,
            sample_weight=sample_weight,
        )
        raw_weight = tree.predict(_gate_matrix(held, feature_columns))
        weight = np.clip(raw_weight, 0.0, maximum_tabpfn_weight)
        estimate = (
            held.control_prediction.to_numpy(float)
            + weight
            * (
                held.challenger_prediction.to_numpy(float)
                - held.control_prediction.to_numpy(float)
            )
        )
        prediction[held.index] = estimate
        selected_weights[held.index] = weight

        training_weight = np.clip(
            tree.predict(_gate_matrix(training, feature_columns)),
            0.0,
            maximum_tabpfn_weight,
        )
        training_estimate = (
            training.control_prediction.to_numpy(float)
            + training_weight * training_delta
        )
        provenance.append(
            {
                "outer_fold": int(outer_fold),
                "training_rows": len(training),
                "training_uavs": training.uav_id.astype(str).nunique(),
                "validation_rows": len(held),
                "validation_uavs": held.uav_id.astype(str).nunique(),
                "uav_overlap": 0,
                "informative_training_rows": int(usable.sum()),
                "training_control_rmse": regression_metrics(
                    training.observed_rul,
                    training.control_prediction,
                )["rmse"],
                "training_regime_rmse": regression_metrics(
                    training.observed_rul,
                    training_estimate,
                )["rmse"],
                "held_weight_mean": float(weight.mean()),
                "held_weight_minimum": float(weight.min()),
                "held_weight_maximum": float(weight.max()),
                "held_zero_weight_fraction": float(np.mean(weight == 0.0)),
                "held_maximum_weight_fraction": float(
                    np.mean(weight == maximum_tabpfn_weight)
                ),
                "feature_importances_json": json.dumps(
                    {
                        name: float(value)
                        for name, value in zip(
                            feature_columns,
                            tree.feature_importances_,
                            strict=True,
                        )
                    },
                    sort_keys=True,
                ),
                "tree_rules": export_text(tree, feature_names=feature_columns),
            }
        )
    if not np.isfinite(prediction).all() or not np.isfinite(selected_weights).all():
        raise ValueError("PE_24 regime predictions are incomplete")
    return prediction, selected_weights, provenance


def _aligned_predictions(rows: pd.DataFrame) -> pd.DataFrame:
    keys = [*COMMON_KEYS]
    control = rows.loc[
        rows.method.eq("control"),
        [*keys, "predicted_rul", *CONTROL_DIAGNOSTICS],
    ].rename(columns={"predicted_rul": "control_prediction"})
    challenger = rows.loc[
        rows.method.eq("tabpfn_full"),
        [*keys, "predicted_rul"],
    ].rename(columns={"predicted_rul": "challenger_prediction"})
    aligned = control.merge(challenger, on=keys, validate="one_to_one")
    aligned["challenger_gap"] = (
        aligned.challenger_prediction - aligned.control_prediction
    )
    return aligned.sort_values(
        ["outer_fold", "scenario", "uav_id", "cutoff"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_24")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workflow, _, root = load_workflow(
        args.config,
        "regime_tabpfn_workflows",
        args.workflow,
    )
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    checkpoint = reporting / "fold_predictions.csv"
    registration = pre_registration_contract(workflow)
    registration_path = reporting / "pre_registration.json"
    if registration_path.is_file():
        existing_registration = json.loads(
            registration_path.read_text(encoding="utf-8")
        )
        if existing_registration != registration:
            raise RuntimeError(
                "PE_24 settings changed after the run was registered; use a new "
                "pipeline.run directory for a different scientific experiment"
            )
    else:
        registration_path.write_text(
            json.dumps(registration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.force:
        for path in (
            checkpoint,
            reporting / "winner_manifest.json",
            reporting / "gate_provenance.csv",
            reporting / "regime_weights.csv",
        ):
            path.unlink(missing_ok=True)

    dependency = tabpfn_dependency_status(workflow["tabpfn"])
    (reporting / "dependency_manifest.json").write_text(
        json.dumps(dependency, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not dependency["ready"]:
        raise RuntimeError(
            "PE_24 requires the pinned TabPFN package and local checkpoint; "
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
                    f"PE_24 cell {job_number}/{len(jobs)} {method} already complete",
                    flush=True,
                )
                continue
            print(
                f"PE_24 cell {job_number}/{len(jobs)}: {method} "
                f"{job.evaluation_level} seed={job.split_seed} "
                f"outer={job.outer_fold} inner={job.inner_fold}",
                flush=True,
            )
            if method == "control":
                diagnostics = _fit_control(workflow, training, validation)
                records = _control_records(job, validation, diagnostics)
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
                records = prediction_records(
                    job,
                    validation,
                    {"tabpfn_full": prediction},
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
                [completed, pd.DataFrame.from_records(records)],
                ignore_index=True,
            )
            completed.to_csv(checkpoint, index=False)

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
        raise ValueError("PE_24 checkpoint contains duplicate prediction rows")
    observed = completed.groupby(
        [
            "split_seed",
            "source_outer_fold",
            "evaluation_level",
            "inner_fold",
            "method",
        ]
    ).ngroups
    expected = len(jobs) * len(FITTED_METHODS)
    if observed != expected:
        raise ValueError(f"PE_24 completed {observed}/{expected} prediction cells")

    outer_rows = _aligned_predictions(
        completed.loc[completed.evaluation_level.eq("outer")]
    )
    inner_rows = _aligned_predictions(
        completed.loc[completed.evaluation_level.eq("inner")]
    )
    feature_columns = [str(value) for value in workflow["gate_features"]]
    regime, selected_weights, gate_provenance = cross_fit_regime_blend(
        outer_rows,
        inner_rows,
        feature_columns=feature_columns,
        maximum_depth=int(workflow["gate_maximum_depth"]),
        minimum_samples_leaf=int(workflow["gate_minimum_samples_leaf"]),
        maximum_tabpfn_weight=float(workflow["gate_maximum_tabpfn_weight"]),
        minimum_challenger_gap=float(workflow["gate_minimum_challenger_gap"]),
        random_state=int(workflow["gate_random_state"]),
    )
    pd.DataFrame(gate_provenance).to_csv(
        reporting / "gate_provenance.csv",
        index=False,
    )
    weight_rows = outer_rows[COMMON_KEYS].copy()
    weight_rows["selected_tabpfn_weight"] = selected_weights
    weight_rows.to_csv(reporting / "regime_weights.csv", index=False)

    global_blend, global_provenance = cross_fit_outer_blend(
        outer_rows,
        inner_rows,
        weights=[float(value) for value in workflow["global_anchor_weights"]],
    )
    pd.DataFrame(global_provenance).to_csv(
        reporting / "global_blend_provenance.csv",
        index=False,
    )
    methods = {
        "control": outer_rows.control_prediction.to_numpy(float),
        "tabpfn_full": outer_rows.challenger_prediction.to_numpy(float),
        "global_tabpfn_blend": global_blend,
        "regime_tabpfn_blend": regime,
    }
    primary_method = "regime_tabpfn_blend"
    manifest = method_report(
        outer_rows,
        methods,
        root=root,
        control="control",
        minimum_fold_wins=int(workflow["minimum_fold_wins"]),
        minimum_relative_rmse_improvement=float(
            workflow["minimum_relative_rmse_improvement"]
        ),
        promotion_eligible_methods={primary_method},
        require_bootstrap_improvement=bool(
            workflow["require_bootstrap_improvement"]
        ),
        minimum_pooled_r2=float(workflow["minimum_pooled_r2"]),
    )
    gate_contract = {
        "features": feature_columns,
        "maximum_depth": int(workflow["gate_maximum_depth"]),
        "minimum_samples_leaf": int(workflow["gate_minimum_samples_leaf"]),
        "maximum_tabpfn_weight": float(workflow["gate_maximum_tabpfn_weight"]),
        "minimum_challenger_gap": float(workflow["gate_minimum_challenger_gap"]),
        "random_state": int(workflow["gate_random_state"]),
        "target": "bounded_squared_error_optimal_tabpfn_weight",
        "sample_weight": "equal_uav_rows_times_squared_challenger_gap",
    }
    dependency.update(
        {
            "all_prediction_cells_completed": True,
            "split_seeds": [int(value) for value in workflow["split_seeds"]],
            "nested_inner_predictions_complete": True,
            "primary_method": primary_method,
            "diagnostic_only_methods": ["tabpfn_full", "global_tabpfn_blend"],
            "gate_contract": gate_contract,
            "pre_registration": registration,
            "pre_registration_path": str(
                registration_path.relative_to(REPOSITORY_ROOT)
            ).replace("\\", "/"),
            "uses_locked_evaluation": False,
            "uses_test_labels": False,
        }
    )
    manifest.update(dependency)
    (reporting / "winner_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (reporting / "dependency_manifest.json").write_text(
        json.dumps(dependency, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
