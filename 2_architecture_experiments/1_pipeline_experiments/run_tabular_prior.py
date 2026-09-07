"""Evaluate pinned TabPFN and CatBoost in nested grouped comparisons."""

from __future__ import annotations

import argparse
from importlib import metadata
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from advanced_r2_utils import (
    COMMON_KEYS,
    REPOSITORY_ROOT,
    load_workflow,
    method_report,
    read_method_predictions,
)
from oof_experiment_utils import regression_metrics


PHASE2 = REPOSITORY_ROOT / "2_architecture_experiments" / "2_model_architecture_study"
if str(PHASE2 / "2_tabular_data_adapter") not in sys.path:
    sys.path.insert(0, str(PHASE2 / "2_tabular_data_adapter"))
from tabular_data_adapter import TabularDataAdapter  # noqa: E402


def compact_columns(features: pd.DataFrame, target: np.ndarray, count: int) -> list[str]:
    """Rank features using training-target correlation only."""

    correlations = []
    centered_target = target - target.mean()
    target_norm = float(np.linalg.norm(centered_target))
    for column in features:
        values = features[column].to_numpy(float)
        centered = values - values.mean()
        denominator = float(np.linalg.norm(centered) * target_norm)
        score = (
            abs(float(np.dot(centered, centered_target) / denominator))
            if denominator > 0
            else 0.0
        )
        correlations.append((score, str(column)))
    return [column for _, column in sorted(correlations, reverse=True)[:count]]


def fit_tabpfn(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    validation_x: pd.DataFrame,
    settings: dict,
) -> np.ndarray:
    from tabpfn import TabPFNRegressor

    parameters = inspect.signature(TabPFNRegressor).parameters
    kwargs = {}
    if "device" in parameters:
        kwargs["device"] = str(settings["device"])
    checkpoint = str(settings.get("checkpoint", "auto"))
    checkpoint_argument = None
    for name in ("model_path", "model_name", "model_name_or_path"):
        if checkpoint != "auto" and name in parameters:
            kwargs[name] = checkpoint
            checkpoint_argument = name
            break
    if checkpoint != "auto" and checkpoint_argument is None:
        raise RuntimeError(
            "The pinned TabPFN API exposes no recognized local-checkpoint argument"
        )
    model = TabPFNRegressor(**kwargs)
    model.fit(train_x.to_numpy(float), train_y)
    return np.asarray(model.predict(validation_x.to_numpy(float)), float).reshape(-1)


def tabpfn_dependency_status(settings: dict) -> dict:
    expected_version = str(settings["version"])
    try:
        installed_version = metadata.version("tabpfn")
    except metadata.PackageNotFoundError:
        installed_version = None
    checkpoint_value = str(settings["checkpoint"])
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = REPOSITORY_ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    import_error = None
    if installed_version == expected_version:
        try:
            __import__("tabpfn")
        except Exception as error:
            import_error = f"{type(error).__name__}: {error}"
    ready = (
        installed_version == expected_version
        and checkpoint.is_file()
        and import_error is None
    )
    return {
        "expected_version": expected_version,
        "installed_version": installed_version,
        "checkpoint": checkpoint_value,
        "resolved_checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "import_error": import_error,
        "ready": ready,
        "uses_remote_api": False,
    }


def fit_catboost(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    validation_x: pd.DataFrame,
    weights: np.ndarray | None,
    settings: dict,
) -> np.ndarray:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=int(settings["iterations"]),
        depth=int(settings["depth"]),
        learning_rate=float(settings["learning_rate"]),
        l2_leaf_reg=float(settings["l2_leaf_reg"]),
        loss_function="RMSE",
        random_seed=int(settings["seed"]),
        verbose=False,
        allow_writing_files=False,
        thread_count=int(settings["thread_count"]),
    )
    model.fit(train_x, train_y, sample_weight=weights)
    return np.asarray(model.predict(validation_x), float)


def cross_fit_outer_blend(
    outer_rows: pd.DataFrame,
    selection_rows: pd.DataFrame,
    *,
    weights: list[float],
) -> tuple[np.ndarray, list[dict]]:
    """Select each blend on inner OOF rows and apply it to held outer UAVs."""

    prediction = np.full(len(outer_rows), np.nan, dtype=float)
    provenance = []
    for outer_fold, held in outer_rows.groupby("outer_fold", sort=True):
        training = selection_rows.loc[selection_rows.outer_fold.eq(outer_fold)]
        if training.empty:
            raise ValueError(f"PE_18 has no inner predictions for fold {outer_fold}")
        if set(training.uav_id.astype(str)) & set(held.uav_id.astype(str)):
            raise ValueError("PE_18 blend selection has UAV overlap")
        scored = []
        for weight in weights:
            estimate = (
                (1.0 - weight) * training.control_prediction.to_numpy(float)
                + weight * training.challenger_prediction.to_numpy(float)
            )
            scored.append(
                (regression_metrics(training.observed_rul, estimate)["rmse"], weight)
            )
        _, selected = min(scored, key=lambda item: (item[0], item[1]))
        prediction[held.index] = (
            (1.0 - selected) * held.control_prediction.to_numpy(float)
            + selected * held.challenger_prediction.to_numpy(float)
        )
        provenance.append(
            {
                "outer_fold": int(outer_fold),
                "selected_challenger_weight": float(selected),
                "training_uavs": training.uav_id.astype(str).nunique(),
                "validation_uavs": held.uav_id.astype(str).nunique(),
                "uav_overlap": 0,
            }
        )
    if not np.isfinite(prediction).all():
        raise ValueError("PE_18 blend predictions are incomplete")
    return prediction, provenance


def fitting_target(dataset: Any, target_cap: float) -> np.ndarray:
    if dataset.fitting_target is not None:
        return dataset.fitting_target.to_numpy(float)
    return np.minimum(dataset.target.to_numpy(float), target_cap)


def fit_prediction_cells(
    adapter: TabularDataAdapter,
    workflow: dict,
    tabpfn_status: dict,
    *,
    checkpoint: Path,
    force: bool,
) -> tuple[pd.DataFrame, list[dict]]:
    if force:
        checkpoint.unlink(missing_ok=True)
    completed = pd.read_csv(checkpoint) if checkpoint.is_file() else pd.DataFrame()
    if not completed.empty and "evaluation_level" not in completed:
        raise ValueError("Old PE_18 checkpoint lacks nesting metadata; rerun with --force")
    unavailable = []
    feature_set = str(workflow["feature_set"])
    target_cap = float(workflow["target_cap"])
    records = []
    for outer_fold in adapter.outer_fold_labels():
        evaluations = [
            ("outer", -1, adapter.get_final_search_split(outer_fold, feature_set))
        ]
        evaluations.extend(
            (
                "inner",
                int(inner_fold),
                adapter.get_inner_selection_split(outer_fold, inner_fold, feature_set),
            )
            for inner_fold in adapter.inner_fold_labels(outer_fold)
        )
        for evaluation_level, inner_fold, split in evaluations:
            train_y = fitting_target(split.training, target_cap)
            compact = compact_columns(
                split.training.features,
                train_y,
                int(workflow["compact_feature_count"]),
            )
            views = {"full": list(split.training.features.columns), "compact": compact}
            for family in ("catboost", "tabpfn"):
                for view, columns in views.items():
                    method = f"{family}_{view}"
                    already_complete = (
                        not completed.empty
                        and (
                            completed.method.astype(str).eq(method)
                            & completed.evaluation_level.astype(str).eq(evaluation_level)
                            & completed.outer_fold.astype(int).eq(int(outer_fold))
                            & completed.inner_fold.astype(int).eq(int(inner_fold))
                        ).any()
                    )
                    if already_complete:
                        continue
                    try:
                        if family == "catboost":
                            prediction = fit_catboost(
                                split.training.features[columns],
                                train_y,
                                split.validation.features[columns],
                                None
                                if split.training.sample_weights is None
                                else split.training.sample_weights.to_numpy(float),
                                workflow["catboost"],
                            )
                        else:
                            prediction = fit_tabpfn(
                                split.training.features[columns],
                                train_y,
                                split.validation.features[columns],
                                {
                                    **workflow["tabpfn"],
                                    "checkpoint": tabpfn_status["resolved_checkpoint"],
                                },
                            )
                    except (ImportError, ModuleNotFoundError) as error:
                        unavailable.append(
                            {
                                "method": method,
                                "evaluation_level": evaluation_level,
                                "outer_fold": int(outer_fold),
                                "inner_fold": int(inner_fold),
                                "reason": str(error),
                            }
                        )
                        continue
                    metadata_rows = split.validation.metadata.reset_index(drop=True)
                    for index, row in metadata_rows.iterrows():
                        records.append(
                            {
                                "evaluation_level": evaluation_level,
                                "outer_fold": int(outer_fold),
                                "inner_fold": int(inner_fold),
                                "uav_id": str(row.uav_id),
                                "scenario": str(row.scenario),
                                "cutoff": float(row.cutoff),
                                "observed_rul": float(split.validation.target.iloc[index]),
                                "method": method,
                                "predicted_rul": max(float(prediction[index]), 0.0),
                                "selected_feature_count": len(columns),
                            }
                        )
                    completed = pd.concat(
                        [completed, pd.DataFrame(records)], ignore_index=True
                    )
                    completed.to_csv(checkpoint, index=False)
                    records.clear()
    return completed, unavailable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_18")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workflow, _, root = load_workflow(
        args.config, "tabular_prior_workflows", args.workflow
    )
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    dependency = tabpfn_dependency_status(workflow["tabpfn"])
    (reporting / "dependency_manifest.json").write_text(
        json.dumps(dependency, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if bool(workflow["tabpfn"].get("required", False)) and not dependency["ready"]:
        raise RuntimeError(
            "PE_18 requires the pinned TabPFN package and local checkpoint; "
            "see dependency_manifest.json and the experiment README"
        )

    adapter = TabularDataAdapter(REPOSITORY_ROOT / workflow["tabular_manifest"])
    completed, unavailable = fit_prediction_cells(
        adapter,
        workflow,
        dependency,
        checkpoint=reporting / "fold_predictions.csv",
        force=args.force,
    )
    outer = completed.loc[completed.evaluation_level.eq("outer")].copy()
    inner = completed.loc[completed.evaluation_level.eq("inner")].copy()
    control = pd.read_csv(REPOSITORY_ROOT / workflow["control_predictions"])
    control = control.loc[
        control.candidate_number.eq(int(workflow["control_candidate_number"]))
    ]
    identities = control[
        ["outer_fold", "uav_id", "scenario", "cutoff", "observed_rul", "predicted_rul"]
    ].copy()
    identities["inner_fold"] = -1
    order = ["outer_fold", "scenario", "uav_id", "cutoff"]
    identities = identities.sort_values(order).reset_index(drop=True)
    identity_columns = [*order, "observed_rul"]
    methods = {"control": identities.predicted_rul.to_numpy(float)}

    inner_control = read_method_predictions(
        workflow["control_inner_predictions"],
        method=str(workflow["control_inner_method"]),
    )[COMMON_KEYS + ["predicted_rul"]].rename(
        columns={"predicted_rul": "control_prediction"}
    )
    blend_provenance = []
    threshold_overrides = {}
    for name, group in outer.groupby("method"):
        aligned = group.sort_values(order).reset_index(drop=True)
        if not aligned[identity_columns].equals(identities[identity_columns]):
            raise ValueError(f"PE_18 method {name} does not align with the control")
        direct = aligned.predicted_rul.to_numpy(float)
        methods[str(name)] = direct
        threshold_overrides[str(name)] = float(
            workflow["minimum_relative_rmse_improvement"]
        )
        inner_challenger = inner.loc[inner.method.eq(name)][
            COMMON_KEYS + ["predicted_rul"]
        ].rename(columns={"predicted_rul": "challenger_prediction"})
        selection_rows = inner_control.merge(
            inner_challenger, on=COMMON_KEYS, validate="one_to_one"
        ).reset_index(drop=True)
        if len(selection_rows) != len(inner_control):
            raise ValueError(f"PE_18 inner predictions for {name} do not align")
        blend_rows = identities[identity_columns].copy()
        blend_rows["control_prediction"] = identities.predicted_rul.to_numpy(float)
        blend_rows["challenger_prediction"] = direct
        blend_prediction, records = cross_fit_outer_blend(
            blend_rows,
            selection_rows,
            weights=[float(value) for value in workflow["challenger_weights"]],
        )
        blend_name = f"control_plus_{name}"
        methods[blend_name] = blend_prediction
        threshold_overrides[blend_name] = float(
            workflow["blend_minimum_relative_rmse_improvement"]
        )
        for record in records:
            record["method"] = blend_name
        blend_provenance.extend(records)
    if set(methods) == {"control"}:
        raise ValueError("PE_18 produced no complete challenger methods")
    pd.DataFrame(blend_provenance).to_csv(
        reporting / "blend_provenance.csv", index=False
    )
    manifest = method_report(
        identities,
        methods,
        root=root,
        control="control",
        minimum_fold_wins=int(workflow["minimum_fold_wins"]),
        minimum_relative_rmse_improvement=float(
            workflow["minimum_relative_rmse_improvement"]
        ),
        method_improvement_thresholds=threshold_overrides,
    )
    dependency.update(
        {
            "unavailable_cells": unavailable,
            "all_tabpfn_cells_completed": not any(
                row["method"].startswith("tabpfn") for row in unavailable
            ),
            "uses_test_labels": False,
        }
    )
    (reporting / "dependency_manifest.json").write_text(
        json.dumps(dependency, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if unavailable and bool(workflow["tabpfn"].get("required", False)):
        raise RuntimeError("TabPFN is required but unavailable; see dependency_manifest.json")
    print(json.dumps({**manifest, **dependency}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
