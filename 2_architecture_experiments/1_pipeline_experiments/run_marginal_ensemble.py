"""Refit temporal challengers on outer folds and cross-fit small blend weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
for directory in (PHASE2 / "3_sequence_data_adapter", PHASE2 / "4_model_adapters"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from model_registry import ModelAdapterFactory  # noqa: E402
from no_op_training_monitor import NoOpTrainingMonitor  # noqa: E402
from sequence_data_adapter import SequenceDataAdapter  # noqa: E402


IDENTITY_COLUMNS = ["outer_fold", "uav_id", "scenario", "cutoff", "observed_rul"]


def cross_fit_blend(
    rows: pd.DataFrame,
    weights: list[float],
    selection_rows: pd.DataFrame,
) -> tuple[np.ndarray, list[dict]]:
    """Choose each weight inside the outer training UAVs and apply it externally."""

    predicted = np.full(len(rows), np.nan)
    provenance = []
    for outer_fold, held in rows.groupby("outer_fold", sort=True):
        training = selection_rows.loc[selection_rows.outer_fold.eq(outer_fold)]
        if training.empty:
            raise ValueError(f"PE_19 has no inner selection rows for fold {outer_fold}")
        if set(training.uav_id.astype(str)) & set(held.uav_id.astype(str)):
            raise ValueError("PE_19 blend selection has UAV overlap")
        scored = []
        for weight in weights:
            estimate = (
                (1.0 - weight) * training.tree_prediction.to_numpy(float)
                + weight * training.challenger_prediction.to_numpy(float)
            )
            score = regression_metrics(training.observed_rul, estimate)["rmse"]
            scored.append((score, weight))
        _, selected = min(scored, key=lambda item: (item[0], item[1]))
        predicted[held.index] = (
            (1.0 - selected) * held.tree_prediction.to_numpy(float)
            + selected * held.challenger_prediction.to_numpy(float)
        )
        provenance.append(
            {
                "outer_fold": int(outer_fold),
                "selected_challenger_weight": float(selected),
                "training_uavs": training.uav_id.nunique(),
                "validation_uavs": held.uav_id.nunique(),
                "uav_overlap": 0,
            }
        )
    if not np.isfinite(predicted).all():
        raise ValueError("PE_19 predictions are incomplete")
    return predicted, provenance


def fit_temporal_outer_predictions(
    workflow: dict,
    *,
    checkpoint: Path,
    force: bool,
) -> pd.DataFrame:
    if force:
        checkpoint.unlink(missing_ok=True)
    completed = pd.read_csv(checkpoint) if checkpoint.is_file() else pd.DataFrame()
    selected = pd.read_csv(REPOSITORY_ROOT / workflow["selected_configurations"])
    adapter = SequenceDataAdapter(REPOSITORY_ROOT / workflow["sequence_manifest"])
    factory = ModelAdapterFactory(REPOSITORY_ROOT / workflow["specification"])
    records = []
    for family in [str(value) for value in workflow["challengers"]]:
        family_rows = selected.loc[selected.model_family.astype(str).eq(family)]
        if family_rows.outer_fold.nunique() != len(adapter.outer_fold_labels()):
            raise ValueError(f"PE_19 needs one selected {family} configuration per outer fold")
        for fold in adapter.outer_fold_labels():
            if (
                not completed.empty
                and (
                    completed.model_family.astype(str).eq(family)
                    & completed.outer_fold.astype(int).eq(int(fold))
                ).any()
            ):
                continue
            choice = family_rows.loc[family_rows.outer_fold.astype(int).eq(int(fold))]
            if len(choice) != 1:
                raise ValueError(f"PE_19 expected one {family} configuration for fold {fold}")
            row = choice.iloc[0]
            hyperparameters = json.loads(str(row.hyperparameters_json))
            lookback = int(row.lookback)
            iterations = row.outer_retraining_iterations
            fixed_iterations = None if pd.isna(iterations) else int(iterations)
            split = adapter.get_final_search_split(int(fold), lookback)
            model = factory.create(
                family,
                hyperparameters,
                seed=int(workflow["model_seed"]),
                training_iterations=fixed_iterations,
                training_monitor=NoOpTrainingMonitor(),
            )
            model.fit(split.training, None)
            prediction = np.asarray(model.predict(split.validation), dtype=float)
            metadata_rows = split.validation.metadata.reset_index(drop=True)
            for index, metadata_row in metadata_rows.iterrows():
                records.append(
                    {
                        "model_family": family,
                        "outer_fold": int(fold),
                        "uav_id": str(metadata_row.uav_id),
                        "scenario": str(metadata_row.scenario),
                        "cutoff": float(metadata_row.cutoff),
                        "observed_rul": float(split.validation.target.iloc[index]),
                        "predicted_rul": max(float(prediction[index]), 0.0),
                        "configuration_id": str(row.configuration_id),
                        "lookback": lookback,
                        "fixed_training_iterations": fixed_iterations,
                    }
                )
            completed = pd.concat([completed, pd.DataFrame(records)], ignore_index=True)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            completed.to_csv(checkpoint, index=False, compression="gzip")
            records.clear()
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_19")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workflow, _, root = load_workflow(
        args.config, "marginal_ensemble_workflows", args.workflow
    )
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    temporal = fit_temporal_outer_predictions(
        workflow,
        checkpoint=reporting / "temporal_outer_predictions.csv.gz",
        force=args.force,
    )
    control = pd.read_csv(REPOSITORY_ROOT / workflow["tree_predictions"])
    control = control.loc[
        control.candidate_number.astype(int).eq(int(workflow["control_candidate_number"]))
    ].copy()
    if control.empty or control.duplicated(IDENTITY_COLUMNS[:-1]).any():
        raise ValueError("PE_19 tree control is empty or has duplicate endpoints")
    tree = control[IDENTITY_COLUMNS + ["predicted_rul"]].rename(
        columns={"predicted_rul": "tree_prediction"}
    )
    order = ["outer_fold", "scenario", "uav_id", "cutoff"]
    tree = tree.sort_values(order).reset_index(drop=True)
    inner_tree = read_method_predictions(
        workflow["tree_inner_predictions"], method=str(workflow["tree_inner_method"])
    )[COMMON_KEYS + ["predicted_rul"]].rename(
        columns={"predicted_rul": "tree_prediction"}
    )
    inner_temporal = pd.read_csv(REPOSITORY_ROOT / workflow["temporal_inner_predictions"])
    methods = {"control": tree.tree_prediction.to_numpy(float)}
    provenance = []
    for family in [str(value) for value in workflow["challengers"]]:
        challenger = temporal.loc[temporal.model_family.astype(str).eq(family)]
        challenger = challenger[IDENTITY_COLUMNS + ["predicted_rul"]].rename(
            columns={"predicted_rul": "challenger_prediction"}
        )
        challenger = challenger.sort_values(order).reset_index(drop=True)
        if not challenger[IDENTITY_COLUMNS].equals(tree[IDENTITY_COLUMNS]):
            raise ValueError(f"PE_19 challenger {family} does not align with Run 7")
        rows = tree.copy()
        rows["challenger_prediction"] = challenger.challenger_prediction.to_numpy(float)
        inner_challenger = inner_temporal.loc[
            inner_temporal.model_family.astype(str).eq(family)
        ][COMMON_KEYS + ["predicted_rul"]].rename(
            columns={"predicted_rul": "challenger_prediction"}
        )
        selection_rows = inner_tree.merge(
            inner_challenger, on=COMMON_KEYS, validate="one_to_one"
        ).reset_index(drop=True)
        if len(selection_rows) != len(inner_tree):
            raise ValueError(f"PE_19 inner predictions for {family} do not align")
        prediction, records = cross_fit_blend(
            rows,
            [float(value) for value in workflow["challenger_weights"]],
            selection_rows,
        )
        name = f"tree_plus_{family}"
        methods[name] = prediction
        for record in records:
            record["method"] = name
        provenance.extend(records)
    pd.DataFrame(provenance).to_csv(reporting / "blend_provenance.csv", index=False)
    manifest = method_report(
        tree,
        methods,
        root=root,
        control="control",
        minimum_fold_wins=int(workflow["minimum_fold_wins"]),
        minimum_relative_rmse_improvement=float(
            workflow["minimum_relative_rmse_improvement"]
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
