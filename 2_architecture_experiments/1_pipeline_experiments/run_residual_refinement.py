"""Evaluate residual-head regularization and calibration coverage for PE_16."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from advanced_r2_utils import load_workflow, method_report, REPOSITORY_ROOT


PHASE2 = REPOSITORY_ROOT / "2_architecture_experiments" / "2_model_architecture_study"
PHASE1 = REPOSITORY_ROOT / "1_dataset_construction"
for directory in (
    PHASE1 / "5_prefix_feature_engineering",
    PHASE2 / "2_tabular_data_adapter",
    PHASE2 / "4_model_adapters",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_prefix_features import build_feature_table  # noqa: E402
from model_registry import ModelAdapterFactory  # noqa: E402
from no_op_training_monitor import NoOpTrainingMonitor  # noqa: E402
from tabular_data_adapter import TabularDataAdapter  # noqa: E402


GENERATED_DISTINCT_20 = "generated_distinct_20"


def distinct_calibration_manifest(
    raw: pd.DataFrame,
    *,
    endpoint_count: int,
    minimum_rul: float,
    maximum_rul: float,
) -> pd.DataFrame:
    """Choose deterministic, unique terminal-support endpoints for every UAV."""

    if endpoint_count < 2:
        raise ValueError("PE_16 distinct calibration requires at least two endpoints")
    records = []
    for uav_id, history in raw.groupby("uav_id", sort=True):
        eligible = history.loc[
            history["RUL"].astype(float).between(minimum_rul, maximum_rul)
        ].sort_values("flight_cycle")
        eligible = eligible.drop_duplicates("flight_cycle", keep="first")
        if len(eligible) < endpoint_count:
            raise ValueError(
                f"PE_16 needs {endpoint_count} eligible cutoffs for {uav_id}; "
                f"found {len(eligible)}"
            )
        positions = np.rint(
            np.linspace(0, len(eligible) - 1, endpoint_count)
        ).astype(int)
        selected = eligible.iloc[positions]
        if selected["flight_cycle"].astype(int).nunique() != endpoint_count:
            raise ValueError(f"PE_16 generated duplicate cutoffs for {uav_id}")
        for row in selected.itertuples(index=False):
            cutoff = int(row.flight_cycle)
            records.append(
                {
                    "sample_id": f"pe16_distinct20::{uav_id}::{cutoff:04d}",
                    "scenario": "pe16_distinct20",
                    "uav_id": str(uav_id),
                    "cutoff": cutoff,
                    "RUL": float(row.RUL),
                }
            )
    manifest = pd.DataFrame.from_records(records)
    counts = manifest.groupby("uav_id")["cutoff"].nunique()
    if counts.empty or not counts.eq(endpoint_count).all():
        raise ValueError("PE_16 distinct calibration coverage is incomplete")
    return manifest.sort_values(["uav_id", "cutoff"]).reset_index(drop=True)


def ensure_distinct_calibration_features(
    workflow: dict,
    *,
    feature_names: list[str],
    output_path: Path,
    force: bool,
) -> Path:
    endpoint_count = int(workflow["distinct_calibration_endpoint_count"])
    minimum_rul = float(workflow["calibration_minimum_rul"])
    maximum_rul = float(workflow["calibration_maximum_rul"])
    required = ["sample_id", "scenario", "uav_id", "cutoff", "RUL", *feature_names]
    if output_path.is_file() and not force:
        header = pd.read_csv(output_path, nrows=0)
        if set(required).issubset(header.columns):
            coverage = pd.read_csv(output_path, usecols=["uav_id", "cutoff", "RUL"])
            counts = coverage.groupby("uav_id")["cutoff"].nunique()
            support_ok = coverage["RUL"].between(minimum_rul, maximum_rul).all()
            if not counts.empty and counts.eq(endpoint_count).all() and support_ok:
                return output_path

    raw = pd.read_csv(REPOSITORY_ROOT / workflow["train_csv"])
    manifest = distinct_calibration_manifest(
        raw,
        endpoint_count=endpoint_count,
        minimum_rul=minimum_rul,
        maximum_rul=maximum_rul,
    )
    features = build_feature_table(
        raw,
        manifest,
        feature_profile=str(workflow["feature_profile"]),
    )[required]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False, compression="gzip")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_16")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workflow, _, root = load_workflow(
        args.config, "residual_refinement_workflows", args.workflow
    )
    reporting = root / "reporting"
    contracts = root / "contracts"
    reporting.mkdir(parents=True, exist_ok=True)
    contracts.mkdir(parents=True, exist_ok=True)
    source_contract = json.loads(
        (REPOSITORY_ROOT / workflow["source_contract"]).read_text(encoding="utf-8")
    )
    adapter = TabularDataAdapter(REPOSITORY_ROOT / workflow["tabular_manifest"])
    factory = ModelAdapterFactory(REPOSITORY_ROOT / workflow["specification"])
    feature_set = str(workflow["feature_set"])
    feature_names = adapter.feature_names(feature_set)
    distinct_calibration_path = ensure_distinct_calibration_features(
        workflow,
        feature_names=feature_names,
        output_path=root / "calibration" / "distinct_20_features.csv.gz",
        force=args.force,
    )
    candidates = {}
    for candidate in workflow["candidates"]:
        name = str(candidate["name"])
        contract = copy.deepcopy(source_contract)
        calibration_source = str(candidate["calibration_features_path"])
        contract["calibration_features_path"] = (
            distinct_calibration_path.relative_to(REPOSITORY_ROOT).as_posix()
            if calibration_source == GENERATED_DISTINCT_20
            else calibration_source
        )
        contract["calibration_weighting"] = str(candidate["calibration_weighting"])
        contract["correction_strength"] = float(candidate["correction_strength"])
        family = str(candidate["residual_family"])
        if family == "ridge":
            contract["residual_model"] = {
                "family": "ridge",
                "alpha": float(candidate["ridge_alpha"]),
            }
        else:
            contract["residual_model"] = {
                "family": "hist_gradient_boosting",
                "maximum_iterations": int(candidate["maximum_iterations"]),
                "maximum_leaf_nodes": int(candidate["maximum_leaf_nodes"]),
                "minimum_samples_leaf": int(candidate["minimum_samples_leaf"]),
                "l2_regularization": float(candidate["l2_regularization"]),
                "learning_rate": float(candidate["learning_rate"]),
                "seed": int(workflow["model_seed"]),
            }
        path = contracts / f"{name}.json"
        path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        candidates[name] = {"ensemble_contract_path": path.relative_to(REPOSITORY_ROOT).as_posix()}

    checkpoint = reporting / "fold_predictions.csv"
    if args.force:
        checkpoint.unlink(missing_ok=True)
    completed = pd.read_csv(checkpoint) if checkpoint.is_file() else pd.DataFrame()
    records = []
    for name, hyperparameters in candidates.items():
        for fold in adapter.outer_fold_labels():
            if (
                not completed.empty
                and (
                    completed.method.astype(str).eq(name)
                    & completed.outer_fold.astype(int).eq(fold)
                ).any()
            ):
                continue
            split = adapter.get_final_search_split(fold, feature_set)
            model = factory.create(
                "residual_corrected_tree_ensemble",
                hyperparameters,
                seed=int(workflow["model_seed"]),
                allow_disabled=True,
                training_monitor=NoOpTrainingMonitor(),
            )
            model.fit(split.training, None)
            prediction = model.predict(split.validation)
            for index, row in split.validation.metadata.iterrows():
                records.append(
                    {
                        "outer_fold": int(fold),
                        "inner_fold": int(fold),
                        "uav_id": str(row.uav_id),
                        "scenario": str(row.scenario),
                        "cutoff": float(row.cutoff),
                        "observed_rul": float(split.validation.target.iloc[index]),
                        "method": name,
                        "predicted_rul": float(prediction[index]),
                    }
                )
            completed = pd.concat([completed, pd.DataFrame(records)], ignore_index=True)
            completed.to_csv(checkpoint, index=False)
            records.clear()

    order = ["outer_fold", "scenario", "uav_id", "cutoff"]
    control = str(workflow["control"])
    identities = completed.loc[completed.method.eq(control)].sort_values(order).reset_index(drop=True)
    if identities.empty:
        raise ValueError(f"PE_16 has no {control!r} control predictions")
    methods = {}
    identity_columns = [*order, "observed_rul"]
    for name, group in completed.groupby("method"):
        aligned = group.sort_values(order).reset_index(drop=True)
        if not aligned[identity_columns].equals(identities[identity_columns]):
            raise ValueError(f"PE_16 method {name} does not align with the control")
        methods[str(name)] = aligned.predicted_rul.to_numpy(float)
    manifest = method_report(
        identities,
        methods,
        root=root,
        control=control,
        minimum_fold_wins=int(workflow["minimum_fold_wins"]),
        minimum_relative_rmse_improvement=float(
            workflow["minimum_relative_rmse_improvement"]
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
