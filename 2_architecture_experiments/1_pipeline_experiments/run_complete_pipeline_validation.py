"""Run fresh-scenario outer-UAV validation for frozen Run 6 and Run 7 systems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from advanced_r2_utils import load_workflow, method_report, subset_dataset, REPOSITORY_ROOT
from oof_experiment_utils import write_json


PIPELINE_DIR = Path(__file__).resolve().parent
PHASE1 = REPOSITORY_ROOT / "1_dataset_construction"
PHASE2 = REPOSITORY_ROOT / "2_architecture_experiments" / "2_model_architecture_study"
for directory in (
    PHASE1,
    PHASE1 / "3_test_like_validation_scenarios",
    PHASE1 / "5_prefix_feature_engineering",
    PHASE1 / "2_UAV_grouped_validation_folds",
    PHASE2 / "2_tabular_data_adapter",
    PHASE2 / "4_model_adapters",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_prefix_features import build_feature_table  # noqa: E402
from create_test_like_scenarios import make_validation_scenarios  # noqa: E402
from create_uav_grouped_folds import balanced_group_folds  # noqa: E402
from model_registry import ModelAdapterFactory  # noqa: E402
from no_op_training_monitor import NoOpTrainingMonitor  # noqa: E402
from tabular_data_adapter import TabularDataAdapter, TabularDataset  # noqa: E402


def selected_hyperparameters(path: Path) -> dict:
    rows = pd.read_csv(path)
    flags = rows["selected"]
    if is_bool_dtype(flags.dtype):
        mask = flags
    else:
        mask = flags.astype(str).str.strip().str.lower().map(
            {"true": True, "false": False}
        )
        if mask.isna().any():
            raise ValueError(f"Invalid selected flags in {path}")
    selected = rows.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected one selected candidate in {path}")
    return json.loads(str(selected.iloc[0].hyperparameters_json))


def scenario_dataset(table: pd.DataFrame, feature_names: list[str]) -> TabularDataset:
    metadata = table[[
        "sample_id", "scenario", "outer_fold", "uav_id", "cutoff",
        "terminal_lifetime", "lifetime_quantile",
    ]].copy()
    return TabularDataset(
        features=table[feature_names].copy(),
        metadata=metadata,
        target=table["RUL"].astype(float).copy(),
        sample_weights=None,
        fitting_target=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_14")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workflow, _, root = load_workflow(
        args.config, "validation_audit_workflows", args.workflow
    )
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    source_manifest = (REPOSITORY_ROOT / workflow["tabular_manifest"]).resolve()
    adapter = TabularDataAdapter(source_manifest)
    feature_set = str(workflow["feature_set"])
    feature_names = adapter.feature_names(feature_set)
    raw = pd.read_csv(REPOSITORY_ROOT / workflow["train_csv"])
    history_summary = pd.read_csv(REPOSITORY_ROOT / workflow["outer_folds"])[
        ["uav_id", "row_count", "final_cycle", "terminal_lifetime"]
    ]
    test_lengths = pd.read_csv(REPOSITORY_ROOT / workflow["test_cutoffs"])[
        "final_cycle"
    ].to_numpy(int)

    suites = []
    split_tables = {}
    for split_seed in workflow["split_seeds"]:
        outer = balanced_group_folds(
            history_summary,
            n_folds=int(workflow["outer_fold_count"]),
            seed=int(split_seed),
            fold_column="outer_fold",
        )
        split_tables[int(split_seed)] = outer
        for profile in workflow["profiles"]:
            name = str(profile["name"])
            maximum = profile.get("maximum_rul")
            scenarios = make_validation_scenarios(
                raw,
                outer,
                test_lengths,
                scenario_count=int(profile["scenario_count"]),
                seed=int(split_seed) + int(profile["scenario_seed_offset"]),
                scenario_prefix=f"{name}_{split_seed}",
                assignment=str(profile["assignment"]),
                minimum_rul=profile.get("minimum_rul"),
                maximum_rul=maximum,
            )
            scenarios["profile"] = name
            scenarios["split_seed"] = int(split_seed)
            suites.append(scenarios)
    scenario_manifest = pd.concat(suites, ignore_index=True)
    scenario_manifest.to_csv(reporting / "fresh_scenarios.csv", index=False)
    feature_table = build_feature_table(
        raw,
        scenario_manifest.drop(columns=["profile", "split_seed"]),
        feature_profile="extended",
    )
    feature_table["profile"] = scenario_manifest["profile"].to_numpy()
    feature_table["split_seed"] = scenario_manifest["split_seed"].to_numpy()
    evaluation = scenario_dataset(feature_table, feature_names)
    full_training = adapter.load_training(feature_set)

    systems = {}
    for name, system in workflow["systems"].items():
        specification = REPOSITORY_ROOT / system["specification"]
        candidates = REPOSITORY_ROOT / system["selected_candidates"]
        systems[name] = (
            ModelAdapterFactory(specification),
            str(system["family"]),
            {
                **selected_hyperparameters(candidates),
                **(
                    {
                        "calibration_features_path": str(
                            system["calibration_features_path"]
                        ),
                        "calibration_internal_folds": int(
                            system["calibration_internal_folds"]
                        ),
                        "calibration_degree": int(system["calibration_degree"]),
                        "calibration_ridge_alpha": float(
                            system["calibration_ridge_alpha"]
                        ),
                    }
                    if "calibration_features_path" in system
                    else {}
                ),
            },
        )

    checkpoint_path = reporting / "fold_predictions.csv"
    if args.force:
        checkpoint_path.unlink(missing_ok=True)
    completed = pd.read_csv(checkpoint_path) if checkpoint_path.is_file() else pd.DataFrame()
    records = []
    for split_number, split_seed in enumerate(workflow["split_seeds"]):
        outer = split_tables[int(split_seed)]
        for profile in sorted(feature_table["profile"].astype(str).unique()):
            # Profile and split membership stay outside TabularDataset so the
            # existing adapter metadata contract remains unchanged.
            profile_mask = (
                feature_table["profile"].astype(str).eq(profile)
                & feature_table["split_seed"].astype(int).eq(int(split_seed))
            ).to_numpy()
            if not profile_mask.any():
                continue
            for system_name, (factory, family, hyperparameters) in systems.items():
                for split_outer_fold in range(int(workflow["outer_fold_count"])):
                    report_fold = split_number * int(workflow["outer_fold_count"]) + split_outer_fold
                    existing = (
                        not completed.empty
                        and (
                            completed["profile"].astype(str).eq(profile)
                            & completed["system"].astype(str).eq(system_name)
                            & completed["split_seed"].astype(int).eq(int(split_seed))
                            & completed["split_outer_fold"].astype(int).eq(split_outer_fold)
                        ).any()
                    )
                    if existing:
                        continue
                    held_uavs = set(
                        outer.loc[outer.outer_fold.eq(split_outer_fold), "uav_id"].astype(str)
                    )
                    training = subset_dataset(
                        full_training,
                        ~full_training.metadata.uav_id.astype(str).isin(held_uavs).to_numpy(),
                    )
                    mask = profile_mask & evaluation.metadata["outer_fold"].astype(int).eq(
                        split_outer_fold
                    ).to_numpy()
                    validation = subset_dataset(evaluation, mask)
                    if set(training.metadata.uav_id.astype(str)) & set(validation.metadata.uav_id.astype(str)):
                        raise ValueError("PE_14 outer evaluation has UAV overlap")
                    model = factory.create(
                        family,
                        hyperparameters,
                        seed=int(workflow["model_seed"]),
                        allow_disabled=True,
                        training_monitor=NoOpTrainingMonitor(),
                    )
                    model.fit(training, None)
                    prediction = model.predict(validation)
                    for index, row in validation.metadata.iterrows():
                        records.append(
                            {
                                "profile": profile,
                                "system": system_name,
                                "outer_fold": int(report_fold),
                                "split_seed": int(split_seed),
                                "split_outer_fold": int(split_outer_fold),
                                "uav_id": str(row.uav_id),
                                "scenario": str(row.scenario),
                                "cutoff": float(row.cutoff),
                                "observed_rul": float(validation.target.iloc[index]),
                                "predicted_rul": float(prediction[index]),
                            }
                        )
                    combined = pd.concat([completed, pd.DataFrame(records)], ignore_index=True)
                    combined.to_csv(checkpoint_path, index=False)
                    completed = combined
                    records.clear()

    manifests = {}
    for profile, rows in completed.groupby("profile"):
        order = ["outer_fold", "scenario", "uav_id", "cutoff", "observed_rul"]
        identities = rows.loc[rows.system.eq("run_6")].sort_values(order).reset_index(drop=True)
        if identities.empty:
            raise ValueError(f"PE_14 profile {profile} has no Run 6 control")
        methods = {}
        for name, group in rows.groupby("system"):
            aligned = group.sort_values(order).reset_index(drop=True)
            if not aligned[order].equals(identities[order]):
                raise ValueError(
                    f"PE_14 profile {profile} system {name} does not align with Run 6"
                )
            methods[str(name)] = aligned["predicted_rul"].to_numpy(float)
        profile_root = root / f"profile_{profile}"
        manifests[str(profile)] = method_report(
            identities,
            methods,
            root=profile_root,
            control="run_6",
            minimum_fold_wins=int(workflow["minimum_fold_wins"]),
            minimum_relative_rmse_improvement=float(
                workflow["minimum_relative_rmse_improvement"]
            ),
        )
    audit = {
        "status": "complete",
        "profiles": manifests,
        "scenario_rows": len(scenario_manifest),
        "scenario_uavs": scenario_manifest.uav_id.nunique(),
        "outer_split_seeds": [int(value) for value in workflow["split_seeds"]],
        "target_dependent_nominal_support": any(
            profile.get("maximum_rul") is not None for profile in workflow["profiles"]
        ),
        "run_6_calibrator_fit_on_inner_oof_training_uavs": True,
        "run_7_residual_head_fit_on_inner_oof_training_uavs": True,
        "frozen_hyperparameters_reused": True,
        "estimates_training_or_selection_uncertainty": False,
        "uses_test_labels": False,
        "uses_kaggle_feedback_for_selection": False,
    }
    write_json(reporting / "audit_manifest.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
