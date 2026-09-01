"""Run cheap synthetic checks for PE_7-PE_9 and dual Phase 3 policies."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
PHASE_3_SETTINGS_DIR = (
    REPOSITORY_ROOT
    / "3_final_model_training_and_inference"
    / "1_winning_architecture_selection"
)
PHASE_1_DIR = REPOSITORY_ROOT / "1_dataset_construction"
if str(PHASE_3_SETTINGS_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE_3_SETTINGS_DIR))
if str(PHASE_1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE_1_DIR))

from align_oof_predictions import OOFAlignmentError, align_sources  # noqa: E402
from degradation_onset import build_personalized_targets  # noqa: E402
from experiment_config import read_experiment_config  # noqa: E402
from phase_1_config import load_phase_one_profile  # noqa: E402
from stack_oof_predictions import evaluate_stacks  # noqa: E402
from verify_phase_3_settings import Phase3Settings  # noqa: E402


def _oof_table(offset: float) -> pd.DataFrame:
    records = []
    validation_row = 0
    for outer_fold in range(2):
        for inner_fold in range(2):
            for uav_offset in range(2):
                uav = f"UAV_{outer_fold}_{inner_fold}_{uav_offset}"
                for scenario, cutoff in (("development_01", 20), ("development_02", 40)):
                    observed = float(120 - cutoff + 2 * uav_offset)
                    records.append(
                        {
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "validation_row": validation_row,
                            "uav_id": uav,
                            "scenario": scenario,
                            "cutoff": cutoff,
                            "observed_rul": observed,
                            "predicted_rul": observed + offset + 0.1 * outer_fold,
                        }
                    )
                    validation_row += 1
    return pd.DataFrame(records)


def main() -> None:
    temporary = SCRIPT_DIR / ".r2_contract_verification"
    temporary.mkdir(exist_ok=True)
    try:
        pe6_settings = SCRIPT_DIR / "experiments" / "PE_6" / "settings.toml"
        resolved_pe6 = read_experiment_config(pe6_settings)
        resolved_pe6_path = temporary / "resolved_pe6.json"
        resolved_pe6_path.write_text(
            json.dumps(resolved_pe6, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pe6_profile = load_phase_one_profile(
            "temporal_dense",
            resolved_pe6_path,
            "early_and_middle",
        )
        if pe6_profile.scenario_profile.name != "early_and_middle":
            raise RuntimeError("Resolved PE_6 scenario profile was not preserved")
        if {variant.name for variant in pe6_profile.prefix_variants} != {
            "current20",
            "dense_stride_1",
            "dense_stride_2",
        }:
            raise RuntimeError("Resolved PE_6 prefix variants are incomplete")

        tree = _oof_table(2.0)
        temporal = _oof_table(-1.0)
        tree.to_csv(temporary / "tree.csv", index=False)
        temporal.to_csv(temporary / "temporal.csv", index=False)
        sources = {
            "tree": {
                "path": (temporary / "tree.csv").relative_to(REPOSITORY_ROOT).as_posix()
            },
            "temporal": {
                "path": (temporary / "temporal.csv").relative_to(REPOSITORY_ROOT).as_posix()
            },
        }
        aligned = align_sources(sources)
        predictions, folds, provenance = evaluate_stacks(
            aligned,
            tree_source="tree",
            temporal_source="temporal",
            seed=13,
        )
        if predictions.empty or folds["method"].nunique() != 7:
            raise RuntimeError("Synthetic stacking methods are incomplete")
        if provenance.empty or not provenance["uav_overlap"].eq(0).all():
            raise RuntimeError("Synthetic stacking provenance failed")
        duplicate = pd.concat([tree, tree.iloc[[0]]], ignore_index=True)
        duplicate.to_csv(temporary / "duplicate.csv", index=False)
        try:
            align_sources({"tree": sources["tree"], "duplicate": {"path": (temporary / "duplicate.csv").relative_to(REPOSITORY_ROOT).as_posix()}})
        except OOFAlignmentError:
            pass
        else:
            raise RuntimeError("Duplicate OOF endpoint was accepted")

        features = pd.DataFrame(
            {
                "feature__trend": np.tile(np.arange(1, 7), 2),
                "feature__stable": np.ones(12),
            }
        )
        metadata = pd.DataFrame(
            {
                "uav_id": ["A"] * 6 + ["B"] * 6,
                "cutoff": list(range(10, 70, 10)) * 2,
            }
        )
        raw = pd.Series([70, 60, 50, 40, 30, 20] * 2, dtype=float)
        onset = build_personalized_targets(
            features,
            metadata,
            raw,
            detector="monotonic_health_index",
        )
        if len(onset.fitting_target) != len(raw) or (onset.fitting_target > raw).any():
            raise RuntimeError("Personalized fitting targets violate raw RUL")

        Phase3Settings.model_validate(
            {
                "settings_version": 1,
                "run_number": 99,
                "phase_2_run_number": 1,
                "selected_model_family": "xgboost",
                "final_search": {
                    "candidate_budget": 1,
                    "search_seed": 13,
                    "model_seed": 13,
                },
                "submission_policies": [
                    {
                        "name": "accuracy_q50",
                        "calibration": "conditional_quantile",
                        "safety_offset": 0.0,
                        "non_overprediction_coverage": 0.50,
                        "calibration_prediction_bin_edges": [0.0, 50.0, 125.0],
                        "calibration_minimum_bin_rows": 5,
                    },
                    {
                        "name": "conservative_q55",
                        "calibration": "conditional_quantile",
                        "safety_offset": 0.0,
                        "non_overprediction_coverage": 0.55,
                        "calibration_prediction_bin_edges": [0.0, 50.0, 125.0],
                        "calibration_minimum_bin_rows": 5,
                    },
                ],
                "canonical_submission_policy": "accuracy_q50",
            }
        )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    print("R2 experiment contract verification passed")


if __name__ == "__main__":
    main()
