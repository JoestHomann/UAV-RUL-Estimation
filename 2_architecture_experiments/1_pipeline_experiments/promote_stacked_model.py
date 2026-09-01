"""Freeze a PE_7 development winner into a reviewable promotion contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from experiment_config import read_experiment_config
from experiment_paths import repository_path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class StackPromotionError(ValueError):
    """Explain a failed promotion or leakage-provenance check."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_7")
    parser.add_argument("--stack-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_experiment_config(args.config.resolve())
    workflow = config.get("stacking_workflows", {}).get(args.workflow)
    if not isinstance(workflow, dict):
        raise StackPromotionError(f"Unknown stacking workflow {args.workflow!r}")
    stack_dir = args.stack_dir.resolve()
    manifest = json.loads((stack_dir / "stacking_manifest.json").read_text(encoding="utf-8"))
    winner = manifest.get("winner")
    if not isinstance(winner, str):
        print("PE_7 has no gate-passing stack; no promotion contract was written")
        return
    provenance = pd.read_csv(stack_dir / "meta_fold_provenance.csv")
    if not provenance.empty and (
        "uav_overlap" not in provenance or not provenance["uav_overlap"].eq(0).all()
    ):
        raise StackPromotionError("Meta-model provenance does not prove UAV isolation")
    frozen = workflow.get("production_contract")
    if not isinstance(frozen, dict):
        raise StackPromotionError("PE_7 production_contract is missing")
    tree_configuration = json.loads(
        repository_path(
            REPOSITORY_ROOT,
            str(frozen.get("tree_selected_configuration")),
        ).read_text(encoding="utf-8")
    )
    temporal_manifest = json.loads(
        repository_path(
            REPOSITORY_ROOT,
            str(frozen.get("temporal_winner_manifest")),
        ).read_text(encoding="utf-8")
    )
    temporal_family = temporal_manifest.get("winner")
    if not isinstance(temporal_family, str) or temporal_manifest.get(
        "seed_stability_passed"
    ) is not True:
        raise StackPromotionError("Temporal component is not seed-confirmed")
    temporal_configurations = pd.read_csv(
        repository_path(
            REPOSITORY_ROOT,
            str(frozen.get("temporal_selected_configurations")),
        )
    )
    temporal_configurations = temporal_configurations.loc[
        temporal_configurations["model_family"].eq(temporal_family)
    ].copy()
    if temporal_configurations.empty:
        raise StackPromotionError("Temporal winner has no selected configurations")
    grouped = (
        temporal_configurations.groupby("configuration_json", as_index=False)
        .agg(
            outer_fold_count=("outer_fold", "nunique"),
            mean_inner_rmse=("mean_inner_rmse", "mean"),
        )
        .sort_values(
            ["outer_fold_count", "mean_inner_rmse", "configuration_json"],
            ascending=[False, True, True],
        )
    )
    frozen_configuration_json = str(grouped.iloc[0]["configuration_json"])
    temporal_rows = temporal_configurations.loc[
        temporal_configurations["configuration_json"].eq(
            frozen_configuration_json
        )
    ]
    temporal_configuration = json.loads(frozen_configuration_json)
    iteration_values = pd.to_numeric(
        temporal_rows["outer_retraining_iterations"], errors="coerce"
    ).dropna()
    if iteration_values.empty:
        raise StackPromotionError("Temporal winner has no fixed training duration")
    temporal_specification = json.loads(
        repository_path(
            REPOSITORY_ROOT,
            str(frozen.get("temporal_phase_2_specification")),
        ).read_text(encoding="utf-8")
    )
    tree_component = {
        "family": str(tree_configuration["model_family"]),
        "feature_set": str(tree_configuration["feature_set"]),
        "hyperparameters": tree_configuration["hyperparameters"],
        "seed": int(tree_configuration["model_seed"]),
        "training_iterations": tree_configuration.get(
            "final_training_iterations"
        ),
    }
    temporal_component = {
        "family": temporal_family,
        "lookback": int(temporal_configuration["lookback"]),
        "hyperparameters": temporal_configuration["hyperparameters"],
        "seed": 13,
        "training_iterations": int(round(float(iteration_values.median()))),
        "neural_training": temporal_specification["settings"]["neural_training"],
        "configuration_rule": (
            "most outer-fold selections, then lowest mean inner RMSE, "
            "then canonical configuration JSON"
        ),
    }
    predictions = pd.read_csv(stack_dir / "stack_oof_predictions.csv.gz")
    component_columns = ["prediction__tree", "prediction__temporal"]
    if not set(component_columns).issubset(predictions):
        raise StackPromotionError("Stack predictions have no production components")
    if winner == "nonnegative_ridge":
        meta_model: Any = Ridge(alpha=1.0, positive=True).fit(
            predictions[component_columns], predictions["observed_rul"]
        )
    elif winner == "shallow_xgboost":
        meta_model = XGBRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.03,
            min_child_weight=5.0,
            subsample=0.8,
            reg_lambda=5.0,
            objective="reg:squarederror",
            tree_method="hist",
            device="cpu",
            random_state=int(workflow.get("seed", 13)),
            n_jobs=1,
        ).fit(predictions[component_columns], predictions["observed_rul"])
    elif winner.startswith("blend_"):
        temporal_weight = int(winner.rsplit("_", 1)[1]) / 100.0
        meta_model = {
            "method": "convex_blend",
            "tree_weight": 1.0 - temporal_weight,
            "temporal_weight": temporal_weight,
        }
    else:
        raise StackPromotionError(f"Method {winner!r} is not deployable")
    meta_path = output.resolve().parent / "frozen_oof_meta_model.joblib"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta_model, meta_path)
    contract: dict[str, Any] = {
        "contract_version": 1,
        "family": "heterogeneous_oof_stack",
        "representation": "heterogeneous",
        "method": winner,
        "uses_locked_evaluation": False,
        "meta_model_fitted_from_oof_only": True,
        "alignment_keys": workflow.get("alignment_keys"),
        "components": {
            "tree": tree_component,
            "temporal": temporal_component,
        },
        "target_policy": frozen.get("target_policy"),
        "prediction_policy": frozen.get("prediction_policy"),
        "stacking_artifacts": manifest.get("artifacts"),
        "frozen_meta_model": meta_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "adapter_hyperparameters": {
            "tree_component": tree_component,
            "temporal_component": temporal_component,
            "meta_model_path": meta_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "alignment_keys": workflow.get("alignment_keys"),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Promoted PE_7 {winner} to {output}")


if __name__ == "__main__":
    main()
