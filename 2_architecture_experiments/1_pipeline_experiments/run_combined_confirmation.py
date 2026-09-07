"""Combine independently confirmed forecast history and TabPFN predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_r2_utils import COMMON_KEYS, REPOSITORY_ROOT, load_workflow, method_report
from oof_experiment_utils import regression_metrics
from run_tabular_prior import cross_fit_outer_blend


def _repository_file(value: str) -> Path:
    path = (REPOSITORY_ROOT / value).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Source path escapes the repository: {value}") from error
    if not path.is_file():
        raise ValueError(f"Required source is missing: {path}")
    return path


def _require_promoted(path_value: str, expected_winner: str) -> dict:
    path = _repository_file(path_value)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read prerequisite manifest {path}: {error}") from error
    if manifest.get("promoted") is not True or manifest.get("winner") != expected_winner:
        raise ValueError(
            f"Prerequisite {path} did not promote {expected_winner!r}; "
            f"observed promoted={manifest.get('promoted')!r}, "
            f"winner={manifest.get('winner')!r}"
        )
    return manifest


def _method_rows(path_value: str, method: str) -> pd.DataFrame:
    path = _repository_file(path_value)
    table = pd.read_csv(path)
    required = {
        "evaluation_level",
        *COMMON_KEYS,
        "method",
        "predicted_rul",
    }
    missing = sorted(required - set(table.columns))
    rows = table.loc[table.method.astype(str).eq(method)].copy()
    if missing or rows.empty:
        raise ValueError(f"Prediction source {path} is empty or missing {missing}")
    keys = ["evaluation_level", *COMMON_KEYS]
    if rows.duplicated(keys).any():
        raise ValueError(f"Prediction source {path} contains duplicate {method} rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_22")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workflow, _, root = load_workflow(
        args.config, "combined_confirmation_workflows", args.workflow
    )
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    contract_path = reporting / "final_candidate_contract.json"
    if args.force:
        for path in (reporting / "winner_manifest.json", contract_path):
            path.unlink(missing_ok=True)

    prerequisites = {
        "history": _require_promoted(
            str(workflow["history_winner_manifest"]),
            str(workflow["history_required_winner"]),
        ),
        "tabpfn_original": _require_promoted(
            str(workflow["tabpfn_winner_manifest"]),
            str(workflow["tabpfn_required_winner"]),
        ),
        "tabpfn_multiseed": _require_promoted(
            str(workflow["tabpfn_confirmation_manifest"]),
            str(workflow["tabpfn_required_winner"]),
        ),
    }
    history = _method_rows(
        str(workflow["history_predictions"]),
        str(workflow["history_method"]),
    )
    tabpfn = _method_rows(
        str(workflow["tabpfn_predictions"]),
        str(workflow["tabpfn_method"]),
    )
    keys = ["evaluation_level", *COMMON_KEYS]
    paired = history[keys + ["predicted_rul"]].rename(
        columns={"predicted_rul": "control_prediction"}
    ).merge(
        tabpfn[keys + ["predicted_rul"]].rename(
            columns={"predicted_rul": "challenger_prediction"}
        ),
        on=keys,
        validate="one_to_one",
    )
    if len(paired) != len(history) or len(paired) != len(tabpfn):
        raise ValueError("PE_20 history and PE_18 TabPFN predictions do not align")
    outer = paired.loc[paired.evaluation_level.eq("outer")].copy()
    inner = paired.loc[paired.evaluation_level.eq("inner")].copy()
    if outer.empty or inner.empty:
        raise ValueError("Combined confirmation requires both outer and inner predictions")
    held = outer.drop(columns="evaluation_level").reset_index(drop=True)
    selection = inner.drop(columns="evaluation_level").reset_index(drop=True)
    prediction, provenance = cross_fit_outer_blend(
        held,
        selection,
        weights=[float(value) for value in workflow["challenger_weights"]],
    )
    for row in provenance:
        row["method"] = "history_plus_tabpfn_full"
    pd.DataFrame(provenance).to_csv(reporting / "blend_provenance.csv", index=False)

    identities = held.copy()
    methods = {
        "prediction_history": held.control_prediction.to_numpy(float),
        "history_plus_tabpfn_full": prediction,
    }
    manifest = method_report(
        identities,
        methods,
        root=root,
        control="prediction_history",
        minimum_fold_wins=int(workflow["minimum_fold_wins"]),
        minimum_relative_rmse_improvement=float(
            workflow["minimum_relative_rmse_improvement"]
        ),
    )

    weights = [float(value) for value in workflow["challenger_weights"]]
    scored = []
    for weight in weights:
        estimate = (
            (1.0 - weight) * held.control_prediction.to_numpy(float)
            + weight * held.challenger_prediction.to_numpy(float)
        )
        scored.append((regression_metrics(held.observed_rul, estimate)["rmse"], weight))
    selected_rmse, final_weight = min(scored, key=lambda item: (item[0], item[1]))
    manifest.update(
        {
            "prerequisites_promoted": True,
            "final_oof_selected_tabpfn_weight": float(final_weight),
            "final_oof_selected_rmse": float(selected_rmse),
            "uses_test_labels": False,
        }
    )
    (reporting / "winner_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if manifest.get("promoted") is True and manifest.get("winner") == "history_plus_tabpfn_full":
        candidate = {
            "contract_version": 1,
            "status": "frozen",
            "method": "history_plus_tabpfn_full",
            "feature_set": str(workflow["feature_set"]),
            "target_cap": float(workflow["target_cap"]),
            "model_seed": int(workflow["model_seed"]),
            "history_lags": [int(value) for value in workflow["history_lags"]],
            "feature_profile": str(workflow["feature_profile"]),
            "near_cap_threshold": float(workflow["near_cap_threshold"]),
            "tabpfn_weight": float(final_weight),
            "tabpfn_weight_grid": weights,
            "weight_selection": "pooled outer OOF predictions from training UAVs only",
            "source_contract": str(workflow["source_contract"]),
            "specification": str(workflow["specification"]),
            "tabular_manifest": str(workflow["tabular_manifest"]),
            "train_csv": str(workflow["train_csv"]),
            "test_csv": str(workflow["test_csv"]),
            "tabpfn": dict(workflow["tabpfn"]),
            "prerequisite_manifests": {
                "history": str(workflow["history_winner_manifest"]),
                "tabpfn_original": str(workflow["tabpfn_winner_manifest"]),
                "tabpfn_multiseed": str(workflow["tabpfn_confirmation_manifest"]),
            },
            "test_data_loaded": False,
            "test_labels_loaded": False,
        }
        contract_path.write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        contract_path.unlink(missing_ok=True)
    print(json.dumps({**manifest, "prerequisites": prerequisites}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

