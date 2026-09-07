"""Cross-fit causal sensor and forecast-history residual features for PE_15."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from advanced_r2_utils import (
    COMMON_KEYS,
    REPOSITORY_ROOT,
    causal_filter_features,
    cross_fit_residual,
    load_workflow,
    method_report,
    prediction_history_features,
    read_method_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_15")
    args = parser.parse_args()
    workflow, _, root = load_workflow(
        args.config, "causal_filtering_workflows", args.workflow
    )
    base = read_method_predictions(
        workflow["source_predictions"], method=str(workflow["base_method"])
    )
    control = read_method_predictions(
        workflow["source_predictions"], method=str(workflow["control_method"])
    )
    control = control[COMMON_KEYS + ["predicted_rul"]].rename(
        columns={"predicted_rul": "control_prediction"}
    )
    rows = base.merge(control, on=COMMON_KEYS, validate="one_to_one")
    rows = rows.rename(columns={"predicted_rul": "base_prediction"})
    raw = pd.read_csv(REPOSITORY_ROOT / workflow["train_csv"])
    history_input = rows.rename(columns={"base_prediction": "predicted_rul"})
    history = prediction_history_features(history_input)
    rows = pd.concat([rows, history], axis=1)
    common_features = [
        "base_prediction", "cutoff", "uncertainty_std", "uncertainty_range",
        "family_disagreement", *[str(value) for value in workflow["existing_residual_features"]],
    ]
    filter_columns: dict[str, list[str]] = {}
    for half_life in workflow["half_lives"]:
        transformed = causal_filter_features(
            raw,
            rows,
            channels=[str(value) for value in workflow["channels"]],
            half_life=float(half_life),
        )
        rows = pd.concat([rows, transformed], axis=1)
        filter_columns[f"sensor_h{float(half_life):g}"] = list(transformed.columns)
    history_columns = list(history.columns)
    methods = {"control": rows.control_prediction.to_numpy(float)}
    provenance = []
    recipes = []
    for name, columns in filter_columns.items():
        recipes.append((name, [*common_features, *columns]))
        recipes.append((f"{name}_history", [*common_features, *columns, *history_columns]))
    recipes.append(("prediction_history", [*common_features, *history_columns]))
    for name, features in recipes:
        prediction, records = cross_fit_residual(
            rows,
            base_column="base_prediction",
            feature_columns=features,
            model_family=str(workflow["residual_family"]),
            strength=float(workflow["correction_strength"]),
            ridge_alpha=float(workflow["ridge_alpha"]),
            maximum_leaf_nodes=int(workflow["maximum_leaf_nodes"]),
            l2_regularization=float(workflow["l2_regularization"]),
        )
        methods[name] = prediction
        for record in records:
            record["method"] = name
        provenance.extend(records)
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(provenance).to_csv(reporting / "cross_fit_provenance.csv", index=False)
    feature_manifest = {
        "channels": workflow["channels"],
        "half_lives": workflow["half_lives"],
        "forecast_history_is_strictly_past": True,
        "near_cap_forecasts_excluded_from_translation": True,
        "future_rows_used": False,
    }
    (reporting / "feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = method_report(
        rows,
        methods,
        root=root,
        control="control",
        minimum_fold_wins=int(workflow["minimum_fold_wins"]),
        minimum_relative_rmse_improvement=float(
            workflow["minimum_relative_rmse_improvement"]
        ),
        promotion_allowed=False,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
