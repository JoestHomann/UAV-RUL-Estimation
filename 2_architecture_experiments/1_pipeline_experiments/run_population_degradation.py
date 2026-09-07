"""Cross-fit population-to-UAV degradation features for PE_17."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from advanced_r2_utils import (
    COMMON_KEYS,
    REPOSITORY_ROOT,
    ScaledRidge,
    equal_uav_weights,
    load_workflow,
    method_report,
    read_method_predictions,
)


def reference_parameters(
    histories: dict[str, pd.DataFrame],
    uavs: set[str],
    channels: list[str],
    edge_window: int,
) -> dict[str, tuple[float, float, float]]:
    result = {}
    for channel in channels:
        healthy = np.concatenate([
            histories[uav][channel].to_numpy(float)[:edge_window] for uav in sorted(uavs)
        ])
        terminal_per_uav = np.array([
            np.median(histories[uav][channel].to_numpy(float)[-edge_window:])
            for uav in sorted(uavs)
        ])
        result[channel] = (
            float(np.median(healthy)),
            float(np.median(terminal_per_uav)),
            float(max(np.median(np.abs(terminal_per_uav - np.median(terminal_per_uav))), 1e-6)),
        )
    return result


def degradation_features(
    rows: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    reference: dict[str, tuple[float, float, float]],
    *,
    recent_window: int,
) -> pd.DataFrame:
    records = []
    for row in rows.itertuples(index=False):
        history = histories[str(row.uav_id)]
        prefix = history.loc[history.flight_cycle.astype(int).le(int(row.cutoff))]
        record = {}
        for channel, (healthy, terminal, terminal_mad) in reference.items():
            values = prefix[channel].to_numpy(float)
            recent = values[-min(recent_window, len(values)):]
            current = float(np.median(recent[-min(5, len(recent)):]))
            slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0]) if len(recent) > 1 else 0.0
            span = terminal - healthy
            safe_span = span if abs(span) > 1e-6 else np.copysign(1e-6, span or 1.0)
            direction = np.sign(safe_span)
            progress = float(np.clip((current - healthy) / safe_span, -2.0, 3.0))
            oriented_rate = direction * slope
            remaining_distance = direction * (terminal - current)
            time_to_terminal = (
                remaining_distance / oriented_rate if oriented_rate > 1e-8 else 500.0
            )
            prefix_name = f"population__{channel}"
            record[f"{prefix_name}__progress"] = progress
            record[f"{prefix_name}__oriented_rate"] = float(oriented_rate)
            record[f"{prefix_name}__time_to_terminal"] = float(np.clip(time_to_terminal, 0.0, 500.0))
            record[f"{prefix_name}__terminal_uncertainty"] = terminal_mad / abs(safe_span)
            record[f"{prefix_name}__weak_rate"] = float(oriented_rate <= 1e-8)
        records.append(record)
    result = pd.DataFrame.from_records(records, index=rows.index)
    if not np.isfinite(result.to_numpy(float)).all():
        raise ValueError("Population degradation features are non-finite")
    return result


def fit_candidate(
    rows: pd.DataFrame,
    raw: pd.DataFrame,
    workflow: dict,
    *,
    family: str,
) -> tuple[np.ndarray, list[dict]]:
    histories = {
        str(uav): history.sort_values("flight_cycle").reset_index(drop=True)
        for uav, history in raw.groupby("uav_id", sort=False)
    }
    predicted = np.full(len(rows), np.nan)
    provenance = []
    base_features = [
        "base_prediction", "cutoff", "uncertainty_std", "uncertainty_range",
        "family_disagreement", *[str(value) for value in workflow["existing_residual_features"]],
    ]
    channels = [str(value) for value in workflow["channels"]]
    for outer_fold, outer_rows in rows.groupby("outer_fold", sort=True):
        for inner_fold, held in outer_rows.groupby("inner_fold", sort=True):
            training = outer_rows.loc[outer_rows.inner_fold.ne(inner_fold)]
            training_uavs = set(training.uav_id.astype(str))
            validation_uavs = set(held.uav_id.astype(str))
            if training_uavs & validation_uavs:
                raise ValueError("PE_17 has UAV overlap")
            reference = reference_parameters(
                histories, training_uavs, channels, int(workflow["edge_window"])
            )
            training_population = degradation_features(
                training, histories, reference, recent_window=int(workflow["recent_window"])
            )
            held_population = degradation_features(
                held, histories, reference, recent_window=int(workflow["recent_window"])
            )
            train_x = pd.concat(
                [training[base_features].reset_index(drop=True), training_population.reset_index(drop=True)], axis=1
            )
            held_x = pd.concat(
                [held[base_features].reset_index(drop=True), held_population.reset_index(drop=True)], axis=1
            )
            target = training.base_prediction.to_numpy(float) - training.observed_rul.to_numpy(float)
            weights = equal_uav_weights(training)
            if family == "ridge":
                model = ScaledRidge(float(workflow["ridge_alpha"])).fit(train_x, target, weights)
            elif family == "hist_gradient_boosting":
                model = HistGradientBoostingRegressor(
                    max_iter=100,
                    max_leaf_nodes=int(workflow["maximum_leaf_nodes"]),
                    min_samples_leaf=20,
                    l2_regularization=float(workflow["l2_regularization"]),
                    learning_rate=0.05,
                    random_state=13,
                ).fit(train_x, target, sample_weight=weights)
            else:
                raise ValueError(f"Unknown PE_17 residual family {family!r}")
            correction = model.predict(held_x)
            predicted[held.index] = np.maximum(
                held.base_prediction.to_numpy(float) - correction, 0.0
            )
            provenance.append(
                {
                    "outer_fold": int(outer_fold), "inner_fold": int(inner_fold),
                    "model_family": family, "training_uavs": len(training_uavs),
                    "validation_uavs": len(validation_uavs), "uav_overlap": 0,
                    "terminal_reference_uavs": len(training_uavs),
                    "query_terminal_rows_used": False,
                }
            )
    if not np.isfinite(predicted).all():
        raise ValueError("PE_17 predictions are incomplete")
    return predicted, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_17")
    args = parser.parse_args()
    workflow, _, root = load_workflow(
        args.config, "population_degradation_workflows", args.workflow
    )
    base = read_method_predictions(
        workflow["source_predictions"], method=str(workflow["base_method"])
    )
    control = read_method_predictions(
        workflow["source_predictions"], method=str(workflow["control_method"])
    )[COMMON_KEYS + ["predicted_rul"]].rename(columns={"predicted_rul": "control_prediction"})
    rows = base.merge(control, on=COMMON_KEYS, validate="one_to_one").rename(
        columns={"predicted_rul": "base_prediction"}
    )
    raw = pd.read_csv(REPOSITORY_ROOT / workflow["train_csv"])
    methods = {"control": rows.control_prediction.to_numpy(float)}
    provenance = []
    for family in workflow["residual_families"]:
        prediction, records = fit_candidate(rows, raw, workflow, family=str(family))
        name = f"population_{family}"
        methods[name] = prediction
        for record in records:
            record["method"] = name
        provenance.extend(records)
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(provenance).to_csv(reporting / "fold_reference_provenance.csv", index=False)
    manifest = method_report(
        rows, methods, root=root, control="control",
        minimum_fold_wins=int(workflow["minimum_fold_wins"]),
        minimum_relative_rmse_improvement=float(workflow["minimum_relative_rmse_improvement"]),
        promotion_allowed=False,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
