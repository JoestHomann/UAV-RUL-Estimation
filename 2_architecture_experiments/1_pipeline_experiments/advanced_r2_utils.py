"""Shared contracts for PE_14 through PE_19 development experiments."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiment_config import read_experiment_config
from experiment_paths import gallery_directory, repository_path, run_directory
from oof_experiment_utils import regression_metrics, write_json


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
COMMON_KEYS = [
    "outer_fold", "inner_fold", "uav_id", "scenario", "cutoff", "observed_rul"
]


def load_workflow(
    config_path: Path,
    table: str,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = read_experiment_config(config_path.resolve())
    workflow = config.get(table, {}).get(name)
    definition = config.get("run_definitions", {}).get(name)
    if not isinstance(workflow, dict) or not isinstance(definition, dict):
        raise ValueError(f"Missing {table}.{name} or run_definitions.{name}")
    root = run_directory(EXPERIMENTS_DIR, name, definition)
    return workflow, definition, root


def read_method_predictions(
    path_value: str,
    *,
    method: str,
) -> pd.DataFrame:
    path = repository_path(REPOSITORY_ROOT, path_value)
    table = pd.read_csv(path)
    if "method" in table.columns:
        table = table.loc[table["method"].astype(str).eq(method)].copy()
    missing = sorted(set(COMMON_KEYS + ["predicted_rul"]) - set(table.columns))
    if missing or table.empty:
        raise ValueError(f"Prediction source {path} is empty or missing {missing}")
    if table.duplicated(COMMON_KEYS).any():
        raise ValueError(f"Prediction source {path} contains duplicate keys")
    numeric = table[["observed_rul", "predicted_rul"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"Prediction source {path} contains non-finite values")
    return table.reset_index(drop=True)


def subset_dataset(dataset: Any, mask: Any) -> Any:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (len(dataset),):
        raise ValueError("Dataset subset mask has the wrong length")

    def take(value: pd.Series | None) -> pd.Series | None:
        return None if value is None else value.loc[selected].reset_index(drop=True)

    return replace(
        dataset,
        features=dataset.features.loc[selected].reset_index(drop=True),
        metadata=dataset.metadata.loc[selected].reset_index(drop=True),
        target=take(dataset.target),
        sample_weights=take(dataset.sample_weights),
        fitting_target=take(dataset.fitting_target),
    )


def equal_uav_weights(rows: pd.DataFrame) -> np.ndarray:
    counts = rows["uav_id"].astype(str).value_counts()
    weights = rows["uav_id"].astype(str).map(lambda value: 1.0 / counts[value]).to_numpy(float)
    return weights * len(weights) / weights.sum()


class ScaledRidge:
    def __init__(self, alpha: float) -> None:
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=float(alpha))

    def fit(self, x: pd.DataFrame, y: np.ndarray, weights: np.ndarray) -> ScaledRidge:
        self.columns = list(x.columns)
        values = self.scaler.fit_transform(x.to_numpy(float))
        self.model.fit(values, y, sample_weight=weights)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if list(x.columns) != self.columns:
            raise ValueError("Residual feature order changed")
        return np.asarray(self.model.predict(self.scaler.transform(x.to_numpy(float))), float)


def cross_fit_residual(
    rows: pd.DataFrame,
    *,
    base_column: str,
    feature_columns: list[str],
    model_family: str,
    strength: float,
    ridge_alpha: float = 10.0,
    maximum_leaf_nodes: int = 7,
    l2_regularization: float = 10.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not 0.0 <= strength <= 1.5:
        raise ValueError("Correction strength must be in [0, 1.5]")
    missing = sorted(set(feature_columns + [base_column]) - set(rows.columns))
    if missing:
        raise ValueError(f"Residual data is missing {missing}")
    prediction = np.full(len(rows), np.nan, dtype=float)
    provenance: list[dict[str, Any]] = []
    for outer_fold, outer_rows in rows.groupby("outer_fold", sort=True):
        for inner_fold, held in outer_rows.groupby("inner_fold", sort=True):
            training = outer_rows.loc[outer_rows["inner_fold"].ne(inner_fold)]
            overlap = set(training.uav_id.astype(str)) & set(held.uav_id.astype(str))
            if overlap:
                raise ValueError("Residual cross-fitting has UAV overlap")
            target = (
                training[base_column].to_numpy(float)
                - training["observed_rul"].to_numpy(float)
            )
            weights = equal_uav_weights(training)
            if model_family == "ridge":
                model: Any = ScaledRidge(ridge_alpha).fit(
                    training[feature_columns], target, weights
                )
            elif model_family == "hist_gradient_boosting":
                model = HistGradientBoostingRegressor(
                    max_iter=100,
                    max_leaf_nodes=int(maximum_leaf_nodes),
                    min_samples_leaf=20,
                    l2_regularization=float(l2_regularization),
                    learning_rate=0.05,
                    random_state=13,
                )
                model.fit(training[feature_columns], target, sample_weight=weights)
            else:
                raise ValueError(f"Unknown residual family {model_family!r}")
            correction = np.asarray(model.predict(held[feature_columns]), float)
            prediction[held.index] = np.maximum(
                held[base_column].to_numpy(float) - strength * correction, 0.0
            )
            provenance.append(
                {
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "training_rows": len(training),
                    "training_uavs": training.uav_id.nunique(),
                    "validation_rows": len(held),
                    "validation_uavs": held.uav_id.nunique(),
                    "uav_overlap": 0,
                    "model_family": model_family,
                    "strength": float(strength),
                    "features": json.dumps(feature_columns),
                }
            )
    if not np.isfinite(prediction).all():
        raise ValueError("Cross-fitted residual predictions are incomplete")
    return prediction, provenance


def causal_filter_features(
    raw: pd.DataFrame,
    endpoints: pd.DataFrame,
    *,
    channels: list[str],
    half_life: float,
) -> pd.DataFrame:
    if half_life <= 0.0:
        raise ValueError("Half-life must be positive")
    alpha = 1.0 - np.exp(np.log(0.5) / float(half_life))
    histories = {
        str(uav): history.sort_values("flight_cycle").reset_index(drop=True)
        for uav, history in raw.groupby("uav_id", sort=False)
    }
    records = []
    cache: dict[tuple[str, int], dict[str, float]] = {}
    for row in endpoints.itertuples(index=False):
        uav = str(row.uav_id)
        cutoff = int(row.cutoff)
        key = (uav, cutoff)
        if key not in cache:
            prefix = histories[uav].loc[
                histories[uav]["flight_cycle"].astype(int).le(cutoff)
            ]
            if prefix.empty or int(prefix.flight_cycle.iloc[-1]) != cutoff:
                raise ValueError(f"No raw prefix for {uav} at {cutoff}")
            features: dict[str, float] = {}
            for channel in channels:
                values = prefix[channel].to_numpy(float)
                filtered = np.empty_like(values)
                filtered[0] = values[0]
                for index in range(1, len(values)):
                    filtered[index] = alpha * values[index] + (1.0 - alpha) * filtered[index - 1]
                recent = filtered[-min(20, len(filtered)):]
                x = np.arange(len(recent), dtype=float)
                slope = float(np.polyfit(x, recent, 1)[0]) if len(recent) > 1 else 0.0
                prefix_name = f"filter_h{half_life:g}__{channel}"
                features[f"{prefix_name}__level"] = float(filtered[-1])
                features[f"{prefix_name}__slope20"] = slope
                features[f"{prefix_name}__innovation"] = float(values[-1] - filtered[-1])
            cache[key] = features
        records.append(cache[key])
    result = pd.DataFrame.from_records(records, index=endpoints.index)
    if not np.isfinite(result.to_numpy(float)).all():
        raise ValueError("Causal filtering produced non-finite features")
    return result


def prediction_history_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize only strictly earlier OOF forecasts from the same fitted base model."""
    result = pd.DataFrame(index=rows.index)
    for name in (
        "history_available", "history_count", "history_last_lag",
        "history_failure_cycle_delta", "history_failure_cycle_spread",
        "history_failure_cycle_slope", "history_near_cap_fraction",
    ):
        result[name] = 0.0
    for _, group in rows.groupby(["outer_fold", "inner_fold", "uav_id"], sort=False):
        ordered = group.sort_values(["cutoff", "scenario"])
        for index, row in ordered.iterrows():
            earlier = ordered.loc[ordered["cutoff"].astype(float).lt(float(row.cutoff))]
            if earlier.empty:
                continue
            # Several validation scenarios can produce the same endpoint. Treat
            # those as repeated forecasts of one historical time, rather than
            # independent x values in the trend calculation.
            earlier = earlier.groupby("cutoff", as_index=False).agg(
                predicted_rul=("predicted_rul", "mean")
            ).sort_values("cutoff")
            failure = earlier["cutoff"].to_numpy(float) + earlier["predicted_rul"].to_numpy(float)
            lags = float(row.cutoff) - earlier["cutoff"].to_numpy(float)
            usable = earlier["predicted_rul"].to_numpy(float) < 124.5
            selected_failure = failure[usable]
            selected_lags = lags[usable]
            result.loc[index, "history_available"] = float(bool(usable.any()))
            result.loc[index, "history_count"] = float(usable.sum())
            result.loc[index, "history_last_lag"] = float(lags.min())
            result.loc[index, "history_near_cap_fraction"] = float((~usable).mean())
            if usable.any():
                current_failure = float(row.cutoff) + float(row.predicted_rul)
                result.loc[index, "history_failure_cycle_delta"] = current_failure - selected_failure[-1]
                result.loc[index, "history_failure_cycle_spread"] = float(np.std(selected_failure))
                if len(selected_failure) > 1:
                    result.loc[index, "history_failure_cycle_slope"] = float(
                        np.polyfit(-selected_lags, selected_failure, 1)[0]
                    )
    return result


def method_report(
    rows: pd.DataFrame,
    methods: dict[str, np.ndarray],
    *,
    root: Path,
    control: str,
    minimum_fold_wins: int,
    minimum_relative_rmse_improvement: float,
    uses_locked_evaluation: bool = False,
    promotion_allowed: bool = True,
    method_improvement_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    reporting = root / "reporting"
    reporting.mkdir(parents=True, exist_ok=True)
    prediction_frames = []
    fold_records = []
    identity = [column for column in COMMON_KEYS if column in rows]
    for name, values in methods.items():
        values = np.asarray(values, dtype=float)
        if values.shape != (len(rows),) or not np.isfinite(values).all():
            raise ValueError(f"Method {name!r} has invalid predictions")
        frame = rows[identity].copy()
        frame["method"] = name
        frame["predicted_rul"] = values
        prediction_frames.append(frame)
        for fold, fold_rows in frame.groupby("outer_fold", sort=True):
            fold_records.append(
                {
                    "method": name,
                    "outer_fold": int(fold),
                    **regression_metrics(
                        fold_rows["observed_rul"], fold_rows["predicted_rul"]
                    ),
                }
            )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    folds = pd.DataFrame(fold_records)
    summary = folds.groupby("method", as_index=False).agg(
        folds=("outer_fold", "nunique"),
        mean_r2=("r2", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_mae=("mae", "mean"),
        mean_bias=("bias", "mean"),
        mean_rms_overprediction=("rms_overprediction", "mean"),
    ).sort_values(["mean_rmse", "method"])
    pooled_records = []
    region_records = []
    for name, frame in predictions.groupby("method", sort=True):
        pooled_records.append({"method": name, **regression_metrics(
            frame["observed_rul"], frame["predicted_rul"]
        )})
        regions = {
            "short_history_le_100": frame["cutoff"].astype(float).le(100.0),
            "rul_1_25": frame["observed_rul"].astype(float).between(1.0, 25.0),
            "rul_26_50": frame["observed_rul"].astype(float).between(26.0, 50.0),
            "rul_51_75": frame["observed_rul"].astype(float).between(51.0, 75.0),
            "rul_76_100": frame["observed_rul"].astype(float).between(76.0, 100.0),
            "rul_101_plus": frame["observed_rul"].astype(float).ge(101.0),
        }
        for region, mask in regions.items():
            selected = frame.loc[mask]
            if selected.empty:
                continue
            region_records.append(
                {
                    "method": name,
                    "region": region,
                    "rows": len(selected),
                    "unique_uavs": selected["uav_id"].astype(str).nunique(),
                    **regression_metrics(
                        selected["observed_rul"], selected["predicted_rul"]
                    ),
                }
            )
    pooled = pd.DataFrame(pooled_records).rename(
        columns={
            "r2": "pooled_r2", "rmse": "pooled_rmse", "mae": "pooled_mae",
            "bias": "pooled_bias",
            "rms_overprediction": "pooled_rms_overprediction",
        }
    )
    summary = summary.merge(pooled, on="method", validate="one_to_one")
    control_folds = folds.loc[folds.method.eq(control), ["outer_fold", "rmse"]].rename(
        columns={"rmse": "control_rmse"}
    )
    candidates = summary.loc[summary.method.ne(control)].copy()
    decisions = []
    threshold_overrides = method_improvement_thresholds or {}
    rng = np.random.default_rng(20260907)
    for row in candidates.itertuples(index=False):
        paired = folds.loc[folds.method.eq(row.method)].merge(control_folds, on="outer_fold")
        wins = int(paired.rmse.lt(paired.control_rmse).sum())
        improvement = float(
            (paired.control_rmse.mean() - paired.rmse.mean()) / paired.control_rmse.mean()
        )
        candidate_rows = predictions.loc[
            predictions.method.eq(row.method)
        ].sort_values(identity).reset_index(drop=True)
        control_rows = predictions.loc[
            predictions.method.eq(control)
        ].sort_values(identity).reset_index(drop=True)
        if not candidate_rows[identity].equals(control_rows[identity]):
            raise ValueError(f"Method {row.method!r} does not align with control")
        group_errors = []
        for uav, indices in candidate_rows.groupby("uav_id", sort=True).groups.items():
            index = np.asarray(list(indices), dtype=int)
            observed = candidate_rows.loc[index, "observed_rul"].to_numpy(float)
            group_errors.append(
                (
                    str(uav),
                    np.square(
                        candidate_rows.loc[index, "predicted_rul"].to_numpy(float)
                        - observed
                    ),
                    np.square(
                        control_rows.loc[index, "predicted_rul"].to_numpy(float)
                        - observed
                    ),
                )
            )
        bootstrap_delta = []
        for _ in range(1000):
            sampled = rng.integers(0, len(group_errors), size=len(group_errors))
            candidate_squared = np.concatenate([group_errors[index][1] for index in sampled])
            control_squared = np.concatenate([group_errors[index][2] for index in sampled])
            bootstrap_delta.append(
                float(np.sqrt(candidate_squared.mean()) - np.sqrt(control_squared.mean()))
            )
        required_improvement = float(
            threshold_overrides.get(str(row.method), minimum_relative_rmse_improvement)
        )
        decisions.append(
            {"method": row.method, "fold_wins": wins, "relative_rmse_improvement": improvement,
             "required_relative_rmse_improvement": required_improvement,
             "paired_rmse_delta_bootstrap_low": float(np.quantile(bootstrap_delta, 0.025)),
             "paired_rmse_delta_bootstrap_high": float(np.quantile(bootstrap_delta, 0.975)),
             "passes_gate": wins >= minimum_fold_wins and improvement >= required_improvement}
        )
    decision_table = pd.DataFrame(decisions)
    passing = decision_table.loc[decision_table.passes_gate] if not decision_table.empty else decision_table
    if passing.empty:
        winner = control
    else:
        winner = str(
            summary.loc[summary.method.isin(passing.method)].sort_values("mean_rmse").iloc[0].method
        )
    predictions.to_csv(reporting / "method_predictions.csv.gz", index=False, compression="gzip")
    folds.to_csv(reporting / "fold_metrics.csv", index=False)
    summary.to_csv(reporting / "summary.csv", index=False)
    pd.DataFrame(region_records).to_csv(reporting / "region_metrics.csv", index=False)
    decision_table.to_csv(reporting / "promotion_decisions.csv", index=False)
    manifest = {
        "status": (
            "promoted"
            if winner != control and promotion_allowed
            else "screening_candidate"
            if winner != control
            else "no_promotion"
        ),
        "winner": winner,
        "control": control,
        "promoted": winner != control and promotion_allowed,
        "promotion_allowed": promotion_allowed,
        "selection_scope": "complete_outer_evaluation" if promotion_allowed else "screening_only",
        "minimum_fold_wins": minimum_fold_wins,
        "minimum_relative_rmse_improvement": minimum_relative_rmse_improvement,
        "uses_locked_evaluation": uses_locked_evaluation,
        "uses_test_labels": False,
        "bootstrap_unit": "uav_id",
        "bootstrap_repetitions": 1000,
        "unique_endpoints": int(
            rows[["uav_id", "cutoff"]].astype(str).drop_duplicates().shape[0]
        ),
    }
    write_json(reporting / "winner_manifest.json", manifest)
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    colors = ["#287271" if name == winner else "#68768a" for name in summary.method]
    axis.bar(summary.method, summary.mean_rmse, color=colors)
    axis.set_ylabel("Mean development RMSE")
    axis.set_title(root.parent.parent.name + " comparison")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure_path = reporting / "comparison.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    gallery = root / "figures"
    gallery.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure_path, gallery / figure_path.name)
    return manifest
