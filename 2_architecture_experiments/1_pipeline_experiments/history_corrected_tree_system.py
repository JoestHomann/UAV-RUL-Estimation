"""Leakage-safe Run 7 residual correction with fixed-lag forecast history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from advanced_r2_utils import subset_dataset


HISTORY_COLUMNS = (
    "history_available",
    "history_count",
    "history_last_lag",
    "history_failure_cycle_delta",
    "history_failure_cycle_spread",
    "history_failure_cycle_slope",
    "history_near_cap_fraction",
)


@dataclass(frozen=True)
class HistoryFitSummary:
    training_rows: int
    training_uavs: int
    calibration_rows: int
    calibration_uavs: int
    internal_folds: int
    xgboost_weight: float
    history_lags: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_rows": self.training_rows,
            "training_uavs": self.training_uavs,
            "calibration_rows": self.calibration_rows,
            "calibration_uavs": self.calibration_uavs,
            "internal_folds": self.internal_folds,
            "xgboost_weight": self.xgboost_weight,
            "history_lags": list(self.history_lags),
        }


def _validate_lags(lags: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in lags}))
    if not values or any(value <= 0 for value in values):
        raise ValueError("Forecast-history lags must be distinct positive integers")
    return values


def build_lagged_dataset(
    raw: pd.DataFrame,
    query: Any,
    *,
    feature_names: list[str],
    lags: list[int] | tuple[int, ...],
    feature_profile: str,
) -> Any:
    """Build current and strictly earlier prefix features for every query row."""

    from build_prefix_features import build_feature_table
    from tabular_data_adapter import TabularDataset

    required_raw = {"uav_id", "flight_cycle"}
    if not required_raw.issubset(raw.columns):
        raise ValueError(f"Raw telemetry is missing {sorted(required_raw - set(raw.columns))}")
    required_metadata = {"uav_id", "cutoff"}
    if not required_metadata.issubset(query.metadata.columns):
        raise ValueError(
            f"Query metadata is missing {sorted(required_metadata - set(query.metadata.columns))}"
        )
    history_lags = _validate_lags(lags)
    bounds = raw.groupby("uav_id")["flight_cycle"].agg(["min", "max"])
    bounds.index = bounds.index.astype(str)
    records: list[dict[str, Any]] = []
    metadata = query.metadata.reset_index(drop=True)
    for query_index, row in metadata.iterrows():
        uav_id = str(row["uav_id"])
        if uav_id not in bounds.index:
            raise ValueError(f"Raw telemetry has no history for {uav_id}")
        cutoff_value = float(row["cutoff"])
        cutoff = int(round(cutoff_value))
        if not np.isclose(cutoff_value, cutoff):
            raise ValueError(f"Non-integer cutoff for {uav_id}: {cutoff_value}")
        minimum = int(bounds.loc[uav_id, "min"])
        maximum = int(bounds.loc[uav_id, "max"])
        if cutoff < minimum or cutoff > maximum:
            raise ValueError(
                f"Cutoff {cutoff} for {uav_id} is outside [{minimum}, {maximum}]"
            )
        scenario = str(row.get("scenario", "query"))
        source_sample = str(row.get("sample_id", f"query_{query_index}"))
        for lag in (0, *history_lags):
            lagged_cutoff = cutoff - lag
            if lagged_cutoff < minimum:
                continue
            records.append(
                {
                    "sample_id": f"history::{query_index}::{lag}::{source_sample}",
                    "scenario": f"{scenario}::lag_{lag}",
                    "uav_id": uav_id,
                    "cutoff": lagged_cutoff,
                    "query_index": int(query_index),
                    "history_lag": int(lag),
                }
            )
    manifest = pd.DataFrame.from_records(records)
    if manifest.empty or not manifest.loc[manifest.history_lag.eq(0), "query_index"].nunique() == len(query):
        raise ValueError("Lagged endpoint construction did not cover every query row")
    built = build_feature_table(
        raw,
        manifest,
        feature_profile=feature_profile,
    )
    missing = sorted(set(feature_names) - set(built.columns))
    if missing:
        raise ValueError(f"Lagged feature construction is missing {missing[:5]}")
    current = built.loc[built.history_lag.eq(0)].sort_values("query_index")
    if not np.allclose(
        current[feature_names].to_numpy(float),
        query.features[feature_names].reset_index(drop=True).to_numpy(float),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError("Rebuilt current-prefix features differ from the query dataset")
    lagged_metadata = built[
        ["sample_id", "scenario", "uav_id", "cutoff", "query_index", "history_lag"]
    ].reset_index(drop=True)
    return TabularDataset(
        features=built[feature_names].reset_index(drop=True),
        metadata=lagged_metadata,
        target=None,
        sample_weights=None,
        fitting_target=None,
    )


def fixed_lag_history_features(
    metadata: pd.DataFrame,
    base_prediction: NDArray[np.float64],
    *,
    query_count: int,
    near_cap_threshold: float,
) -> pd.DataFrame:
    """Summarize only explicit earlier-cutoff predictions for each query."""

    values = np.asarray(base_prediction, dtype=float).reshape(-1)
    if len(metadata) != len(values) or not np.isfinite(values).all():
        raise ValueError("Lagged metadata and predictions must be finite and aligned")
    table = metadata[["query_index", "history_lag", "cutoff"]].copy()
    table["predicted_rul"] = values
    output = pd.DataFrame(0.0, index=np.arange(query_count), columns=HISTORY_COLUMNS)
    for query_index, rows in table.groupby("query_index", sort=True):
        query_index = int(query_index)
        if query_index < 0 or query_index >= query_count:
            raise ValueError("Lagged metadata contains an invalid query index")
        current = rows.loc[rows.history_lag.eq(0)]
        if len(current) != 1:
            raise ValueError(f"Query {query_index} does not have exactly one current row")
        earlier = rows.loc[rows.history_lag.gt(0)].sort_values("history_lag")
        if earlier.empty:
            continue
        output.loc[query_index, "history_last_lag"] = float(earlier.history_lag.min())
        predictions = earlier.predicted_rul.to_numpy(float)
        usable = predictions < float(near_cap_threshold)
        output.loc[query_index, "history_available"] = float(bool(usable.any()))
        output.loc[query_index, "history_count"] = float(usable.sum())
        output.loc[query_index, "history_near_cap_fraction"] = float((~usable).mean())
        if not usable.any():
            continue
        selected = earlier.loc[usable].copy()
        failure_cycles = (
            selected.cutoff.to_numpy(float)
            + selected.predicted_rul.to_numpy(float)
        )
        current_failure = float(current.cutoff.iloc[0] + current.predicted_rul.iloc[0])
        most_recent = int(np.argmin(selected.history_lag.to_numpy(float)))
        output.loc[query_index, "history_failure_cycle_delta"] = (
            current_failure - float(failure_cycles[most_recent])
        )
        output.loc[query_index, "history_failure_cycle_spread"] = float(
            np.std(failure_cycles)
        )
        if len(selected) > 1:
            output.loc[query_index, "history_failure_cycle_slope"] = float(
                np.polyfit(
                    -selected.history_lag.to_numpy(float),
                    failure_cycles,
                    1,
                )[0]
            )
    if not np.isfinite(output.to_numpy(float)).all():
        raise ValueError("Forecast-history features contain non-finite values")
    return output


def _current_rows(
    metadata: pd.DataFrame,
    values: NDArray[np.float64],
    *,
    query_count: int,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    current = metadata.history_lag.eq(0).to_numpy()
    positions = metadata.loc[current, "query_index"].to_numpy(int)
    if len(positions) != query_count or set(positions) != set(range(query_count)):
        raise ValueError("Current lag rows do not map one-to-one to queries")
    result = np.empty((query_count, *array.shape[1:]), dtype=float)
    result[positions] = array[current]
    return result


class HistoryCorrectedTreeSystem:
    """Fit Run 7's members once and compare original/history residual heads."""

    def __init__(
        self,
        adapter: Any,
        *,
        history_lags: list[int] | tuple[int, ...],
        feature_profile: str = "extended",
        near_cap_threshold: float = 124.5,
    ) -> None:
        self.adapter = adapter
        self.history_lags = _validate_lags(history_lags)
        self.feature_profile = str(feature_profile)
        self.near_cap_threshold = float(near_cap_threshold)
        if not np.isfinite(self.near_cap_threshold):
            raise ValueError("Near-cap threshold must be finite")

    def _residual_model(self) -> Any:
        from models.tabular.residual_corrected_tree_ensemble import (
            ScaledRidgeResidualRegressor,
        )

        settings = self.adapter.contract["residual_model"]
        family = str(settings.get("family", "hist_gradient_boosting"))
        if family == "hist_gradient_boosting":
            return HistGradientBoostingRegressor(
                max_iter=int(settings["maximum_iterations"]),
                max_leaf_nodes=int(settings["maximum_leaf_nodes"]),
                min_samples_leaf=int(settings["minimum_samples_leaf"]),
                l2_regularization=float(settings["l2_regularization"]),
                learning_rate=float(settings["learning_rate"]),
                random_state=int(settings["seed"]),
            )
        if family == "ridge":
            return ScaledRidgeResidualRegressor(alpha=float(settings["alpha"]))
        raise ValueError(f"Unsupported residual family {family!r}")

    def _matrices(
        self,
        query: Any,
        lagged: Any,
        member_predictions: NDArray[np.float64],
        *,
        xgboost_weight: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame, NDArray[np.float64]]:
        xgb, extra, uncertainty_std, uncertainty_range, disagreement = (
            self.adapter._member_statistics(member_predictions)
        )
        all_base = np.maximum(
            xgboost_weight * xgb + (1.0 - xgboost_weight) * extra,
            self.adapter.prediction_minimum,
        )
        query_count = len(query)
        base = _current_rows(
            lagged.metadata,
            all_base,
            query_count=query_count,
        )
        current_uncertainty = _current_rows(
            lagged.metadata,
            uncertainty_std,
            query_count=query_count,
        )
        current_range = _current_rows(
            lagged.metadata,
            uncertainty_range,
            query_count=query_count,
        )
        current_disagreement = _current_rows(
            lagged.metadata,
            disagreement,
            query_count=query_count,
        )
        control = self.adapter._residual_matrix(
            query,
            base,
            current_uncertainty,
            current_range,
            current_disagreement,
        )
        history = fixed_lag_history_features(
            lagged.metadata,
            all_base,
            query_count=query_count,
            near_cap_threshold=self.near_cap_threshold,
        )
        enhanced = pd.concat(
            [control.reset_index(drop=True), history.reset_index(drop=True)],
            axis=1,
        )
        return control, enhanced, base

    def fit(self, training: Any, raw: pd.DataFrame) -> HistoryFitSummary:
        from models.tabular.residual_corrected_tree_ensemble import (
            calibration_sample_weights,
        )

        calibration = self.adapter._calibration_data(training)
        feature_names = [str(column) for column in training.features.columns]
        lagged = build_lagged_dataset(
            raw,
            calibration,
            feature_names=feature_names,
            lags=self.history_lags,
            feature_profile=self.feature_profile,
        )
        groups = calibration.metadata.uav_id.astype(str).to_numpy()
        splitter = GroupKFold(n_splits=int(self.adapter.internal_folds))
        oof_members = np.empty(
            (len(lagged), 2 * len(self.adapter.member_seeds)),
            dtype=float,
        )
        provenance: list[dict[str, Any]] = []
        for fold_number, (_, held_query_indices) in enumerate(
            splitter.split(calibration.features, groups=groups)
        ):
            held_uavs = set(groups[held_query_indices])
            training_mask = ~training.metadata.uav_id.astype(str).isin(held_uavs).to_numpy()
            fold_training = subset_dataset(training, training_mask)
            lagged_mask = lagged.metadata.uav_id.astype(str).isin(held_uavs).to_numpy()
            fold_lagged = subset_dataset(lagged, lagged_mask)
            predictions, _ = self.adapter._fit_members(
                fold_training,
                fold_lagged,
                retain=False,
            )
            oof_members[lagged_mask] = predictions
            provenance.append(
                {
                    "internal_fold": int(fold_number),
                    "training_uavs": int(fold_training.metadata.uav_id.nunique()),
                    "held_uavs": int(len(held_uavs)),
                    "uav_overlap": 0,
                }
            )
        if not np.isfinite(oof_members).all():
            raise ValueError("Internal OOF member predictions are incomplete")

        current_members = _current_rows(
            lagged.metadata,
            oof_members,
            query_count=len(calibration),
        )
        xgb, extra, *_ = self.adapter._member_statistics(current_members)
        observed = calibration.target.to_numpy(float)
        calibration_weights = calibration_sample_weights(
            calibration.metadata,
            self.adapter.calibration_weighting,
        )
        scored = []
        for weight in self.adapter.weight_grid:
            estimate = weight * xgb + (1.0 - weight) * extra
            rmse = float(
                np.sqrt(np.average(np.square(estimate - observed), weights=calibration_weights))
            )
            scored.append((rmse, float(weight)))
        _, self.xgboost_weight = min(scored, key=lambda item: (item[0], item[1]))
        control, enhanced, base = self._matrices(
            calibration,
            lagged,
            oof_members,
            xgboost_weight=self.xgboost_weight,
        )
        residual_target = base - observed
        self.control_residual_model = self._residual_model()
        self.history_residual_model = self._residual_model()
        self.control_residual_model.fit(
            control,
            residual_target,
            sample_weight=calibration_weights,
        )
        self.history_residual_model.fit(
            enhanced,
            residual_target,
            sample_weight=calibration_weights,
        )
        print(
            "History ensemble final refit: fitting "
            f"{2 * len(self.adapter.member_seeds)} members",
            flush=True,
        )
        _, self.members = self.adapter._fit_members(
            training,
            calibration,
            retain=True,
        )
        self.feature_names = feature_names
        self.fit_provenance = provenance
        self.summary = HistoryFitSummary(
            training_rows=len(training),
            training_uavs=int(training.metadata.uav_id.nunique()),
            calibration_rows=len(calibration),
            calibration_uavs=int(calibration.metadata.uav_id.nunique()),
            internal_folds=int(self.adapter.internal_folds),
            xgboost_weight=float(self.xgboost_weight),
            history_lags=self.history_lags,
        )
        return self.summary

    def predict_methods(self, query: Any, raw: pd.DataFrame) -> dict[str, NDArray[np.float64]]:
        if not hasattr(self, "members"):
            raise ValueError("History-corrected tree system is not fitted")
        lagged = build_lagged_dataset(
            raw,
            query,
            feature_names=self.feature_names,
            lags=self.history_lags,
            feature_profile=self.feature_profile,
        )
        member_predictions = np.column_stack(
            [model.predict(lagged) for model in self.members]
        )
        control, enhanced, base = self._matrices(
            query,
            lagged,
            member_predictions,
            xgboost_weight=self.xgboost_weight,
        )
        strength = float(self.adapter.correction_strength)
        raw_control = base - strength * np.asarray(
            self.control_residual_model.predict(control), dtype=float
        )
        raw_history = base - strength * np.asarray(
            self.history_residual_model.predict(enhanced), dtype=float
        )
        policy = self.adapter.prediction_policy
        return {
            "control": np.maximum(
                policy.adjust_predictions(raw_control),
                self.adapter.prediction_minimum,
            ),
            "prediction_history": np.maximum(
                policy.adjust_predictions(raw_history),
                self.adapter.prediction_minimum,
            ),
        }
