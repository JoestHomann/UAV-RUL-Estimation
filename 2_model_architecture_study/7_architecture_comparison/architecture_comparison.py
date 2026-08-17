"""Calculate the complete, non-ranking Phase 2 architecture comparison.

The stage consumes only the consolidated outputs accepted by the Step 7 gate.
It reports predictive metrics, reliability groups, paired whole-UAV bootstrap
uncertainty, random-seed variation, and computational cost. Architecture names
always remain in settings order; performance is never converted into a rank or
an automatically selected winner.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import sleep
from typing import Any

import numpy as np
import pandas as pd

from comparison_gate import ArchitectureComparisonPlan


METRICS = ("r2", "rmse", "mae", "bias")
AGE_BANDS = ("1-50", "51-100", "101-200", ">200")
GROUP_TYPES = ("overall", "outer_fold", "scenario", "age_band", "lifetime_quantile")

PREDICTION_COLUMNS = {
    "settings_version",
    "model_family",
    "configuration_id",
    "seed",
    "outer_fold",
    "scenario",
    "sample_id",
    "uav_id",
    "cutoff",
    "terminal_lifetime",
    "lifetime_quantile",
    "feature_set",
    "lookback",
    "y_true",
    "y_pred",
    "residual",
}

RUN_COLUMNS = {
    "settings_version",
    "model_family",
    "configuration_id",
    "seed",
    "outer_fold",
    "training_rows",
    "training_uavs",
    "validation_rows",
    "validation_uavs",
    "locked_scenarios",
    "training_seconds",
    "inference_seconds",
    "epochs_or_iterations",
    "trainable_parameters",
    "serialized_model_bytes",
    "prediction_rows",
}


class ArchitectureComparisonError(ValueError):
    """Represent invalid locked results or comparison calculations."""


@dataclass(frozen=True)
class ComparisonTables:
    """Keep all traceable Step 7 tables together before persistence."""

    architecture_comparison: pd.DataFrame
    seed_metrics: pd.DataFrame
    grouped_metrics: pd.DataFrame
    grouped_architecture_metrics: pd.DataFrame
    bootstrap_architecture_metrics: pd.DataFrame
    paired_metric_differences: pd.DataFrame
    efficiency_summary: pd.DataFrame


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate the four metrics fixed by the architecture study settings."""

    observed = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    residual = predicted - observed
    denominator = float(np.sum(np.square(observed - np.mean(observed))))
    r2 = (
        float("nan")
        if denominator <= 0.0
        else 1.0 - float(np.sum(np.square(residual))) / denominator
    )
    return {
        "r2": r2,
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


def _weighted_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    row_weights: np.ndarray,
) -> dict[str, float]:
    """Calculate metrics after resampling complete UAV groups with replacement."""

    weight_sum = float(np.sum(row_weights))
    weighted_mean = float(np.sum(row_weights * y_true) / weight_sum)
    residual = y_pred - y_true
    denominator = float(
        np.sum(row_weights * np.square(y_true - weighted_mean))
    )
    squared_error = float(np.sum(row_weights * np.square(residual)) / weight_sum)
    return {
        "r2": (
            float("nan")
            if denominator <= 0.0
            else 1.0
            - float(np.sum(row_weights * np.square(residual))) / denominator
        ),
        "rmse": float(np.sqrt(squared_error)),
        "mae": float(np.sum(row_weights * np.abs(residual)) / weight_sum),
        "bias": float(np.sum(row_weights * residual) / weight_sum),
    }


def _numeric_then_text_sort_key(value: Any) -> tuple[int, float | str]:
    """Sort numeric group labels numerically and other labels alphabetically."""

    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _replace_with_retry(temporary_path: Path, path: Path) -> None:
    """Atomically replace ``path`` with ``temporary_path``, tolerating Windows locks.

    POSIX ``rename`` succeeds even while another process has ``path`` open.
    Windows instead raises ``PermissionError`` (WinError 32) if some other
    process -- for example a lingering antivirus/indexer scan right after the
    file is created -- currently has it open. That lock is always transient,
    so a short bounded retry resolves it without weakening the atomicity of
    the replace itself.
    """

    last_error: PermissionError | None = None
    for attempt in range(10):
        try:
            temporary_path.replace(path)
            return
        except PermissionError as error:
            last_error = error
            sleep(0.1 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _write_csv(table: pd.DataFrame, path: Path, *, compressed: bool = False) -> None:
    """Atomically write one stable CSV, optionally with deterministic gzip."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    compression: str | dict[str, Any] | None = None
    if compressed:
        compression = {"method": "gzip", "mtime": 0}
    table.to_csv(
        temporary_path,
        index=False,
        float_format="%.12g",
        compression=compression,
    )
    _replace_with_retry(temporary_path, path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Atomically write one readable JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _replace_with_retry(temporary_path, path)


class ArchitectureComparisonAnalyzer:
    """Validate and compare one complete set of locked architecture results."""

    def __init__(
        self,
        predictions: pd.DataFrame,
        model_runs: pd.DataFrame,
        plan: ArchitectureComparisonPlan,
    ) -> None:
        self.plan = plan
        self.settings = plan.settings
        self.settings_version = int(self.settings["settings_version"])
        self.predictions = predictions.copy()
        self.model_runs = model_runs.copy()
        self._validate_inputs()

        # The age categories are derived only after the locked input has passed
        # every structural check. These are the exact Phase 1 reporting bands.
        self.predictions["age_band"] = pd.cut(
            self.predictions["cutoff"],
            bins=[0, 50, 100, 200, np.inf],
            labels=list(AGE_BANDS),
            include_lowest=True,
            right=True,
        )

    def calculate(self) -> ComparisonTables:
        """Calculate all declared tables without ranking model families."""

        seed_metrics = self._seed_metrics()
        grouped_metrics = self._grouped_metrics()
        grouped_architecture = self._average_group_metrics(grouped_metrics)
        bootstrap_metrics = self._bootstrap_metrics()
        architecture_comparison = self._architecture_summary(
            seed_metrics,
            bootstrap_metrics,
        )
        paired_differences = self._paired_differences(
            architecture_comparison,
            bootstrap_metrics,
        )
        efficiency = self._efficiency_summary()
        return ComparisonTables(
            architecture_comparison=architecture_comparison,
            seed_metrics=seed_metrics,
            grouped_metrics=grouped_metrics,
            grouped_architecture_metrics=grouped_architecture,
            bootstrap_architecture_metrics=bootstrap_metrics,
            paired_metric_differences=paired_differences,
            efficiency_summary=efficiency,
        )

    def _validate_inputs(self) -> None:
        """Verify schemas, run coverage, alignment, and locked target integrity."""

        missing_predictions = sorted(PREDICTION_COLUMNS - set(self.predictions.columns))
        missing_runs = sorted(RUN_COLUMNS - set(self.model_runs.columns))
        if missing_predictions:
            raise ArchitectureComparisonError(
                f"Locked predictions are missing columns {missing_predictions}"
            )
        if missing_runs:
            raise ArchitectureComparisonError(
                f"Model runs are missing columns {missing_runs}"
            )
        if len(self.predictions) != self.plan.expected_prediction_rows:
            raise ArchitectureComparisonError(
                "Locked prediction row count differs from the complete Step 6 plan"
            )
        if len(self.model_runs) != self.plan.expected_run_count:
            raise ArchitectureComparisonError(
                "Model-run row count differs from the complete Step 6 plan"
            )

        numeric_prediction_columns = [
            "settings_version",
            "seed",
            "outer_fold",
            "scenario",
            "cutoff",
            "terminal_lifetime",
            "y_true",
            "y_pred",
            "residual",
        ]
        numeric_predictions = self.predictions[numeric_prediction_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if not np.isfinite(numeric_predictions.to_numpy(dtype=float)).all():
            raise ArchitectureComparisonError(
                "Locked predictions contain missing or non-finite numeric values"
            )
        self.predictions[numeric_prediction_columns] = numeric_predictions

        numeric_run_columns = [
            "settings_version",
            "seed",
            "outer_fold",
            "training_rows",
            "training_uavs",
            "validation_rows",
            "validation_uavs",
            "locked_scenarios",
            "training_seconds",
            "inference_seconds",
            "serialized_model_bytes",
            "prediction_rows",
        ]
        numeric_runs = self.model_runs[numeric_run_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if not np.isfinite(numeric_runs.to_numpy(dtype=float)).all():
            raise ArchitectureComparisonError(
                "Model runs contain missing or non-finite required numeric values"
            )
        self.model_runs[numeric_run_columns] = numeric_runs

        if set(self.predictions["settings_version"].astype(int)) != {
            self.settings_version
        }:
            raise ArchitectureComparisonError(
                "Locked predictions do not use the active settings version"
            )
        if set(self.model_runs["settings_version"].astype(int)) != {
            self.settings_version
        }:
            raise ArchitectureComparisonError(
                "Model runs do not use the active settings version"
            )
        if set(self.predictions["model_family"].astype(str)) != set(
            self.plan.enabled_families
        ):
            raise ArchitectureComparisonError(
                "Locked predictions do not contain exactly the enabled families"
            )
        if set(self.model_runs["model_family"].astype(str)) != set(
            self.plan.enabled_families
        ):
            raise ArchitectureComparisonError(
                "Model runs do not contain exactly the enabled families"
            )
        if (self.predictions["y_pred"] < float(
            self.settings["evaluation"]["prediction_minimum"]
        )).any():
            raise ArchitectureComparisonError(
                "Locked predictions violate the configured nonnegative boundary"
            )
        if not np.allclose(
            self.predictions["residual"],
            self.predictions["y_pred"] - self.predictions["y_true"],
            rtol=1e-10,
            atol=1e-9,
        ):
            raise ArchitectureComparisonError(
                "Stored residuals disagree with prediction minus target"
            )

        duplicate_keys = ["model_family", "seed", "outer_fold", "scenario", "uav_id"]
        if self.predictions.duplicated(duplicate_keys).any():
            raise ArchitectureComparisonError(
                "Locked predictions contain duplicate family/seed/fold/scenario/UAV rows"
            )
        run_keys = ["model_family", "seed", "outer_fold"]
        if self.model_runs.duplicated(run_keys).any():
            raise ArchitectureComparisonError(
                "Model runs contain duplicate family/seed/fold rows"
            )

        expected_training_uavs = int(self.settings["phase_1"]["expected_training_uavs"])
        fold_count = len(self.plan.outer_fold_labels)
        expected_validation_uavs = expected_training_uavs // fold_count
        expected_outer_training_uavs = expected_training_uavs - expected_validation_uavs
        expected_scenarios = int(self.settings["phase_1"]["expected_locked_scenarios"])
        expected_training_rows = (
            expected_outer_training_uavs
            * int(self.settings["phase_1"]["expected_prefixes_per_training_uav"])
        )
        expected_rows_per_run = expected_validation_uavs * expected_scenarios

        expected_run_keys = {
            (family, outer_fold, seed)
            for family in self.plan.enabled_families
            for outer_fold in self.plan.outer_fold_labels
            for seed in self.plan.seeds_by_family[family]
        }
        observed_run_keys = {
            (str(row.model_family), int(row.outer_fold), int(row.seed))
            for row in self.model_runs.itertuples(index=False)
        }
        if observed_run_keys != expected_run_keys:
            raise ArchitectureComparisonError(
                "Model runs do not cover every expected family/fold/seed key"
            )

        expected_run_facts = {
            "training_rows": expected_training_rows,
            "training_uavs": expected_outer_training_uavs,
            "validation_rows": expected_rows_per_run,
            "validation_uavs": expected_validation_uavs,
            "locked_scenarios": expected_scenarios,
            "prediction_rows": expected_rows_per_run,
        }
        for column, expected_value in expected_run_facts.items():
            if not (self.model_runs[column].astype(int) == expected_value).all():
                raise ArchitectureComparisonError(
                    f"Model-run column {column!r} does not equal {expected_value}"
                )
        if (self.model_runs[["training_seconds", "inference_seconds"]] < 0).any().any():
            raise ArchitectureComparisonError(
                "Model runs contain negative timing measurements"
            )
        if (self.model_runs["serialized_model_bytes"] <= 0).any():
            raise ArchitectureComparisonError(
                "Model runs contain a nonpositive serialized model size"
            )

        grouped_predictions = self.predictions.groupby(run_keys, sort=False, observed=True)
        prediction_key_set = {
            (str(family), int(seed), int(fold))
            for family, seed, fold in grouped_predictions.groups
        }
        normalized_prediction_keys = {
            (family, seed, fold)
            for family, fold, seed in expected_run_keys
        }
        if prediction_key_set != normalized_prediction_keys:
            raise ArchitectureComparisonError(
                "Locked predictions do not cover every expected family/fold/seed key"
            )

        run_lookup = self.model_runs.set_index(run_keys)
        for (family, seed, outer_fold), group in grouped_predictions:
            if len(group) != expected_rows_per_run:
                raise ArchitectureComparisonError(
                    f"Run {family}/{outer_fold}/{seed} has {len(group)} predictions"
                )
            if group["uav_id"].nunique() != expected_validation_uavs:
                raise ArchitectureComparisonError(
                    f"Run {family}/{outer_fold}/{seed} has an invalid UAV count"
                )
            if set(group["scenario"].astype(int)) != set(range(1, expected_scenarios + 1)):
                raise ArchitectureComparisonError(
                    f"Run {family}/{outer_fold}/{seed} has invalid scenario labels"
                )
            expected_configuration = str(
                run_lookup.loc[(family, seed, outer_fold), "configuration_id"]
            )
            if set(group["configuration_id"].astype(str)) != {expected_configuration}:
                raise ArchitectureComparisonError(
                    f"Run {family}/{outer_fold}/{seed} has inconsistent configuration IDs"
                )

        self._validate_evaluation_alignment(expected_training_uavs, expected_scenarios)

    def _validate_evaluation_alignment(
        self,
        expected_uavs: int,
        expected_scenarios: int,
    ) -> None:
        """Confirm that every architecture predicts the same locked endpoints."""

        evaluation_keys = ["outer_fold", "scenario", "uav_id"]
        first_family = self.plan.enabled_families[0]
        first_seed = self.plan.seeds_by_family[first_family][0]
        reference = self.predictions.loc[
            (self.predictions["model_family"] == first_family)
            & (self.predictions["seed"].astype(int) == first_seed)
        ].sort_values(evaluation_keys, kind="stable")
        expected_endpoints = expected_uavs * expected_scenarios
        if len(reference) != expected_endpoints:
            raise ArchitectureComparisonError(
                "Reference architecture does not contain every locked endpoint"
            )
        if reference["uav_id"].nunique() != expected_uavs:
            raise ArchitectureComparisonError(
                "Reference architecture does not contain every training UAV"
            )
        if not (reference.groupby("uav_id")["outer_fold"].nunique() == 1).all():
            raise ArchitectureComparisonError(
                "A UAV appears in more than one locked outer fold"
            )
        if not (reference.groupby("uav_id")["scenario"].nunique() == expected_scenarios).all():
            raise ArchitectureComparisonError(
                "A UAV does not contain every locked validation scenario"
            )

        exact_metadata = [
            "outer_fold",
            "scenario",
            "uav_id",
            "sample_id",
            "cutoff",
            "lifetime_quantile",
        ]
        numeric_metadata = ["terminal_lifetime", "y_true"]
        reference_exact = reference[exact_metadata].reset_index(drop=True).astype(str)
        reference_numeric = reference[numeric_metadata].reset_index(drop=True).to_numpy(float)
        for family in self.plan.enabled_families:
            for seed in self.plan.seeds_by_family[family]:
                candidate = self.predictions.loc[
                    (self.predictions["model_family"] == family)
                    & (self.predictions["seed"].astype(int) == seed)
                ].sort_values(evaluation_keys, kind="stable")
                if len(candidate) != expected_endpoints:
                    raise ArchitectureComparisonError(
                        f"Architecture {family!r}, seed {seed} is missing endpoints"
                    )
                if not candidate[exact_metadata].reset_index(drop=True).astype(str).equals(
                    reference_exact
                ):
                    raise ArchitectureComparisonError(
                        f"Architecture {family!r}, seed {seed} uses different endpoint metadata"
                    )
                if not np.allclose(
                    candidate[numeric_metadata].reset_index(drop=True).to_numpy(float),
                    reference_numeric,
                    rtol=0.0,
                    atol=1e-10,
                ):
                    raise ArchitectureComparisonError(
                        f"Architecture {family!r}, seed {seed} uses different locked targets"
                    )

    def _seed_metrics(self) -> pd.DataFrame:
        """Calculate overall performance separately for every retained seed."""

        records: list[dict[str, Any]] = []
        for family in self.plan.enabled_families:
            for seed in self.plan.seeds_by_family[family]:
                group = self.predictions.loc[
                    (self.predictions["model_family"] == family)
                    & (self.predictions["seed"].astype(int) == seed)
                ]
                record: dict[str, Any] = {
                    "settings_version": self.settings_version,
                    "model_family": family,
                    "seed": seed,
                    "rows": len(group),
                    "uavs": group["uav_id"].nunique(),
                }
                record.update(_regression_metrics(group["y_true"], group["y_pred"]))
                records.append(record)
        return pd.DataFrame.from_records(records)

    def _grouped_metrics(self) -> pd.DataFrame:
        """Calculate every reliability view separately for every retained seed."""

        group_definitions = {
            "overall": None,
            "outer_fold": "outer_fold",
            "scenario": "scenario",
            "age_band": "age_band",
            "lifetime_quantile": "lifetime_quantile",
        }
        records: list[dict[str, Any]] = []
        for family in self.plan.enabled_families:
            for seed in self.plan.seeds_by_family[family]:
                family_seed = self.predictions.loc[
                    (self.predictions["model_family"] == family)
                    & (self.predictions["seed"].astype(int) == seed)
                ]
                for group_type, column in group_definitions.items():
                    if column is None:
                        groups = [("all", family_seed)]
                    else:
                        if group_type == "outer_fold":
                            ordered_values: list[Any] = list(
                                self.plan.outer_fold_labels
                            )
                        elif group_type == "scenario":
                            ordered_values = list(
                                range(
                                    1,
                                    int(
                                        self.settings["phase_1"][
                                            "expected_locked_scenarios"
                                        ]
                                    )
                                    + 1,
                                )
                            )
                        elif group_type == "age_band":
                            ordered_values = list(AGE_BANDS)
                        else:
                            # Phase 1 lifetime quantiles are integer labels.
                            # Numeric sorting prevents their first occurrence
                            # in the fold table from defining a confusing axis.
                            ordered_values = sorted(
                                family_seed[column].dropna().unique().tolist(),
                                key=_numeric_then_text_sort_key,
                            )
                        groups = [
                            (value, family_seed.loc[family_seed[column] == value])
                            for value in ordered_values
                        ]
                    for position, (value, group) in enumerate(groups):
                        record: dict[str, Any] = {
                            "settings_version": self.settings_version,
                            "model_family": family,
                            "seed": seed,
                            "group_type": group_type,
                            "group_value": str(value),
                            "group_position": position,
                            "rows": len(group),
                            "uavs": group["uav_id"].nunique(),
                        }
                        record.update(
                            _regression_metrics(group["y_true"], group["y_pred"])
                        )
                        records.append(record)
        return pd.DataFrame.from_records(records)

    def _average_group_metrics(self, grouped: pd.DataFrame) -> pd.DataFrame:
        """Average each reliability metric across a family's retained seeds."""

        records: list[dict[str, Any]] = []
        key_columns = ["model_family", "group_type", "group_value", "group_position"]
        for keys, group in grouped.groupby(key_columns, sort=False, observed=True):
            family, group_type, group_value, group_position = keys
            record: dict[str, Any] = {
                "settings_version": self.settings_version,
                "model_family": family,
                "group_type": group_type,
                "group_value": group_value,
                "group_position": int(group_position),
                "seed_count": group["seed"].nunique(),
                "rows_per_seed": int(group["rows"].iloc[0]),
                "uavs_per_seed": int(group["uavs"].iloc[0]),
            }
            for metric in METRICS:
                values = group[metric].to_numpy(dtype=float)
                record[f"{metric}_mean"] = float(np.mean(values))
                record[f"{metric}_seed_sd"] = float(np.std(values, ddof=0))
            records.append(record)
        result = pd.DataFrame.from_records(records)
        result["family_order"] = result["model_family"].map(
            {family: index for index, family in enumerate(self.plan.enabled_families)}
        )
        return result.sort_values(
            ["family_order", "group_type", "group_position"],
            kind="stable",
        ).drop(columns="family_order").reset_index(drop=True)

    def _aligned_prediction_arrays(
        self,
    ) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]], np.ndarray]:
        """Align every seed prediction to one shared endpoint and UAV order."""

        keys = ["outer_fold", "scenario", "uav_id"]
        first_family = self.plan.enabled_families[0]
        first_seed = self.plan.seeds_by_family[first_family][0]
        reference = self.predictions.loc[
            (self.predictions["model_family"] == first_family)
            & (self.predictions["seed"].astype(int) == first_seed)
        ].sort_values(keys, kind="stable").reset_index(drop=True)
        prediction_arrays: dict[str, list[np.ndarray]] = {}
        for family in self.plan.enabled_families:
            prediction_arrays[family] = []
            for seed in self.plan.seeds_by_family[family]:
                ordered = self.predictions.loc[
                    (self.predictions["model_family"] == family)
                    & (self.predictions["seed"].astype(int) == seed)
                ].sort_values(keys, kind="stable")
                prediction_arrays[family].append(
                    ordered["y_pred"].to_numpy(dtype=np.float64)
                )
        uav_names = sorted(reference["uav_id"].astype(str).unique())
        uav_index = {uav: index for index, uav in enumerate(uav_names)}
        row_uav_indices = reference["uav_id"].astype(str).map(uav_index).to_numpy(int)
        return reference, prediction_arrays, row_uav_indices

    def _bootstrap_metrics(self) -> pd.DataFrame:
        """Apply one paired whole-UAV resample to every family and every seed."""

        repetitions = int(self.settings["evaluation"]["bootstrap_repetitions"])
        seed = int(self.settings["evaluation"]["bootstrap_seed"])
        reference, predictions, row_uav_indices = self._aligned_prediction_arrays()
        y_true = reference["y_true"].to_numpy(dtype=np.float64)
        uav_count = reference["uav_id"].nunique()
        rng = np.random.default_rng(seed)
        records: list[dict[str, Any]] = []

        for repetition in range(repetitions):
            # Sampling integer UAV positions and turning them into row weights
            # reproduces concatenated group resampling without repeatedly
            # constructing large temporary data frames.
            sampled = rng.integers(0, uav_count, size=uav_count)
            uav_weights = np.bincount(sampled, minlength=uav_count).astype(float)
            row_weights = uav_weights[row_uav_indices]
            for family in self.plan.enabled_families:
                per_seed = [
                    _weighted_regression_metrics(y_true, values, row_weights)
                    for values in predictions[family]
                ]
                record: dict[str, Any] = {
                    "settings_version": self.settings_version,
                    "bootstrap_repetition": repetition,
                    "model_family": family,
                    "sampled_uav_draws": uav_count,
                    "seed_count": len(per_seed),
                }
                for metric in METRICS:
                    # Averaging metric values preserves individual-seed model
                    # evaluation. Averaging predictions here would instead
                    # evaluate an undeclared ensemble architecture.
                    record[metric] = float(
                        np.mean([result[metric] for result in per_seed])
                    )
                records.append(record)
        return pd.DataFrame.from_records(records)

    def _architecture_summary(
        self,
        seed_metrics: pd.DataFrame,
        bootstrap: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine seed means, seed variation, and UAV-bootstrap intervals."""

        records: list[dict[str, Any]] = []
        for family in self.plan.enabled_families:
            family_seeds = seed_metrics.loc[seed_metrics["model_family"] == family]
            family_bootstrap = bootstrap.loc[bootstrap["model_family"] == family]
            record: dict[str, Any] = {
                "settings_version": self.settings_version,
                "model_family": family,
                "seed_count": family_seeds["seed"].nunique(),
                "evaluation_rows_per_seed": int(family_seeds["rows"].iloc[0]),
                "evaluation_uavs_per_seed": int(family_seeds["uavs"].iloc[0]),
            }
            for metric in METRICS:
                seed_values = family_seeds[metric].to_numpy(dtype=float)
                bootstrap_values = family_bootstrap[metric].to_numpy(dtype=float)
                finite_bootstrap = bootstrap_values[np.isfinite(bootstrap_values)]
                if not len(finite_bootstrap):
                    raise ArchitectureComparisonError(
                        f"Bootstrap produced no finite {metric} values for {family!r}"
                    )
                record[f"{metric}_mean"] = float(np.mean(seed_values))
                record[f"{metric}_seed_sd"] = float(np.std(seed_values, ddof=0))
                record[f"{metric}_ci_lower_95"] = float(
                    np.quantile(finite_bootstrap, 0.025)
                )
                record[f"{metric}_ci_upper_95"] = float(
                    np.quantile(finite_bootstrap, 0.975)
                )
            records.append(record)
        return pd.DataFrame.from_records(records)

    def _paired_differences(
        self,
        architecture_summary: pd.DataFrame,
        bootstrap: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate paired family A minus family B intervals without ranking."""

        point_lookup = architecture_summary.set_index("model_family")
        bootstrap_lookup = {
            family: bootstrap.loc[bootstrap["model_family"] == family]
            .sort_values("bootstrap_repetition")
            .reset_index(drop=True)
            for family in self.plan.enabled_families
        }
        records: list[dict[str, Any]] = []
        for first_index, family_a in enumerate(self.plan.enabled_families):
            for family_b in self.plan.enabled_families[first_index + 1 :]:
                for metric in METRICS:
                    differences = (
                        bootstrap_lookup[family_a][metric].to_numpy(float)
                        - bootstrap_lookup[family_b][metric].to_numpy(float)
                    )
                    finite = differences[np.isfinite(differences)]
                    interpretation = {
                        "r2": "positive means family_a has higher R2",
                        "rmse": "negative means family_a has lower RMSE",
                        "mae": "negative means family_a has lower MAE",
                        "bias": "signed bias difference; zero is the target for each family",
                    }[metric]
                    records.append(
                        {
                            "settings_version": self.settings_version,
                            "family_a": family_a,
                            "family_b": family_b,
                            "metric": metric,
                            "difference_a_minus_b": float(
                                point_lookup.loc[family_a, f"{metric}_mean"]
                                - point_lookup.loc[family_b, f"{metric}_mean"]
                            ),
                            "ci_lower_95": float(np.quantile(finite, 0.025)),
                            "ci_upper_95": float(np.quantile(finite, 0.975)),
                            "bootstrap_repetitions": len(finite),
                            "interpretation": interpretation,
                        }
                    )
        return pd.DataFrame.from_records(records)

    def _efficiency_summary(self) -> pd.DataFrame:
        """Summarize measured cost without mixing it into a performance score."""

        def mean_or_nan(series: pd.Series) -> float:
            numeric = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
            return float(np.mean(numeric)) if len(numeric) else float("nan")

        def sd_or_nan(series: pd.Series) -> float:
            numeric = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
            return float(np.std(numeric, ddof=0)) if len(numeric) else float("nan")

        records: list[dict[str, Any]] = []
        for family in self.plan.enabled_families:
            runs = self.model_runs.loc[self.model_runs["model_family"] == family]
            total_predictions = float(runs["prediction_rows"].sum())
            records.append(
                {
                    "settings_version": self.settings_version,
                    "model_family": family,
                    "run_count": len(runs),
                    "training_seconds_mean_per_run": mean_or_nan(runs["training_seconds"]),
                    "training_seconds_sd_per_run": sd_or_nan(runs["training_seconds"]),
                    "training_seconds_total": float(runs["training_seconds"].sum()),
                    "inference_seconds_mean_per_run": mean_or_nan(runs["inference_seconds"]),
                    "inference_seconds_sd_per_run": sd_or_nan(runs["inference_seconds"]),
                    "inference_seconds_total": float(runs["inference_seconds"].sum()),
                    "inference_milliseconds_per_endpoint": float(
                        1000.0 * runs["inference_seconds"].sum() / total_predictions
                    ),
                    "trainable_parameters_mean": mean_or_nan(runs["trainable_parameters"]),
                    "trainable_parameters_sd": sd_or_nan(runs["trainable_parameters"]),
                    "serialized_model_bytes_mean": mean_or_nan(
                        runs["serialized_model_bytes"]
                    ),
                    "serialized_model_bytes_sd": sd_or_nan(
                        runs["serialized_model_bytes"]
                    ),
                }
            )
        return pd.DataFrame.from_records(records)


def save_comparison(
    tables: ComparisonTables,
    plan: ArchitectureComparisonPlan,
    output_dir: Path,
) -> dict[str, Any]:
    """Persist tables, figures, and a manifest that explicitly forbids ranking."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table_paths = {
        "architecture_comparison": output_dir / "architecture_comparison.csv",
        "seed_metrics": output_dir / "seed_metrics.csv",
        "grouped_metrics": output_dir / "grouped_metrics.csv",
        "grouped_architecture_metrics": output_dir
        / "grouped_architecture_metrics.csv",
        "bootstrap_architecture_metrics": output_dir
        / "bootstrap_architecture_metrics.csv.gz",
        "paired_metric_differences": output_dir / "paired_metric_differences.csv",
        "efficiency_summary": output_dir / "efficiency_summary.csv",
    }
    _write_csv(tables.architecture_comparison, table_paths["architecture_comparison"])
    _write_csv(tables.seed_metrics, table_paths["seed_metrics"])
    _write_csv(tables.grouped_metrics, table_paths["grouped_metrics"])
    _write_csv(
        tables.grouped_architecture_metrics,
        table_paths["grouped_architecture_metrics"],
    )
    _write_csv(
        tables.bootstrap_architecture_metrics,
        table_paths["bootstrap_architecture_metrics"],
        compressed=True,
    )
    _write_csv(
        tables.paired_metric_differences,
        table_paths["paired_metric_differences"],
    )
    _write_csv(tables.efficiency_summary, table_paths["efficiency_summary"])

    # Plotting is separated from calculation so the numerical tables remain
    # directly testable and reusable without a display backend.
    from plot_architecture_comparison import create_comparison_figures

    figure_paths = create_comparison_figures(tables, plan, output_dir / "figures")
    manifest = {
        "comparison_version": 1,
        "settings_version": int(plan.settings["settings_version"]),
        "status": "complete",
        "step_6_prerequisite": "complete",
        "enabled_families": list(plan.enabled_families),
        "metrics": list(METRICS),
        "reported_groups": list(GROUP_TYPES),
        # "uav_id" was the only supported value; the setting itself was
        # removed as dead configuration, so this is now a fixed literal.
        "bootstrap_unit": "uav_id",
        "bootstrap_repetitions": int(
            plan.settings["evaluation"]["bootstrap_repetitions"]
        ),
        "bootstrap_seed": int(plan.settings["evaluation"]["bootstrap_seed"]),
        "seed_aggregation": "mean of individual-seed metric values",
        "predictions_ensembled_across_seeds": False,
        "paired_comparison_keys": ["uav_id", "scenario"],
        "automatic_architecture_ranking": False,
        "automatic_architecture_selection": False,
        "winner_artifact_written": False,
        "locked_results_used_for_tuning": False,
        "test_data_loaded": False,
        "artifacts": {
            name: path.relative_to(output_dir).as_posix()
            for name, path in table_paths.items()
        },
        "figures": [
            path.relative_to(output_dir).as_posix() for path in figure_paths
        ],
    }
    _write_json(manifest, output_dir / "comparison_manifest.json")
    return manifest
