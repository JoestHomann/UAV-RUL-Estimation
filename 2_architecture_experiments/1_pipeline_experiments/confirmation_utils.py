"""Shared grouped-split and checkpoint helpers for PE_20 through PE_23."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from advanced_r2_utils import REPOSITORY_ROOT, subset_dataset


@dataclass(frozen=True)
class EvaluationJob:
    split_seed: int
    report_fold: int
    outer_fold: int
    evaluation_level: str
    inner_fold: int
    training_uavs: frozenset[str]
    validation_uavs: frozenset[str]


def _read(path_value: str, columns: list[str]) -> pd.DataFrame:
    path = (REPOSITORY_ROOT / path_value).resolve()
    try:
        return pd.read_csv(path, usecols=columns)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise ValueError(f"Cannot read grouped split source {path}: {error}") from error


def fixed_partitions(
    *,
    outer_path: str,
    inner_path: str,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer = _read(outer_path, ["uav_id", "outer_fold"])
    inner = _read(inner_path, ["outer_fold", "uav_id", "inner_fold"])
    outer["uav_id"] = outer.uav_id.astype(str)
    inner["uav_id"] = inner.uav_id.astype(str)
    outer["split_seed"] = int(split_seed)
    inner["split_seed"] = int(split_seed)
    return outer, inner


def generated_partitions(
    *,
    history_summary_path: str,
    split_seeds: Iterable[int],
    outer_fold_count: int,
    inner_fold_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from create_uav_grouped_folds import balanced_group_folds, make_inner_folds

    history = _read(
        history_summary_path,
        ["uav_id", "row_count", "final_cycle", "terminal_lifetime"],
    )
    outer_tables = []
    inner_tables = []
    for seed_value in split_seeds:
        seed = int(seed_value)
        outer = balanced_group_folds(
            history,
            n_folds=int(outer_fold_count),
            seed=seed,
            fold_column="outer_fold",
        )
        inner = make_inner_folds(
            outer,
            n_inner_folds=int(inner_fold_count),
            seed=seed,
        )
        outer["split_seed"] = seed
        inner["split_seed"] = seed
        outer_tables.append(outer[["split_seed", "uav_id", "outer_fold"]])
        inner_tables.append(
            inner[["split_seed", "outer_fold", "uav_id", "inner_fold"]]
        )
    if not outer_tables:
        raise ValueError("At least one split seed is required")
    return (
        pd.concat(outer_tables, ignore_index=True),
        pd.concat(inner_tables, ignore_index=True),
    )


def validate_partitions(
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    *,
    expected_outer_folds: int,
    expected_inner_folds: int,
) -> None:
    required_outer = {"split_seed", "uav_id", "outer_fold"}
    required_inner = {"split_seed", "outer_fold", "uav_id", "inner_fold"}
    if not required_outer.issubset(outer) or not required_inner.issubset(inner):
        raise ValueError("Grouped partition tables are missing required columns")
    if outer.duplicated(["split_seed", "uav_id"]).any():
        raise ValueError("Outer partitions assign one UAV more than once per seed")
    for seed, seed_outer in outer.groupby("split_seed", sort=True):
        folds = set(seed_outer.outer_fold.astype(int))
        if len(folds) != int(expected_outer_folds):
            raise ValueError(f"Split seed {seed} has {len(folds)} outer folds")
        expected_uavs = set(seed_outer.uav_id.astype(str))
        seed_inner = inner.loc[inner.split_seed.astype(int).eq(int(seed))]
        for outer_fold in sorted(folds):
            rows = seed_inner.loc[seed_inner.outer_fold.astype(int).eq(outer_fold)]
            held_outer = set(
                seed_outer.loc[
                    seed_outer.outer_fold.astype(int).eq(outer_fold), "uav_id"
                ].astype(str)
            )
            expected_training = expected_uavs - held_outer
            if set(rows.uav_id.astype(str)) != expected_training:
                raise ValueError(
                    f"Inner partitions for seed {seed}, outer {outer_fold} "
                    "do not equal the outer-training UAVs"
                )
            if rows.uav_id.astype(str).duplicated().any():
                raise ValueError("An inner partition assigns a UAV more than once")
            if rows.inner_fold.astype(int).nunique() != int(expected_inner_folds):
                raise ValueError(
                    f"Seed {seed}, outer {outer_fold} has the wrong inner-fold count"
                )


def evaluation_jobs(
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    *,
    include_inner: bool,
) -> list[EvaluationJob]:
    jobs: list[EvaluationJob] = []
    report_fold = 0
    for seed, seed_outer in outer.groupby("split_seed", sort=True):
        all_uavs = frozenset(seed_outer.uav_id.astype(str))
        for outer_fold in sorted(seed_outer.outer_fold.astype(int).unique()):
            validation = frozenset(
                seed_outer.loc[
                    seed_outer.outer_fold.astype(int).eq(outer_fold), "uav_id"
                ].astype(str)
            )
            training = all_uavs - validation
            if training & validation or not training or not validation:
                raise ValueError("Outer job contains overlapping or empty UAV groups")
            jobs.append(
                EvaluationJob(
                    split_seed=int(seed),
                    report_fold=report_fold,
                    outer_fold=int(outer_fold),
                    evaluation_level="outer",
                    inner_fold=-1,
                    training_uavs=frozenset(training),
                    validation_uavs=validation,
                )
            )
            if include_inner:
                selected = inner.loc[
                    inner.split_seed.astype(int).eq(int(seed))
                    & inner.outer_fold.astype(int).eq(int(outer_fold))
                ]
                for inner_fold in sorted(selected.inner_fold.astype(int).unique()):
                    inner_validation = frozenset(
                        selected.loc[
                            selected.inner_fold.astype(int).eq(inner_fold), "uav_id"
                        ].astype(str)
                    )
                    inner_training = training - inner_validation
                    if inner_training & inner_validation or not inner_training:
                        raise ValueError("Inner job contains overlapping or empty UAV groups")
                    jobs.append(
                        EvaluationJob(
                            split_seed=int(seed),
                            report_fold=report_fold,
                            outer_fold=int(outer_fold),
                            evaluation_level="inner",
                            inner_fold=int(inner_fold),
                            training_uavs=frozenset(inner_training),
                            validation_uavs=inner_validation,
                        )
                    )
            report_fold += 1
    return jobs


def select_uavs(dataset: Any, uav_ids: frozenset[str] | set[str]) -> Any:
    mask = dataset.metadata.uav_id.astype(str).isin(uav_ids).to_numpy()
    selected = subset_dataset(dataset, mask)
    observed = set(selected.metadata.uav_id.astype(str))
    if observed != set(uav_ids):
        missing = sorted(set(uav_ids) - observed)
        raise ValueError(f"Dataset does not cover requested UAVs: {missing[:5]}")
    return selected


def cell_complete(table: pd.DataFrame, job: EvaluationJob, methods: set[str]) -> bool:
    if table.empty:
        return False
    required = {
        "split_seed",
        "source_outer_fold",
        "evaluation_level",
        "inner_fold",
        "method",
    }
    if not required.issubset(table):
        raise ValueError("Existing checkpoint lacks nested confirmation metadata")
    rows = table.loc[
        table.split_seed.astype(int).eq(job.split_seed)
        & table.source_outer_fold.astype(int).eq(job.outer_fold)
        & table.evaluation_level.astype(str).eq(job.evaluation_level)
        & table.inner_fold.astype(int).eq(job.inner_fold)
    ]
    keys = ["method", "uav_id", "scenario", "cutoff"]
    if set(rows.method.astype(str)) != methods or rows.duplicated(keys).any():
        return False
    if set(rows.uav_id.astype(str)) != set(job.validation_uavs):
        return False
    counts = rows.groupby("method").size()
    return len(counts) == len(methods) and counts.nunique() == 1


def prediction_records(
    job: EvaluationJob,
    validation: Any,
    methods: dict[str, NDArray[np.float64]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    metadata = validation.metadata.reset_index(drop=True)
    if validation.target is None:
        raise ValueError("Confirmation validation data must be labelled")
    for method, values in methods.items():
        prediction = np.asarray(values, dtype=float).reshape(-1)
        if prediction.shape != (len(validation),) or not np.isfinite(prediction).all():
            raise ValueError(f"Method {method!r} produced invalid predictions")
        for index, row in metadata.iterrows():
            records.append(
                {
                    "split_seed": job.split_seed,
                    "outer_fold": job.report_fold,
                    "source_outer_fold": job.outer_fold,
                    "evaluation_level": job.evaluation_level,
                    "inner_fold": job.inner_fold,
                    "uav_id": str(row.uav_id),
                    "scenario": str(row.get("scenario", "development")),
                    "cutoff": float(row.cutoff),
                    "observed_rul": float(validation.target.iloc[index]),
                    "method": str(method),
                    "predicted_rul": max(float(prediction[index]), 0.0),
                }
            )
    return records
