"""Run a grouped feature-importance screening study on Run 10 XGBoost.

This exploratory analysis reuses the 15 completed XGBoost outer-fold models
(five grouped UAV folds and three model seeds).  It evaluates only historical
held-out endpoints.  Whole feature blocks are permuted within each historical
scenario, preserving the correlations inside a sensor or transformation block.
Uncertainty is calculated by resampling complete UAVs.

The output is a screening result.  Features selected from it must be confirmed
by a training-only ranking plus outer-fold retraining ablation before changing
the production feature set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from plot_engineered_run_10_xgboost_explanations import (
    BLUE,
    GREEN,
    GRID,
    HERE,
    RUN,
    configure_style,
    feature_family,
    load_adapter,
    prepare_inputs,
    sensor_family,
)


REPORTING = RUN / "reporting"
OUTPUT = REPORTING / "feature_importance_study"
DEFAULT_PERMUTATIONS = 20
DEFAULT_BOOTSTRAPS = 2_000
DEFAULT_SEED = 20270912


def grouped_columns(names: tuple[str, ...]) -> dict[str, dict[str, list[int]]]:
    groups: dict[str, dict[str, list[int]]] = {"feature_family": {}, "sensor": {}}
    for index, name in enumerate(names):
        groups["feature_family"].setdefault(feature_family(name), []).append(index)
        groups["sensor"].setdefault(sensor_family(name), []).append(index)
    expected = set(range(len(names)))
    for group_type, definitions in groups.items():
        observed = sorted(index for columns in definitions.values() for index in columns)
        if observed != sorted(expected):
            raise ValueError(f"{group_type} groups do not cover every feature exactly once")
    return groups


def permuted_prediction_error(
    estimator,
    values: np.ndarray,
    observed: np.ndarray,
    strata: np.ndarray,
    columns: list[int],
    repetitions: int,
    rng: np.random.Generator,
    prediction_minimum: float,
) -> np.ndarray:
    """Return mean squared error per row over block-permutation repetitions."""

    rows = len(values)
    stacked = np.tile(values, (repetitions, 1))
    unique_strata = pd.unique(strata)
    for repetition in range(repetitions):
        block = stacked[repetition * rows : (repetition + 1) * rows]
        for stratum in unique_strata:
            target_rows = np.flatnonzero(strata == stratum)
            donor_rows = rng.permutation(target_rows)
            block[np.ix_(target_rows, columns)] = values[np.ix_(donor_rows, columns)]
    prediction = estimator.predict(stacked).reshape(repetitions, rows)
    prediction = np.maximum(prediction, prediction_minimum)
    return np.square(prediction - observed[None, :]).mean(axis=0)


def collect_permutation_rows(
    repetitions: int, random_seed: int
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, dict[str, list[int]]]]:
    predictions = pd.read_csv(REPORTING / "predictions.csv")
    historical = predictions.loc[
        (predictions["model_family"] == "xgboost")
        & (predictions["suite"] == "historical")
    ].copy()
    model_paths = sorted(RUN.glob("cells/xgboost__fold_*__seed_*/model.joblib"))
    if len(model_paths) != 15:
        raise ValueError(f"Expected 15 XGBoost artifacts, found {len(model_paths)}")
    first_adapter = load_adapter(model_paths[0])
    _raw, features, names, key_index = prepare_inputs(first_adapter)
    values = features[list(names)].to_numpy(np.float32)
    groups = grouped_columns(names)
    records: list[pd.DataFrame] = []

    for model_number, model_path in enumerate(model_paths):
        cell = model_path.parent.name
        _, fold_text, seed_text = cell.split("__")
        fold = int(fold_text.removeprefix("fold_"))
        model_seed = int(seed_text.removeprefix("seed_"))
        adapter = load_adapter(model_path)
        adapter.estimator.set_params(device="cpu", n_jobs=4)
        adapter.estimator.get_booster().set_param({"device": "cpu", "nthread": 4})
        held = historical.loc[
            (historical["outer_fold"] == fold)
            & (historical["model_seed"] == model_seed)
        ].reset_index(drop=True)
        positions = key_index.get_indexer(
            pd.MultiIndex.from_arrays([held["uav_id"], held["cutoff"]])
        )
        if (positions < 0).any():
            raise ValueError(f"Historical endpoint missing for {cell}")
        held_values = values[positions]
        observed = held["observed_rul"].to_numpy(float)
        baseline_prediction = np.maximum(
            adapter.estimator.predict(held_values), adapter.prediction_minimum
        )
        # CPU and GPU tree traversal differ only in floating-point summation;
        # the accepted tolerance is one thousandth of an RUL cycle.
        if not np.allclose(baseline_prediction, held["predicted_rul"], atol=1e-3):
            raise ValueError(f"Saved prediction mismatch for {cell}")
        baseline_se = np.square(baseline_prediction - observed)
        strata = held["scenario"].to_numpy()

        for type_number, (group_type, definitions) in enumerate(groups.items()):
            for group_number, (group, columns) in enumerate(sorted(definitions.items())):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [random_seed, model_number, type_number, group_number]
                    )
                )
                permuted_se = permuted_prediction_error(
                    adapter.estimator,
                    held_values,
                    observed,
                    strata,
                    columns,
                    repetitions,
                    rng,
                    adapter.prediction_minimum,
                )
                frame = held[
                    [
                        "uav_id",
                        "cutoff",
                        "scenario",
                        "endpoint_seed",
                        "observed_rul",
                        "outer_fold",
                        "model_seed",
                    ]
                ].copy()
                frame["group_type"] = group_type
                frame["group"] = group
                frame["feature_count"] = len(columns)
                frame["baseline_squared_error"] = baseline_se
                frame["permuted_squared_error"] = permuted_se
                records.append(frame)
        print(f"Completed {cell} ({model_number + 1}/{len(model_paths)})", flush=True)
    return pd.concat(records, ignore_index=True), names, groups


def bootstrap_rmse_delta(
    uav_rows: pd.DataFrame, repetitions: int, seed: int
) -> tuple[float, float]:
    base = uav_rows["baseline_squared_error"].to_numpy(float)
    permuted = uav_rows["permuted_squared_error"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uav_rows), size=(repetitions, len(uav_rows)))
    deltas = np.sqrt(permuted[draws].mean(axis=1)) - np.sqrt(base[draws].mean(axis=1))
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def summarize(
    rows: pd.DataFrame, bootstrap_repetitions: int, random_seed: int
) -> pd.DataFrame:
    # Average repeated perturbations, model seeds, and historical scenarios for
    # each UAV before resampling.  This keeps the inferential unit equal to UAV.
    uav = (
        rows.groupby(
            ["group_type", "group", "feature_count", "uav_id"], as_index=False
        )[["baseline_squared_error", "permuted_squared_error"]]
        .mean()
    )
    y = rows.drop_duplicates(
        ["uav_id", "cutoff", "scenario", "endpoint_seed", "observed_rul"]
    )["observed_rul"].to_numpy(float)
    target_variance = float(np.mean(np.square(y - y.mean())))
    results: list[dict[str, float | int | str]] = []
    for group_number, ((group_type, group, feature_count), subset) in enumerate(
        uav.groupby(["group_type", "group", "feature_count"], sort=True)
    ):
        baseline_mse = float(subset["baseline_squared_error"].mean())
        permuted_mse = float(subset["permuted_squared_error"].mean())
        low, high = bootstrap_rmse_delta(
            subset,
            bootstrap_repetitions,
            random_seed + group_number,
        )
        results.append(
            {
                "group_type": group_type,
                "group": group,
                "feature_count": int(feature_count),
                "uavs": int(subset["uav_id"].nunique()),
                "baseline_rmse": float(np.sqrt(baseline_mse)),
                "permuted_rmse": float(np.sqrt(permuted_mse)),
                "rmse_increase": float(np.sqrt(permuted_mse) - np.sqrt(baseline_mse)),
                "rmse_increase_ci_lower": low,
                "rmse_increase_ci_upper": high,
                "r2_drop": float((permuted_mse - baseline_mse) / target_variance),
            }
        )
    result = pd.DataFrame(results)
    result["permutation_rank"] = result.groupby("group_type")["rmse_increase"].rank(
        method="min", ascending=False
    )
    return result.sort_values(["group_type", "permutation_rank", "group"])


def attach_shap(summary: pd.DataFrame) -> pd.DataFrame:
    path = REPORTING / "figures_3" / "xgboost_historical_feature_influence.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Run plot_engineered_run_10_xgboost_explanations.py before this study"
        )
    features = pd.read_csv(path)
    total = float(features["mean_abs_contribution"].sum())
    shap_frames = []
    for group_type, column in (
        ("feature_family", "feature_family"),
        ("sensor", "sensor_family"),
    ):
        grouped = (
            features.groupby(column, as_index=False)["mean_abs_contribution"]
            .sum()
            .rename(columns={column: "group"})
        )
        grouped["group_type"] = group_type
        grouped["shap_share_percent"] = 100 * grouped["mean_abs_contribution"] / total
        grouped["shap_rank"] = grouped["mean_abs_contribution"].rank(
            method="min", ascending=False
        )
        shap_frames.append(grouped)
    shap = pd.concat(shap_frames, ignore_index=True)
    result = summary.merge(shap, on=["group_type", "group"], validate="one_to_one")
    result["consensus_rank"] = result[["permutation_rank", "shap_rank"]].mean(axis=1)
    return result.sort_values(["group_type", "consensus_rank", "group"])


def plot_permutation(summary: pd.DataFrame) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.0), gridspec_kw={"wspace": 0.48})
    selections = [
        ("feature_family", "Feature transformation", BLUE, None),
        ("sensor", "Sensor or flight-age group", GREEN, 12),
    ]
    for ax, (group_type, title, color, limit) in zip(axes, selections):
        subset = summary.loc[summary["group_type"] == group_type].copy()
        if limit is not None:
            subset = subset.nlargest(limit, "rmse_increase")
        subset = subset.sort_values("rmse_increase")
        low = subset["rmse_increase"] - subset["rmse_increase_ci_lower"]
        high = subset["rmse_increase_ci_upper"] - subset["rmse_increase"]
        ax.barh(subset["group"], subset["rmse_increase"], color=color, alpha=0.9)
        ax.errorbar(
            subset["rmse_increase"],
            subset["group"],
            xerr=np.vstack([low.clip(lower=0), high.clip(lower=0)]),
            fmt="none",
            ecolor="#333333",
            elinewidth=1.0,
            capsize=2,
        )
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Historical RMSE increase after permutation [cycles]")
        ax.set_ylabel("")
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.savefig(OUTPUT / "historical_grouped_permutation_importance.png")
    plt.close(fig)


def plot_consensus(summary: pd.DataFrame) -> None:
    configure_style()
    subset = summary.loc[summary["group_type"] == "sensor"].copy()
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    ax.scatter(
        subset["shap_share_percent"],
        subset["rmse_increase"],
        s=60,
        color=GREEN,
        edgecolors="white",
        linewidths=0.5,
    )
    labels = subset.nsmallest(10, "consensus_rank")
    for row in labels.itertuples(index=False):
        ax.annotate(
            row.group,
            (row.shap_share_percent, row.rmse_increase),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=9,
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Share of absolute TreeSHAP contribution [%]")
    ax.set_ylabel("Historical RMSE increase after permutation [cycles]")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.tight_layout()
    fig.savefig(OUTPUT / "historical_shap_permutation_consensus.png")
    plt.close(fig)


def write_protocol(arguments: argparse.Namespace, names: tuple[str, ...]) -> None:
    protocol = {
        "status": "exploratory_post_hoc_screening",
        "source_run": "run_10",
        "model_family": "xgboost",
        "suite": "historical",
        "features": len(names),
        "outer_folds": 5,
        "model_seeds": [13, 37, 73],
        "permutation_repetitions": arguments.permutations,
        "permutation_strata": "historical scenario",
        "bootstrap_repetitions": arguments.bootstraps,
        "bootstrap_unit": "complete UAV",
        "random_seed": arguments.seed,
        "interpretation": (
            "TreeSHAP measures model use; grouped permutation measures predictive "
            "dependence under perturbation. Neither establishes causality. Confirm "
            "feature removal with training-only ranking and grouped outer-fold retraining."
        ),
    }
    (OUTPUT / "study_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.permutations < 2 or args.bootstraps < 100:
        raise ValueError("Use at least 2 permutations and 100 UAV bootstrap draws")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, names, _groups = collect_permutation_rows(args.permutations, args.seed)
    rows.to_csv(OUTPUT / "historical_grouped_permutation_endpoint_errors.csv", index=False)
    result = attach_shap(summarize(rows, args.bootstraps, args.seed))
    result.to_csv(OUTPUT / "historical_feature_importance_consensus.csv", index=False)
    plot_permutation(result)
    plot_consensus(result)
    write_protocol(args, names)
    print(f"Wrote feature-importance screening study to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
