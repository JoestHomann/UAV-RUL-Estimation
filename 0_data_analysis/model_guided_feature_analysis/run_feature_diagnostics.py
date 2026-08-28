"""Compare feature recipes with fixed XGBoost and ExtraTrees development fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
PHASE_1_ROOT = REPOSITORY_ROOT / "1_dataset_construction"
if str(PHASE_1_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_1_ROOT))

from feature_recipes import catalog_feature_sets  # noqa: E402


DEFAULT_FEATURE_ROOT = (
    PHASE_1_ROOT / "5_prefix_feature_engineering" / "artifacts"
)
DEFAULT_CATALOG = (
    PHASE_1_ROOT / "6_feature_sets" / "artifacts" / "feature_catalog.csv"
)
DEFAULT_FOLDS = (
    PHASE_1_ROOT
    / "2_UAV_grouped_validation_folds"
    / "artifacts"
    / "outer_folds.csv"
)
DEFAULT_ANOMALIES = (
    REPOSITORY_ROOT
    / "0_data_analysis"
    / "core_data_analysis"
    / "figures"
    / "anomalies"
    / "uav_anomaly_priority.csv"
)
DEFAULT_TRAIN = REPOSITORY_ROOT / "data" / "train.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "artifacts"
MODEL_NAMES = ("xgboost", "extra_trees")
METADATA_COLUMNS = {
    "sample_id",
    "scenario",
    "outer_fold",
    "uav_id",
    "cutoff",
    "RUL",
    "terminal_lifetime",
    "lifetime_quantile",
    "prefix_number",
    "sample_weight",
}

XGBOOST_PARAMETERS: dict[str, Any] = {
    "n_estimators": 342,
    "max_depth": 8,
    "learning_rate": 0.011845262553864248,
    "min_child_weight": 0.9955971345967498,
    "subsample": 0.7752736396167055,
    "colsample_bytree": 0.6986003693464361,
    "reg_alpha": 0.3547372178764013,
    "reg_lambda": 1.6417773645720573,
}
EXTRA_TREES_PARAMETERS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 20,
    "max_features": 0.67,
    "min_samples_leaf": 5,
}


def parse_parameter_overrides(value: str | None, option: str) -> dict[str, Any]:
    """Decode model overrides passed by the experiment launcher."""

    if value is None:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{option} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{option} must decode to a JSON object")
    if "random_state" in payload:
        raise ValueError(f"{option} must use --seed instead of random_state")
    return payload


def effective_model_parameters(
    name: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge editable run settings with the documented diagnostic defaults."""

    if name == "extra_trees":
        parameters = {**EXTRA_TREES_PARAMETERS, "n_jobs": 1}
    elif name == "xgboost":
        parameters = {
            **XGBOOST_PARAMETERS,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cpu",
            "n_jobs": 1,
        }
    else:
        raise ValueError(f"Unknown model {name!r}")
    parameters.update(overrides or {})
    return parameters


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def make_model(name: str, seed: int, parameters: dict[str, Any]) -> Any:
    if name == "extra_trees":
        return ExtraTreesRegressor(
            **parameters,
            random_state=seed,
        )
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            **parameters,
            random_state=seed,
        )
    raise ValueError(f"Unknown model {name!r}")


def feature_block(feature_name: str) -> str:
    body = feature_name.removeprefix("feature__")
    if body in {"flight_cycle", "log1p_flight_cycle"}:
        return "age"
    statistic = body.split("__", maxsplit=1)[1]
    if "_minus_w" in statistic or "_minus_history_" in statistic:
        return "acceleration"
    if statistic.startswith("state_"):
        return "state"
    if any(
        token in statistic
        for token in ("median", "iqr", "mad", "q10", "q90", "robust_z")
    ):
        return "robust"
    if statistic.startswith("w5_"):
        return "window_5"
    if statistic.startswith("w20_"):
        return "window_20"
    if statistic.startswith("w50_"):
        return "window_50"
    if any(token in statistic for token in ("slope", "delta", "abs_delta")):
        return "change"
    if statistic in {"last", "first", "baseline_mean"}:
        return "level_baseline"
    return "history_distribution"


def feature_channel(feature_name: str) -> str:
    body = feature_name.removeprefix("feature__")
    return "age" if "__" not in body else body.split("__", maxsplit=1)[0]


def grouped_columns(feature_names: list[str]) -> dict[tuple[str, str], list[int]]:
    groups: dict[tuple[str, str], list[int]] = {}
    for index, name in enumerate(feature_names):
        for group_type, group_name in (
            ("feature_block", feature_block(name)),
            ("channel", feature_channel(name)),
        ):
            groups.setdefault((group_type, group_name), []).append(index)
    return groups


def scenario_permutation(metadata: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    permutation = np.arange(len(metadata))
    if "scenario" not in metadata:
        return rng.permutation(permutation)
    for _, indices in metadata.groupby("scenario", sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        permutation[positions] = rng.permutation(positions)
    return permutation


def permutation_records(
    *,
    model: Any,
    model_name: str,
    feature_set: str,
    outer_fold: int,
    features: np.ndarray,
    feature_names: list[str],
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    baseline_prediction: np.ndarray,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    baseline_rmse = rmse(y_true, baseline_prediction)
    records: list[dict[str, Any]] = []
    for group_number, ((group_type, group_name), columns) in enumerate(
        sorted(grouped_columns(feature_names).items()),
        start=1,
    ):
        scores: list[float] = []
        for repetition in range(repetitions):
            rng = np.random.default_rng(
                seed + outer_fold * 10_000 + group_number * 100 + repetition
            )
            permutation = scenario_permutation(metadata, rng)
            permuted = features.copy()
            permuted[:, columns] = features[permutation][:, columns]
            scores.append(rmse(y_true, np.maximum(0.0, model.predict(permuted))))
        records.append(
            {
                "model_family": model_name,
                "feature_set": feature_set,
                "outer_fold": outer_fold,
                "group_type": group_type,
                "group_name": group_name,
                "feature_count": len(columns),
                "baseline_rmse": baseline_rmse,
                "permuted_rmse": float(np.mean(scores)),
                "delta_rmse": float(np.mean(scores) - baseline_rmse),
                "repetitions": repetitions,
            }
        )
    return records


def regression_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    y_true = group["y_true"].to_numpy(dtype=float)
    y_pred = group["y_pred"].to_numpy(dtype=float)
    residual = y_pred - y_true
    denominator = float(np.square(y_true - y_true.mean()).sum())
    return {
        "rows": len(group),
        "uavs": int(group["uav_id"].nunique()),
        "rmse": rmse(y_true, y_pred),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r2": 1.0 - float(np.square(residual).sum()) / denominator
        if denominator > 0
        else float("nan"),
    }


def summarize_residuals(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    base = ["model_family", "feature_set"]
    for keys, group in predictions.groupby(base, sort=True):
        records.append(
            {
                "model_family": keys[0],
                "feature_set": keys[1],
                "group_type": "overall",
                "group_value": "all",
                **regression_metrics(group),
            }
        )
    for group_column in ("cutoff_band", "anomaly_quartile"):
        for keys, group in predictions.groupby(
            [*base, group_column],
            sort=True,
            observed=True,
        ):
            records.append(
                {
                    "model_family": keys[0],
                    "feature_set": keys[1],
                    "group_type": group_column,
                    "group_value": str(keys[2]),
                    **regression_metrics(group),
                }
            )
    return pd.DataFrame.from_records(records)


def residual_channel_correlations(
    predictions: pd.DataFrame,
    train_csv: Path,
) -> pd.DataFrame:
    raw = pd.read_csv(train_csv)
    telemetry = [column for column in raw if column.startswith("telemetry_")]
    endpoints = raw[["uav_id", "flight_cycle", *telemetry]].rename(
        columns={"flight_cycle": "cutoff"}
    )
    merged = predictions.merge(
        endpoints,
        on=["uav_id", "cutoff"],
        how="left",
        validate="many_to_one",
    )
    records: list[dict[str, Any]] = []
    for (model, feature_set), group in merged.groupby(
        ["model_family", "feature_set"], sort=True
    ):
        for channel in telemetry:
            correlation = (
                float(group["residual"].corr(group[channel], method="spearman"))
                if group[channel].nunique(dropna=True) > 1
                else float("nan")
            )
            records.append(
                {
                    "model_family": model,
                    "feature_set": feature_set,
                    "channel": channel,
                    "spearman_residual_correlation": correlation,
                }
            )
    return pd.DataFrame.from_records(records)


def residual_agreement(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for feature_set, group in predictions.groupby("feature_set", sort=True):
        left = group.loc[
            group["model_family"] == "xgboost",
            ["sample_id", "y_true", "y_pred", "residual"],
        ]
        right = group.loc[
            group["model_family"] == "extra_trees",
            ["sample_id", "y_pred", "residual"],
        ]
        if left.empty or right.empty:
            continue
        paired = left.merge(right, on="sample_id", suffixes=("_xgb", "_extra"))
        ensemble = 0.5 * (paired["y_pred_xgb"] + paired["y_pred_extra"])
        records.append(
            {
                "feature_set": feature_set,
                "rows": len(paired),
                "residual_pearson": float(
                    paired["residual_xgb"].corr(paired["residual_extra"])
                ),
                "xgboost_rmse": rmse(
                    paired["y_true"].to_numpy(), paired["y_pred_xgb"].to_numpy()
                ),
                "extra_trees_rmse": rmse(
                    paired["y_true"].to_numpy(), paired["y_pred_extra"].to_numpy()
                ),
                "equal_weight_ensemble_rmse": rmse(
                    paired["y_true"].to_numpy(), ensemble.to_numpy()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def feature_drift(
    training: pd.DataFrame,
    test: pd.DataFrame,
    catalog: pd.DataFrame,
    feature_sets: list[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for feature_set in feature_sets:
        names = catalog.loc[catalog[feature_set].astype(bool), "feature_name"].tolist()
        for name in names:
            train_values = training[name].to_numpy(dtype=float)
            test_values = test[name].to_numpy(dtype=float)
            median = float(np.median(train_values))
            q25, q75 = np.quantile(train_values, [0.25, 0.75])
            scale = float((q75 - q25) / 1.349)
            if scale <= 1e-12:
                scale = float(np.std(train_values, ddof=1))
            if scale <= 1e-12:
                scale = 1.0
            records.append(
                {
                    "feature_set": feature_set,
                    "feature_name": name,
                    "channel": feature_channel(name),
                    "feature_block": feature_block(name),
                    "train_median": median,
                    "test_median": float(np.median(test_values)),
                    "standardized_median_shift": float(
                        (np.median(test_values) - median) / scale
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def plot_cutoff_metrics(metrics: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    subset = metrics.loc[metrics["group_type"] == "cutoff_band"].copy()
    labels = subset["feature_set"].drop_duplicates().tolist()
    models = subset["model_family"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        len(labels),
        1,
        figsize=(9, max(3.2 * len(labels), 4)),
        squeeze=False,
        constrained_layout=True,
    )
    bands = ["1-50", "51-100", "101-200", ">200"]
    width = 0.8 / len(models)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2.0) * width
    for axis, feature_set in zip(axes[:, 0], labels, strict=True):
        frame = subset.loc[subset["feature_set"] == feature_set]
        x = np.arange(len(bands))
        for offset, model in zip(offsets, models, strict=True):
            values = (
                frame.loc[frame["model_family"] == model]
                .set_index("group_value")
                .reindex(bands)["rmse"]
            )
            axis.bar(x + offset, values, width=width * 0.9, label=model)
        axis.set_title(feature_set)
        axis.set_ylabel("RMSE")
        axis.set_xticks(x, bands)
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    path = output_dir / "development_rmse_by_cutoff.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_importance(importances: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    summary = (
        importances.loc[importances["group_type"] == "feature_block"]
        .groupby(["model_family", "feature_set", "group_name"], as_index=False)[
            "delta_rmse"
        ]
        .mean()
        .sort_values("delta_rmse", ascending=False)
        .head(24)
        .sort_values("delta_rmse")
    )
    labels = (
        summary["model_family"]
        + " | "
        + summary["feature_set"]
        + " | "
        + summary["group_name"]
    )
    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    axis.barh(labels, summary["delta_rmse"], color="#0072B2")
    axis.axvline(0.0, color="#444444", linewidth=1)
    axis.set_xlabel("Permutation RMSE increase")
    axis.grid(axis="x", alpha=0.2)
    path = output_dir / "grouped_permutation_importance.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-features",
        type=Path,
        default=DEFAULT_FEATURE_ROOT / "training_features.csv.gz",
    )
    parser.add_argument(
        "--development-features",
        type=Path,
        default=DEFAULT_FEATURE_ROOT / "development_validation_features.csv.gz",
    )
    parser.add_argument(
        "--test-features",
        type=Path,
        default=DEFAULT_FEATURE_ROOT / "test_features.csv.gz",
    )
    parser.add_argument("--feature-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--outer-folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--anomaly-priority", type=Path, default=DEFAULT_ANOMALIES)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--feature-sets", nargs="+")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--xgboost-parameters-json")
    parser.add_argument("--extra-trees-parameters-json")
    parser.add_argument("--permutation-repetitions", type=int, default=3)
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.permutation_repetitions <= 0:
        raise ValueError("--permutation-repetitions must be positive")
    model_parameters = {
        "xgboost": effective_model_parameters(
            "xgboost",
            parse_parameter_overrides(
                args.xgboost_parameters_json,
                "--xgboost-parameters-json",
            ),
        ),
        "extra_trees": effective_model_parameters(
            "extra_trees",
            parse_parameter_overrides(
                args.extra_trees_parameters_json,
                "--extra-trees-parameters-json",
            ),
        ),
    }

    training = pd.read_csv(args.training_features)
    development = pd.read_csv(args.development_features)
    test = pd.read_csv(args.test_features)
    catalog = pd.read_csv(args.feature_catalog)
    available_sets = catalog_feature_sets(catalog)
    feature_sets = args.feature_sets or (
        [
            name
            for name in (
                "screened_v1",
                "screened_robust",
                "screened_acceleration",
                "screened_compact",
            )
            if name in available_sets
        ]
        or ["screened"]
    )
    unknown = sorted(set(feature_sets) - set(available_sets))
    if unknown:
        raise ValueError(f"Unknown feature sets {unknown}; available: {available_sets}")

    folds = pd.read_csv(args.outer_folds)
    fold_by_uav = folds.set_index("uav_id")["outer_fold"]
    training_fold = training["uav_id"].map(fold_by_uav)
    if training_fold.isna().any():
        raise ValueError("Training features contain UAVs absent from outer folds")
    if "outer_fold" not in development:
        development = development.copy()
        development["outer_fold"] = development["uav_id"].map(fold_by_uav)

    prediction_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    for feature_set in feature_sets:
        names = catalog.loc[catalog[feature_set].astype(bool), "feature_name"].tolist()
        for outer_fold in sorted(folds["outer_fold"].unique()):
            train_rows = training.loc[training_fold != outer_fold]
            validation_rows = development.loc[
                development["outer_fold"].astype(int) == int(outer_fold)
            ].reset_index(drop=True)
            x_train = train_rows[names].to_numpy(dtype=float)
            y_train = train_rows["RUL"].to_numpy(dtype=float)
            weights = train_rows["sample_weight"].to_numpy(dtype=float)
            x_validation = validation_rows[names].to_numpy(dtype=float)
            y_validation = validation_rows["RUL"].to_numpy(dtype=float)
            for model_name in args.models:
                model = make_model(
                    model_name,
                    args.seed,
                    model_parameters[model_name],
                )
                model.fit(x_train, y_train, sample_weight=weights)
                prediction = np.maximum(0.0, model.predict(x_validation))
                frame = validation_rows[
                    [
                        "sample_id",
                        "scenario",
                        "outer_fold",
                        "uav_id",
                        "cutoff",
                        "terminal_lifetime",
                        "lifetime_quantile",
                    ]
                ].copy()
                frame.insert(0, "feature_set", feature_set)
                frame.insert(0, "model_family", model_name)
                frame["y_true"] = y_validation
                frame["y_pred"] = prediction
                frame["residual"] = prediction - y_validation
                prediction_frames.append(frame)
                if not args.skip_permutation:
                    importance_rows.extend(
                        permutation_records(
                            model=model,
                            model_name=model_name,
                            feature_set=feature_set,
                            outer_fold=int(outer_fold),
                            features=x_validation,
                            feature_names=names,
                            metadata=validation_rows,
                            y_true=y_validation,
                            baseline_prediction=prediction,
                            repetitions=args.permutation_repetitions,
                            seed=args.seed,
                        )
                    )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["cutoff_band"] = pd.cut(
        predictions["cutoff"],
        bins=[0, 50, 100, 200, np.inf],
        labels=["1-50", "51-100", "101-200", ">200"],
        include_lowest=True,
    )
    anomaly = pd.read_csv(args.anomaly_priority)
    anomaly = anomaly.loc[anomaly["split"] == "train", ["uav_id", "anomaly_priority_score"]]
    anomaly = anomaly.drop_duplicates("uav_id")
    anomaly["anomaly_quartile"] = pd.qcut(
        anomaly["anomaly_priority_score"].rank(method="first"),
        4,
        labels=["Q1-low", "Q2", "Q3", "Q4-high"],
    )
    predictions = predictions.merge(anomaly, on="uav_id", how="left", validate="many_to_one")

    metrics = summarize_residuals(predictions)
    correlations = residual_channel_correlations(predictions, args.train_csv)
    agreement = residual_agreement(predictions)
    drift = feature_drift(training, test, catalog, feature_sets)
    importances = pd.DataFrame.from_records(importance_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        args.output_dir / "development_residual_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    metrics.to_csv(args.output_dir / "residual_metrics.csv", index=False)
    correlations.to_csv(
        args.output_dir / "residual_channel_correlations.csv", index=False
    )
    agreement.to_csv(args.output_dir / "model_residual_agreement.csv", index=False)
    drift.to_csv(args.output_dir / "feature_drift.csv", index=False)
    if not importances.empty:
        importances.to_csv(
            args.output_dir / "grouped_permutation_importance.csv", index=False
        )
        plot_importance(importances, args.output_dir, args.dpi)
    plot_cutoff_metrics(metrics, args.output_dir, args.dpi)

    manifest = {
        "status": "complete",
        "models": list(args.models),
        "feature_sets": feature_sets,
        "outer_folds": int(folds["outer_fold"].nunique()),
        "development_scenarios": int(development["scenario"].nunique()),
        "locked_data_loaded": False,
        "test_targets_loaded": False,
        "model_seed": args.seed,
        "permutation_repetitions": (
            0 if args.skip_permutation else args.permutation_repetitions
        ),
        "fixed_model_parameters": {
            name: model_parameters[name] for name in args.models
        },
    }
    (args.output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved model-guided feature diagnostics under {args.output_dir}")


if __name__ == "__main__":
    main()
