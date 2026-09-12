"""Create presentation-ready XGBoost explanations for Run 10.

The plots use only grouped outer-fold predictions from the historical validation
suite.  Per-feature contributions are calculated by XGBoost's native TreeSHAP
implementation and then averaged across the three registered model seeds for
each held-out endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from xgboost import DMatrix


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = HERE / "runs" / "run_10"
REPORTING = RUN / "reporting"
OUTPUT = REPORTING / "figures_3"

sys.path.insert(0, str(HERE / "4_model_adapters"))
from engineered_feature_data import load_script  # noqa: E402


BLUE = "#0072B2"
LIGHT_BLUE = "#56B4E9"
ORANGE = "#D55E00"
GREEN = "#009E73"
YELLOW = "#E69F00"
GRAY = "#7A7A7A"
GRID = "#D9D9D9"


def configure_style() -> None:
    """Apply the visual language used by the other Run 10 figures."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def load_adapter(path: Path):
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "adapter" not in artifact:
        raise ValueError(f"Unexpected model artifact: {path}")
    return artifact["adapter"]


def human_feature(name: str) -> str:
    if name == "flight_cycle":
        return "Flight cycle"
    if name == "flight_cycle_log":
        return "log(1 + flight cycle)"
    sensor, *suffix = name.split("__", maxsplit=1)
    sensor_label = sensor.replace("telemetry_", "Telemetry ")
    if not suffix:
        return f"{sensor_label} — current"
    translations = {
        "baseline_delta": "change from first cycle",
        "hist_mean": "history mean",
        "hist_std": "history SD",
        "last_minus_hist_mean": "current − history mean",
        "hist_slope": "history slope",
        "roll5_mean": "5-cycle mean",
        "roll5_std": "5-cycle SD",
        "roll10_mean": "10-cycle mean",
        "roll10_std": "10-cycle SD",
        "roll20_mean": "20-cycle mean",
        "roll20_std": "20-cycle SD",
    }
    return f"{sensor_label} — {translations.get(suffix[0], suffix[0])}"


def feature_family(name: str) -> str:
    if name.startswith("flight_cycle"):
        return "Flight age"
    if "__" not in name:
        return "Current value"
    suffix = name.split("__", maxsplit=1)[1]
    if suffix == "baseline_delta":
        return "Change from first cycle"
    if suffix == "hist_mean":
        return "History mean"
    if suffix == "hist_std":
        return "History variability"
    if suffix == "last_minus_hist_mean":
        return "Deviation from history"
    if suffix == "hist_slope":
        return "History slope"
    if suffix.endswith("_mean") and suffix.startswith("roll"):
        return "Rolling means"
    if suffix.endswith("_std") and suffix.startswith("roll"):
        return "Rolling variability"
    return "Other"


def sensor_family(name: str) -> str:
    if name.startswith("flight_cycle"):
        return "Flight age"
    return name.split("__", maxsplit=1)[0].replace("telemetry_", "Telemetry ")


def prepare_inputs(first_adapter):
    settings = json.loads((HERE / "engineered_run_10_settings.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(ROOT / settings["raw_training"])
    raw = raw.sort_values(["uav_id", "flight_cycle"]).reset_index(drop=True)
    script = load_script(ROOT / settings["feature_script"])
    features = script.build_features(
        raw.drop(columns="RUL"), list(first_adapter.engineered_sensor_columns)
    )
    names = tuple(column for column in features.columns if column != "uav_id")
    expected = tuple(first_adapter.engineered_feature_schema)
    if names != expected:
        raise ValueError("Reconstructed feature schema differs from the fitted model")
    if not features[["uav_id", "flight_cycle"]].equals(raw[["uav_id", "flight_cycle"]]):
        raise ValueError("Reconstructed feature rows are misaligned")
    key_index = pd.MultiIndex.from_frame(raw[["uav_id", "flight_cycle"]])
    return raw, features, names, key_index


def collect_explanations():
    predictions = pd.read_csv(REPORTING / "predictions.csv")
    historical = predictions.loc[
        (predictions["model_family"] == "xgboost")
        & (predictions["suite"] == "historical")
    ].copy()
    model_paths = sorted(RUN.glob("cells/xgboost__fold_*__seed_*/model.joblib"))
    if not model_paths:
        raise FileNotFoundError("No saved Run 10 XGBoost models found")

    first_adapter = load_adapter(model_paths[0])
    raw, features, names, key_index = prepare_inputs(first_adapter)
    values = features[list(names)].to_numpy(np.float32)
    row_records: list[pd.DataFrame] = []
    model_importance: list[np.ndarray] = []

    for model_path in model_paths:
        cell = model_path.parent.name
        parts = cell.split("__")
        fold = int(parts[1].removeprefix("fold_"))
        seed = int(parts[2].removeprefix("seed_"))
        adapter = load_adapter(model_path)
        if tuple(adapter.feature_names) != names:
            raise ValueError(f"Feature schema changed in {cell}")
        if adapter.fault_mode_strategy != "none" or adapter.hyperparameters.get(
            "signal_compression_strategy"
        ) != "none":
            raise ValueError(f"Unsupported Run 10 transform in {cell}")

        held = historical.loc[
            (historical["outer_fold"] == fold)
            & (historical["model_seed"] == seed)
        ].copy()
        positions = key_index.get_indexer(
            pd.MultiIndex.from_arrays([held["uav_id"], held["cutoff"]])
        )
        if (positions < 0).any():
            raise ValueError(f"Historical endpoint missing from raw histories for {cell}")
        held_values = values[positions]
        booster = adapter.estimator.get_booster()
        contributions = booster.predict(DMatrix(held_values), pred_contribs=True)
        if contributions.shape != (len(held), len(names) + 1):
            raise ValueError(f"Unexpected contribution matrix for {cell}")
        raw_prediction = contributions.sum(axis=1)
        direct_prediction = adapter.estimator.predict(held_values)
        # TreeSHAP's summation order can differ slightly from the prediction
        # kernel on GPU-trained boosters.  The observed difference is below
        # one thousandth of a cycle and is only floating-point accumulation.
        if not np.allclose(raw_prediction, direct_prediction, atol=1e-3):
            raise ValueError(f"TreeSHAP contributions do not sum to predictions for {cell}")
        final_prediction = np.maximum(direct_prediction, adapter.prediction_minimum)
        if not np.allclose(final_prediction, held["predicted_rul"], atol=2e-4):
            raise ValueError(f"Reconstructed predictions differ from reporting output for {cell}")

        explained = held[
            [
                "uav_id",
                "cutoff",
                "scenario",
                "suite",
                "endpoint_seed",
                "observed_rul",
                "outer_fold",
                "model_seed",
            ]
        ].reset_index(drop=True)
        explained["baseline"] = contributions[:, -1]
        explained["raw_prediction"] = raw_prediction
        explained["final_prediction"] = final_prediction
        explained = pd.concat(
            [
                explained,
                pd.DataFrame(contributions[:, :-1], columns=names),
            ],
            axis=1,
        )
        row_records.append(explained)
        model_importance.append(np.abs(contributions[:, :-1]).mean(axis=0))

    seed_rows = pd.concat(row_records, ignore_index=True)
    contribution_columns = list(names)
    endpoint_keys = [
        "uav_id",
        "cutoff",
        "scenario",
        "suite",
        "endpoint_seed",
        "observed_rul",
        "outer_fold",
    ]
    endpoint_rows = (
        seed_rows.groupby(endpoint_keys, as_index=False)[
            ["baseline", "raw_prediction", "final_prediction", *contribution_columns]
        ]
        .mean()
        .sort_values(["outer_fold", "uav_id"])
        .reset_index(drop=True)
    )
    endpoint_rows["residual"] = endpoint_rows["final_prediction"] - endpoint_rows["observed_rul"]
    endpoint_rows["absolute_error"] = endpoint_rows["residual"].abs()

    model_matrix = np.vstack(model_importance)
    importance = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_contribution": np.abs(seed_rows[contribution_columns]).mean(axis=0).to_numpy(),
            "model_mean_abs_contribution": model_matrix.mean(axis=0),
            "model_sd_abs_contribution": model_matrix.std(axis=0, ddof=1),
        }
    ).sort_values("mean_abs_contribution", ascending=False)
    importance["feature_label"] = importance["feature"].map(human_feature)
    importance["feature_family"] = importance["feature"].map(feature_family)
    importance["sensor_family"] = importance["feature"].map(sensor_family)
    return importance, endpoint_rows, names


def plot_global_features(importance: pd.DataFrame) -> None:
    top = importance.head(15).sort_values("mean_abs_contribution")
    fig, ax = plt.subplots(figsize=(10.6, 7.2))
    ax.barh(
        top["feature_label"],
        top["mean_abs_contribution"],
        color=BLUE,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Mean absolute contribution to predicted RUL [cycles]")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.025)
    fig.tight_layout()
    fig.savefig(OUTPUT / "xgboost_historical_global_feature_influence.png")
    plt.close(fig)


def plot_group_influence(importance: pd.DataFrame) -> None:
    total = importance["mean_abs_contribution"].sum()
    family = (
        importance.groupby("feature_family", as_index=False)["mean_abs_contribution"]
        .sum()
        .assign(share=lambda frame: 100 * frame["mean_abs_contribution"] / total)
        .sort_values("share")
    )
    sensor = (
        importance.groupby("sensor_family", as_index=False)["mean_abs_contribution"]
        .sum()
        .assign(share=lambda frame: 100 * frame["mean_abs_contribution"] / total)
        .nlargest(12, "share")
        .sort_values("share")
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.7), gridspec_kw={"wspace": 0.48})
    axes[0].barh(family["feature_family"], family["share"], color=BLUE)
    axes[0].set_xlabel("Share of total absolute contribution [%]")
    axes[0].set_ylabel("")
    axes[0].set_title("Feature transformation")
    axes[0].grid(axis="y", visible=False)
    axes[1].barh(sensor["sensor_family"], sensor["share"], color=GREEN)
    axes[1].set_xlabel("Share of total absolute contribution [%]")
    axes[1].set_ylabel("")
    axes[1].set_title("Sensor or flight-age group")
    axes[1].grid(axis="y", visible=False)
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.margins(y=0.03)
    fig.tight_layout()
    fig.savefig(OUTPUT / "xgboost_historical_group_influence.png")
    plt.close(fig)


def local_case_data(row: pd.Series, names: tuple[str, ...], count: int = 8) -> pd.DataFrame:
    values = pd.Series({name: float(row[name]) for name in names})
    ordered = values.abs().nlargest(count).index
    result = pd.DataFrame({"feature": ordered, "contribution": values.loc[ordered].to_numpy()})
    remainder = float(values.drop(ordered).sum())
    result = pd.concat(
        [
            result,
            pd.DataFrame({"feature": ["All other features"], "contribution": [remainder]}),
        ],
        ignore_index=True,
    )
    result["label"] = result["feature"].map(
        lambda value: value if value == "All other features" else human_feature(value)
    )
    return result


def plot_local_cases(endpoint_rows: pd.DataFrame, names: tuple[str, ...]) -> None:
    accurate = endpoint_rows.loc[endpoint_rows["absolute_error"].idxmin()]
    failure = endpoint_rows.loc[endpoint_rows["absolute_error"].idxmax()]
    cases = [("Accurate held-out example", accurate), ("Largest-error held-out example", failure)]
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.8), gridspec_kw={"wspace": 0.65})
    for ax, (title, row) in zip(axes, cases):
        local = local_case_data(row, names).sort_values("contribution")
        colors = np.where(local["contribution"] >= 0, BLUE, ORANGE)
        ax.barh(local["label"], local["contribution"], color=colors)
        ax.axvline(0, color="#333333", linewidth=0.9)
        ax.set_xlabel("Contribution to predicted RUL [cycles]")
        ax.set_ylabel("")
        details = (
            f"{row['uav_id']} at cycle {int(row['cutoff'])}  |  "
            f"baseline {row['baseline']:.1f} → prediction {row['final_prediction']:.1f}  |  "
            f"observed {row['observed_rul']:.1f}"
        )
        ax.set_title(f"{title}\n{details}", fontsize=11, pad=10)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.tight_layout()
    fig.savefig(OUTPUT / "xgboost_historical_local_explanations.png")
    plt.close(fig)


def plot_dependence(
    importance: pd.DataFrame, endpoint_rows: pd.DataFrame, names: tuple[str, ...]
) -> None:
    top_names = importance.head(3)["feature"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), gridspec_kw={"wspace": 0.34})
    for ax, name in zip(axes, top_names):
        # Recover raw endpoint values from the fitted schema reconstruction.
        # They are joined below in main before this plotting function is called.
        x = endpoint_rows[f"value::{name}"].to_numpy(float)
        y = endpoint_rows[name].to_numpy(float)
        cycles = endpoint_rows["cutoff"].to_numpy(float)
        scatter = ax.scatter(
            x,
            y,
            c=cycles,
            cmap="viridis",
            s=34,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.35,
        )
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(human_feature(name))
        ax.set_xlabel("Feature value")
        ax.set_ylabel("Contribution [RUL cycles]")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    colorbar = fig.colorbar(scatter, ax=axes, shrink=0.86, pad=0.025)
    colorbar.set_label("Flight cycle")
    fig.savefig(OUTPUT / "xgboost_historical_feature_effects.png")
    plt.close(fig)


def attach_raw_endpoint_values(endpoint_rows: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    first_path = sorted(RUN.glob("cells/xgboost__fold_*__seed_*/model.joblib"))[0]
    adapter = load_adapter(first_path)
    _raw, features, reconstructed_names, key_index = prepare_inputs(adapter)
    if tuple(reconstructed_names) != names:
        raise ValueError("Feature schema changed while attaching endpoint values")
    positions = key_index.get_indexer(
        pd.MultiIndex.from_arrays([endpoint_rows["uav_id"], endpoint_rows["cutoff"]])
    )
    if (positions < 0).any():
        raise ValueError("Endpoint feature value missing")
    endpoint_values = features.iloc[positions][list(names)].reset_index(drop=True)
    endpoint_values.columns = [f"value::{name}" for name in names]
    return pd.concat([endpoint_rows.reset_index(drop=True), endpoint_values], axis=1)


def write_supporting_tables(
    importance: pd.DataFrame, endpoint_rows: pd.DataFrame, names: tuple[str, ...]
) -> None:
    importance.to_csv(OUTPUT / "xgboost_historical_feature_influence.csv", index=False)
    family = (
        importance.groupby("feature_family", as_index=False)["mean_abs_contribution"]
        .sum()
        .sort_values("mean_abs_contribution", ascending=False)
    )
    family["share_percent"] = 100 * family["mean_abs_contribution"] / family[
        "mean_abs_contribution"
    ].sum()
    sensor = (
        importance.groupby("sensor_family", as_index=False)["mean_abs_contribution"]
        .sum()
        .sort_values("mean_abs_contribution", ascending=False)
    )
    sensor["share_percent"] = 100 * sensor["mean_abs_contribution"] / sensor[
        "mean_abs_contribution"
    ].sum()
    family.to_csv(OUTPUT / "xgboost_historical_feature_family_influence.csv", index=False)
    sensor.to_csv(OUTPUT / "xgboost_historical_sensor_influence.csv", index=False)

    local_columns = [
        "uav_id",
        "cutoff",
        "scenario",
        "suite",
        "endpoint_seed",
        "observed_rul",
        "outer_fold",
        "baseline",
        "raw_prediction",
        "final_prediction",
        "residual",
        "absolute_error",
        *names,
    ]
    endpoint_rows[local_columns].to_csv(
        OUTPUT / "xgboost_historical_endpoint_contributions.csv", index=False
    )


def main() -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    importance, endpoint_rows, names = collect_explanations()
    endpoint_rows = attach_raw_endpoint_values(endpoint_rows, names)
    plot_global_features(importance)
    plot_group_influence(importance)
    plot_local_cases(endpoint_rows, names)
    plot_dependence(importance, endpoint_rows, names)
    write_supporting_tables(importance, endpoint_rows, names)
    print(f"Wrote XGBoost explanation figures to {OUTPUT}")


if __name__ == "__main__":
    main()
