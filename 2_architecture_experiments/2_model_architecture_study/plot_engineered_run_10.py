"""Create presentation-ready figures for Architecture Study Run 10.

The script reads only the completed reporting artifacts under ``runs/run_10``
and writes PNG figures to ``runs/run_10/reporting/figures``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_REPORTING = HERE / "runs" / "run_10" / "reporting"

DARK_BLUE = "#0072B2"
LIGHT_BLUE = "#56B4E9"
ORANGE = "#D55E00"
GREEN = "#009E73"
YELLOW = "#E69F00"
GRAY = "#7A7A7A"
RED = "#C44E52"
LIGHT_GRAY = "#D9D9D9"

DISPLAY = {
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "multiscale_cnn": "Multi-scale CNN",
    "trajectory_dtw_knn": "Trajectory DTW-kNN",
    "mlp": "MLP",
}

COLORS = {
    "xgboost": DARK_BLUE,
    "lstm": LIGHT_BLUE,
    "transformer": GREEN,
    "multiscale_cnn": ORANGE,
    "trajectory_dtw_knn": YELLOW,
    "mlp": GRAY,
}

SUITE_COLORS = {
    "historical": LIGHT_BLUE,
    "nominal": DARK_BLUE,
    "unrestricted": ORANGE,
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def style_axis(axis: plt.Axes, axis_name: str = "both") -> None:
    axis.grid(True, axis=axis_name, alpha=0.22, color=GRAY, linewidth=0.8)
    axis.set_axisbelow(True)


def save(figure: plt.Figure, output: Path, name: str) -> Path:
    path = output / name
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def nominal_comparison(summary: pd.DataFrame, output: Path) -> Path:
    table = summary.loc[summary["suite"].eq("nominal")].sort_values("rmse")
    figure, axis = plt.subplots(figsize=(13.4, 7.4), constrained_layout=True)
    y = np.arange(len(table))
    bars = axis.barh(
        y,
        table["rmse"],
        color=[COLORS[value] for value in table["model_family"]],
        height=0.64,
    )
    axis.set_yticks(y, [DISPLAY[value] for value in table["model_family"]])
    axis.invert_yaxis()
    axis.set_xlabel("RMSE in flight cycles (lower is better)")
    axis.set_xlim(0, float(table["rmse"].max()) * 1.28)
    style_axis(axis, "x")
    for bar, row in zip(bars, table.itertuples(), strict=True):
        axis.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"RMSE {row.rmse:.2f}   R² {row.r2:.3f}",
            va="center",
            fontsize=11,
            color="#303030",
        )
    return save(figure, output, "run10_nominal_architecture_comparison.png")


def suite_robustness(summary: pd.DataFrame, output: Path) -> Path:
    suites = ["historical", "nominal", "unrestricted"]
    order = (
        summary.loc[summary["suite"].eq("nominal")]
        .sort_values("rmse")["model_family"]
        .tolist()
    )
    figure, axes = plt.subplots(1, 2, figsize=(14.4, 7.2), constrained_layout=True)
    x = np.arange(len(suites))
    for model in order:
        rows = summary.loc[summary["model_family"].eq(model)].set_index("suite").loc[suites]
        axes[0].plot(
            x,
            rows["rmse"],
            marker="o",
            markersize=7,
            linewidth=2.4 if model == "xgboost" else 1.7,
            color=COLORS[model],
            label=DISPLAY[model],
        )
        axes[1].plot(
            x,
            rows["r2"],
            marker="o",
            markersize=7,
            linewidth=2.4 if model == "xgboost" else 1.7,
            color=COLORS[model],
            label=DISPLAY[model],
        )
    for axis in axes:
        axis.set_xticks(x, [value.capitalize() for value in suites])
        style_axis(axis, "y")
    axes[0].set_ylabel("RMSE in flight cycles")
    axes[0].set_title("Absolute error", loc="left", weight="bold")
    axes[1].set_ylabel("R²")
    axes[1].axhline(0, color=GRAY, linewidth=1, linestyle="--")
    axes[1].set_title("Explained variance", loc="left", weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    return save(figure, output, "run10_suite_robustness.png")


def paired_forest(paired: pd.DataFrame, output: Path) -> Path:
    models = [
        "lstm",
        "transformer",
        "multiscale_cnn",
        "trajectory_dtw_knn",
        "mlp",
    ]
    suites = ["historical", "nominal", "unrestricted"]
    offsets = {"historical": -0.22, "nominal": 0.0, "unrestricted": 0.22}
    y = np.arange(len(models))
    figure, axis = plt.subplots(figsize=(13.6, 7.4), constrained_layout=True)
    axis.axvline(0, color="#303030", linewidth=1.2)
    for suite in suites:
        rows = paired.loc[paired["suite"].eq(suite)].set_index("model_family").loc[models]
        center = rows["rmse_delta_vs_xgboost"].to_numpy()
        lower = center - rows["ci_lower"].to_numpy()
        upper = rows["ci_upper"].to_numpy() - center
        axis.errorbar(
            center,
            y + offsets[suite],
            xerr=np.vstack([lower, upper]),
            fmt="o",
            capsize=3,
            markersize=7,
            linewidth=1.7,
            color=SUITE_COLORS[suite],
            label=suite.capitalize(),
        )
    axis.set_yticks(y, [DISPLAY[value] for value in models])
    axis.invert_yaxis()
    axis.set_xlabel("Paired RMSE difference versus XGBoost (flight cycles)")
    style_axis(axis, "x")
    axis.legend(frameon=False, ncol=3, loc="upper right")
    return save(figure, output, "run10_paired_vs_xgboost.png")


def nominal_fold_stability(folds: pd.DataFrame, summary: pd.DataFrame, output: Path) -> Path:
    table = folds.loc[folds["suite"].eq("nominal")].copy()
    table = table.groupby(["model_family", "outer_fold"], as_index=False)["rmse"].mean()
    order = (
        summary.loc[summary["suite"].eq("nominal")]
        .sort_values("rmse")["model_family"]
        .tolist()
    )
    figure, axis = plt.subplots(figsize=(13.6, 7.4), constrained_layout=True)
    positions = np.arange(len(order))
    values = [table.loc[table["model_family"].eq(model), "rmse"].to_numpy() for model in order]
    boxes = axis.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.5},
        whiskerprops={"color": GRAY},
        capprops={"color": GRAY},
    )
    for patch, model in zip(boxes["boxes"], order, strict=True):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.72)
    for position, model, vals in zip(positions, order, values, strict=True):
        jitter = np.linspace(-0.12, 0.12, len(vals))
        axis.scatter(position + jitter, vals, color="#202020", s=24, zorder=3)
    axis.set_xticks(positions, [DISPLAY[value] for value in order], rotation=18, ha="right")
    axis.set_ylabel("Fold RMSE in flight cycles")
    style_axis(axis, "y")
    return save(figure, output, "run10_nominal_fold_stability.png")


def prediction_panels(predictions: pd.DataFrame, summary: pd.DataFrame, output: Path) -> Path:
    data = predictions.loc[predictions["suite"].eq("nominal")].copy()
    keys = ["uav_id", "cutoff", "scenario", "endpoint_seed", "observed_rul", "model_family"]
    data = data.groupby(keys, as_index=False)["predicted_rul"].mean()
    order = (
        summary.loc[summary["suite"].eq("nominal")]
        .sort_values("rmse")["model_family"]
        .tolist()
    )
    nominal = summary.loc[summary["suite"].eq("nominal")].set_index("model_family")
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.1), constrained_layout=True, sharex=True, sharey=True)
    limits = (0, max(float(data["observed_rul"].max()), float(data["predicted_rul"].max())) * 1.02)
    for axis, model in zip(axes.flat, order, strict=True):
        rows = data.loc[data["model_family"].eq(model)]
        axis.scatter(
            rows["observed_rul"],
            rows["predicted_rul"],
            s=14,
            alpha=0.38,
            color=COLORS[model],
            edgecolors="none",
        )
        axis.plot(limits, limits, color="#303030", linestyle="--", linewidth=1)
        metrics = nominal.loc[model]
        axis.set_title(f"{DISPLAY[model]}\nRMSE {metrics.rmse:.2f} · R² {metrics.r2:.3f}", weight="bold")
        style_axis(axis)
    for axis in axes[-1, :]:
        axis.set_xlabel("Observed RUL")
    for axis in axes[:, 0]:
        axis.set_ylabel("Predicted RUL")
    axes.flat[0].set_xlim(limits)
    axes.flat[0].set_ylim(limits)
    return save(figure, output, "run10_nominal_prediction_panels.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reporting", type=Path, default=DEFAULT_REPORTING)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reporting = args.reporting.resolve()
    output = (args.output or reporting / "figures").resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style()

    summary = pd.read_csv(reporting / "summary.csv")
    folds = pd.read_csv(reporting / "fold_metrics.csv")
    paired = pd.read_csv(reporting / "paired_vs_xgboost.csv")
    predictions = pd.read_csv(reporting / "predictions.csv")

    paths = [
        nominal_comparison(summary, output),
        suite_robustness(summary, output),
        paired_forest(paired, output),
        nominal_fold_stability(folds, summary, output),
        prediction_panels(predictions, summary, output),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
