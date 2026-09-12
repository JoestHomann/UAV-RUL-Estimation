"""Plot Run 10 historical-validation metrics by observed-cycle band."""

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

MODEL_ORDER = [
    "xgboost",
    "lstm",
    "transformer",
    "multiscale_cnn",
    "trajectory_dtw_knn",
    "mlp",
]

BAND_ORDER = ["1-50", "51-100", "101-200", ">200"]
BAND_LABELS = ["1–50", "51–100", "101–200", ">200"]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def historical_endpoint_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average repeated model seeds before computing endpoint-level metrics."""
    historical = predictions.loc[predictions["suite"].eq("historical")].copy()
    keys = [
        "uav_id",
        "cutoff",
        "scenario",
        "endpoint_seed",
        "observed_rul",
        "model_family",
    ]
    historical = historical.groupby(keys, as_index=False)["predicted_rul"].mean()
    historical["cycle_band"] = pd.cut(
        historical["cutoff"],
        bins=[0, 50, 100, 200, np.inf],
        labels=BAND_ORDER,
        ordered=True,
    )
    if historical["cycle_band"].isna().any():
        raise ValueError("A historical endpoint lies outside the declared cycle bands")
    return historical


def calculate_metrics(historical: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, band), group in historical.groupby(
        ["model_family", "cycle_band"], observed=True, sort=False
    ):
        error = group["predicted_rul"].to_numpy(float) - group["observed_rul"].to_numpy(float)
        observed = group["observed_rul"].to_numpy(float)
        denominator = np.square(observed - observed.mean()).sum()
        rows.append(
            {
                "model_family": model,
                "cycle_band": str(band),
                "endpoints": len(group),
                "uavs": group["uav_id"].nunique(),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "r2": float(1.0 - np.square(error).sum() / denominator),
                "bias": float(np.mean(error)),
                "residual_variance": float(np.var(error, ddof=0)),
            }
        )
    metrics = pd.DataFrame(rows)
    expected = {(model, band) for model in MODEL_ORDER for band in BAND_ORDER}
    observed = set(zip(metrics["model_family"], metrics["cycle_band"], strict=True))
    if observed != expected:
        raise ValueError("Historical metric table does not contain every model/band pair")
    return metrics


def plot_metric(
    metrics: pd.DataFrame,
    output: Path,
    column: str,
    ylabel: str,
    filename: str,
    *,
    zero_line: bool = False,
) -> Path:
    figure, axis = plt.subplots(figsize=(13.4, 7.4))
    x = np.arange(len(BAND_ORDER))
    counts = (
        metrics.loc[metrics["model_family"].eq("xgboost")]
        .set_index("cycle_band")
        .loc[BAND_ORDER, "endpoints"]
        .astype(int)
        .tolist()
    )
    for model in MODEL_ORDER:
        rows = metrics.loc[metrics["model_family"].eq(model)].set_index("cycle_band").loc[BAND_ORDER]
        axis.plot(
            x,
            rows[column],
            color=COLORS[model],
            marker="o",
            markersize=8,
            linewidth=2.8 if model == "xgboost" else 2.0,
            label=DISPLAY[model],
        )
    if zero_line:
        axis.axhline(0, color="#303030", linewidth=1, linestyle="--")
    axis.set_xticks(
        x,
        [f"{label}\nn={count}" for label, count in zip(BAND_LABELS, counts, strict=True)],
    )
    axis.set_xlabel("Observed flight-cycle band")
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", color=GRAY, alpha=0.22, linewidth=0.8)
    axis.set_axisbelow(True)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.23)
    path = output / filename
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def historical_architecture_comparison(summary: pd.DataFrame, output: Path) -> Path:
    table = summary.loc[summary["suite"].eq("historical")].sort_values("rmse")
    figure, axis = plt.subplots(figsize=(13.4, 7.4))
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
    axis.grid(True, axis="x", color=GRAY, alpha=0.22, linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, row in zip(bars, table.itertuples(), strict=True):
        axis.text(
            bar.get_width() + 0.7,
            bar.get_y() + bar.get_height() / 2,
            f"RMSE {row.rmse:.2f}   R² {row.r2:.3f}",
            va="center",
            fontsize=11,
            color="#303030",
        )
    figure.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.13)
    path = output / "run10_historical_architecture_comparison.png"
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reporting", type=Path, default=DEFAULT_REPORTING)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reporting = args.reporting.resolve()
    output = (args.output or reporting / "figures_2").resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style()

    predictions = pd.read_csv(reporting / "predictions.csv")
    summary = pd.read_csv(reporting / "summary.csv")
    historical = historical_endpoint_predictions(predictions)
    metrics = calculate_metrics(historical)
    metrics.to_csv(output / "historical_metrics_by_cycle_band.csv", index=False)

    paths = [
        historical_architecture_comparison(summary, output),
        plot_metric(metrics, output, "r2", "R²", "historical_cycle_band_r2.png", zero_line=True),
        plot_metric(
            metrics,
            output,
            "rmse",
            "RMSE in flight cycles",
            "historical_cycle_band_rmse.png",
        ),
        plot_metric(
            metrics,
            output,
            "bias",
            "Bias: predicted − observed RUL (cycles)",
            "historical_cycle_band_bias.png",
            zero_line=True,
        ),
        plot_metric(
            metrics,
            output,
            "residual_variance",
            "Residual variance (flight cycles²)",
            "historical_cycle_band_variance.png",
        ),
    ]
    for path in paths:
        print(path)
    print(output / "historical_metrics_by_cycle_band.csv")


if __name__ == "__main__":
    main()
