"""Create readable comparison figures without sorting families by performance."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from comparison_gate import ArchitectureComparisonPlan

if TYPE_CHECKING:
    from architecture_comparison import ComparisonTables


METRIC_LABELS = {
    "r2": "R2",
    "rmse": "RMSE (RUL cycles)",
    "mae": "MAE (RUL cycles)",
    "bias": "Bias (predicted - observed RUL)",
}

OFFSET_CYCLES = (0.0, 3.0, 6.0, 10.0)
RUL_BAND_LABELS = ("0-25", "26-50", "51-100", "above_100")
RUL_BAND_DISPLAY_LABELS = ("0-25", "26-50", "51-100", ">100")


def _display_name(family: str) -> str:
    """Turn a registry key into a compact plot label."""

    special = {
        "cycle_only_baseline": "Cycle-only",
        "mean_baseline": "Mean",
        "regularized_linear": "Regularized linear",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "xgboost": "XGBoost",
        "catboost": "CatBoost",
        "mlp": "MLP",
        "tcn": "TCN",
        "multiscale_cnn": "Multi-scale CNN",
        "sensor_graph_tcn": "Sensor-graph TCN",
        "lstm": "LSTM",
        "transformer": "Transformer",
        "rbf_svr": "RBF-SVR",
        "trajectory_dtw_knn": "Trajectory DTW-kNN",
    }
    return special.get(family, family.replace("_", " ").title())


def _family_colors(families: tuple[str, ...]) -> dict[str, tuple[float, ...]]:
    """Assign stable, visibly distinct colors in settings order."""

    palette = plt.get_cmap("tab10")
    return {family: palette(index % 10) for index, family in enumerate(families)}


def _diagnostic_families(plan: ArchitectureComparisonPlan) -> tuple[str, ...]:
    """Prefer fitted models when a baseline would flatten diagnostic scales."""

    fitted = tuple(
        family
        for family in plan.enabled_families
        if family not in {"mean_baseline", "cycle_only_baseline"}
    )
    return fitted or plan.enabled_families


def _finish_figure(figure: plt.Figure, path: Path) -> Path:
    """Apply final spacing, save one PNG, and release its memory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def _overall_metrics_figure(
    comparison: pd.DataFrame,
    plan: ArchitectureComparisonPlan,
    output_path: Path,
) -> Path:
    """Show all primary and secondary metrics with paired-UAV uncertainty."""

    families = plan.enabled_families
    labels = [_display_name(family) for family in families]
    colors = _family_colors(plan.enabled_families)
    table = comparison.set_index("model_family").loc[list(families)]
    x_positions = np.arange(len(families))
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))

    for axis, metric in zip(axes.flat, ("rmse", "mae", "r2", "bias"), strict=True):
        centers = table[f"{metric}_mean"].to_numpy(float)
        lower = table[f"{metric}_ci_lower_95"].to_numpy(float)
        upper = table[f"{metric}_ci_upper_95"].to_numpy(float)
        errors = np.vstack(
            [np.maximum(centers - lower, 0.0), np.maximum(upper - centers, 0.0)]
        )
        axis.errorbar(
            x_positions,
            centers,
            yerr=errors,
            fmt="none",
            ecolor="#4b5563",
            elinewidth=1.5,
            capsize=4,
            zorder=1,
        )
        axis.scatter(
            x_positions,
            centers,
            s=65,
            c=[colors[family] for family in families],
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )
        if metric == "bias":
            axis.axhline(0.0, color="#6b7280", linewidth=1.0, linestyle="--")
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xticks(x_positions, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)

    figure.suptitle(
        "Locked outer-validation performance\n"
        "points are means across retained seeds; bars are 95% paired UAV-bootstrap intervals",
        fontsize=14,
    )
    return _finish_figure(figure, output_path)


def _reliability_figure(
    grouped: pd.DataFrame,
    plan: ArchitectureComparisonPlan,
    group_type: str,
    title: str,
    output_path: Path,
) -> Path:
    """Plot RMSE and bias across one predefined reliability grouping."""

    families = plan.enabled_families
    colors = _family_colors(families)
    selected = grouped.loc[grouped["group_type"] == group_type]
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for family in families:
        family_rows = selected.loc[selected["model_family"] == family].sort_values(
            "group_position",
            kind="stable",
        )
        x_positions = family_rows["group_position"].to_numpy(int)
        for axis, metric in zip(axes, ("rmse", "bias"), strict=True):
            center = family_rows[f"{metric}_mean"].to_numpy(float)
            spread = family_rows[f"{metric}_seed_sd"].to_numpy(float)
            axis.plot(
                x_positions,
                center,
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=colors[family],
                label=_display_name(family),
            )
            if np.any(spread > 0):
                axis.fill_between(
                    x_positions,
                    center - spread,
                    center + spread,
                    color=colors[family],
                    alpha=0.10,
                    linewidth=0,
                )

    labels_source = (
        selected.loc[selected["model_family"] == families[0]]
        .sort_values("group_position", kind="stable")
    )
    labels = labels_source["group_value"].astype(str).tolist()
    positions = labels_source["group_position"].to_numpy(int)
    axes[0].set_ylabel(METRIC_LABELS["rmse"])
    axes[1].set_ylabel(METRIC_LABELS["bias"])
    axes[1].axhline(0.0, color="#6b7280", linewidth=1.0, linestyle="--")
    axes[1].set_xticks(positions, labels, rotation=30 if len(labels) > 8 else 0)
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=9, loc="best")
    figure.suptitle(
        f"{title}\nlines show seed means; shaded bands show plus or minus one seed SD",
        fontsize=14,
    )
    return _finish_figure(figure, output_path)


def _seed_stability_figure(
    seed_metrics: pd.DataFrame,
    plan: ArchitectureComparisonPlan,
    output_path: Path,
) -> Path:
    """Retain every configured seed so no best seed can be selected silently."""

    families = plan.enabled_families
    labels = [_display_name(family) for family in families]
    colors = _family_colors(families)
    x_positions = np.arange(len(families))
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))

    for family_index, family in enumerate(families):
        rows = seed_metrics.loc[seed_metrics["model_family"] == family]
        offsets = np.linspace(-0.12, 0.12, len(rows)) if len(rows) > 1 else np.array([0.0])
        for axis, metric in zip(axes, ("rmse", "r2"), strict=True):
            values = rows[metric].to_numpy(float)
            axis.scatter(
                family_index + offsets,
                values,
                s=55,
                color=colors[family],
                edgecolor="black",
                linewidth=0.4,
                zorder=2,
            )
            axis.plot(
                [family_index - 0.18, family_index + 0.18],
                [np.mean(values), np.mean(values)],
                color="black",
                linewidth=1.2,
                zorder=3,
            )

    for axis, metric in zip(axes, ("rmse", "r2"), strict=True):
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xticks(x_positions, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Random-seed stability\ncolored points are individual seeds; black segments are seed means",
        fontsize=14,
    )
    return _finish_figure(figure, output_path)


def _efficiency_figure(
    comparison: pd.DataFrame,
    efficiency: pd.DataFrame,
    plan: ArchitectureComparisonPlan,
    output_path: Path,
) -> Path:
    """Keep measured cost visible without combining it into an overall score."""

    families = plan.enabled_families
    colors = _family_colors(families)
    merged = efficiency.merge(
        comparison[["model_family", "rmse_mean"]],
        on="model_family",
        validate="one_to_one",
    ).set_index("model_family").loc[list(families)]
    panels = (
        ("training_seconds_mean_per_run", "Mean training time per run (s)"),
        ("inference_milliseconds_per_endpoint", "Inference time per endpoint (ms)"),
        ("serialized_model_bytes_mean", "Mean serialized model size (bytes)"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(19, 6))

    for axis, (column, x_label) in zip(axes, panels, strict=True):
        x_values = merged[column].to_numpy(float)
        y_values = merged["rmse_mean"].to_numpy(float)
        axis.scatter(
            x_values,
            y_values,
            s=70,
            c=[colors[family] for family in families],
            edgecolor="black",
            linewidth=0.5,
        )
        for family, x_value, y_value in zip(
            families,
            x_values,
            y_values,
            strict=True,
        ):
            axis.annotate(
                _display_name(family),
                (x_value, y_value),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
        finite_positive = x_values[np.isfinite(x_values) & (x_values > 0)]
        if len(finite_positive) == len(x_values) and np.max(finite_positive) / np.min(
            finite_positive
        ) >= 50:
            axis.set_xscale("log")
        axis.set_xlabel(x_label)
        axis.set_ylabel(METRIC_LABELS["rmse"])
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Predictive performance and computational cost\n"
        "cost dimensions remain separate and are not converted into a ranking",
        fontsize=14,
    )
    return _finish_figure(figure, output_path)


def _paired_rmse_figure(
    paired: pd.DataFrame,
    plan: ArchitectureComparisonPlan,
    output_path: Path,
) -> Path:
    """Show every pairwise RMSE difference, and which of them exclude zero.

    The point difference on its own cannot say whether a pair is separated,
    so a cell whose 95% paired UAV-bootstrap interval excludes zero is boxed
    and its value is printed in bold. Everything unboxed is a difference this
    evaluation cannot distinguish from no difference at all.
    """

    families = plan.enabled_families
    family_index = {family: index for index, family in enumerate(families)}
    size = len(families)
    matrix = np.zeros((size, size), dtype=float)
    separated = np.zeros((size, size), dtype=bool)
    rmse_rows = paired.loc[paired["metric"] == "rmse"]
    for row in rmse_rows.itertuples(index=False):
        first = family_index[row.family_a]
        second = family_index[row.family_b]
        difference = float(row.difference_a_minus_b)
        matrix[first, second] = difference
        matrix[second, first] = -difference
        excludes_zero = bool(
            float(row.ci_lower_95) > 0.0 or float(row.ci_upper_95) < 0.0
        )
        separated[first, second] = excludes_zero
        separated[second, first] = excludes_zero

    # One badly diverged family would otherwise set the colour range and flatten
    # every difference among the remaining architectures to the same near-white.
    # Twice the median absolute difference is used instead: the median has a 50%
    # breakdown point, so the scale stays meaningful until half of all pairs are
    # extreme, and no fixed quantile has to be tuned to a particular study. The
    # printed values remain exact, so a saturated cell hides nothing.
    off_diagonal = np.abs(matrix[~np.eye(size, dtype=bool)])
    maximum = float(np.max(off_diagonal)) if off_diagonal.size else 0.0
    scale = 2.0 * float(np.median(off_diagonal)) if off_diagonal.size else 0.0
    if scale <= 0.0 or scale >= maximum:
        scale = maximum if maximum > 0.0 else 1.0
    clipped = maximum > scale

    figure, axis = plt.subplots(figsize=(11.5, 9))
    image = axis.imshow(
        matrix,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale),
    )
    labels = [_display_name(family) for family in families]
    positions = np.arange(size)
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Column architecture")
    axis.set_ylabel("Row architecture")
    for row_index in positions:
        for column_index in positions:
            difference = matrix[row_index, column_index]
            is_separated = separated[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{difference:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold" if is_separated else "normal",
                color="white" if abs(difference) > 0.55 * scale else "black",
            )
            if is_separated:
                axis.add_patch(
                    Rectangle(
                        (column_index - 0.5, row_index - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.6,
                    )
                )
    figure.colorbar(image, ax=axis, label="Row RMSE minus column RMSE (RUL cycles)")
    subtitle = (
        "negative values mean the row architecture has lower RMSE; "
        "boxed cells have a 95% interval that excludes zero"
    )
    if clipped:
        subtitle += (
            f"\ncolour saturates at {scale:.0f} cycles so the largest "
            f"difference ({maximum:.0f}) does not flatten the rest"
        )
    axis.set_title(f"Paired RMSE differences\n{subtitle}", fontsize=11)
    return _finish_figure(figure, output_path)


def _r2_bar_figure(
    comparison: pd.DataFrame,
    plan: ArchitectureComparisonPlan,
    output_path: Path,
) -> Path:
    """Show R2 scores as a horizontal bar chart."""

    families = plan.enabled_families
    labels = [_display_name(family) for family in families]
    colors = _family_colors(families)
    table = comparison.set_index("model_family").loc[list(families)]
    r2_means = table["r2_mean"].to_numpy(float)

    figure, axis = plt.subplots(figsize=(10, 6))
    y_positions = np.arange(len(families))
    
    # Plot bars
    axis.barh(
        y_positions, 
        r2_means, 
        color=[colors[f] for f in families], 
        edgecolor="black", 
        linewidth=0.5
    )

    # Add text labels on or next to the bars
    for y, r2 in zip(y_positions, r2_means):
        offset = 0.01 * max(abs(r2_means)) if max(abs(r2_means)) > 0 else 0.01
        ha = "left" if r2 >= 0 else "right"
        x_pos = r2 + offset if r2 >= 0 else r2 - offset
        axis.text(
            x_pos,
            y,
            f"{r2:.3f}",
            va="center",
            ha=ha,
            fontsize=10,
        )

    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()  # Keep settings order top-to-bottom
    axis.set_xlabel("Mean R² Score")
    axis.set_title("Overall R² Score by Architecture\n(Higher is better)", fontsize=14)
    axis.axvline(0, color='#6b7280', linewidth=1.0, linestyle="--")
    axis.grid(axis="x", alpha=0.25)
    
    # Adjust limits to fit text
    x_min = min(min(r2_means) - abs(min(r2_means))*0.1 - 0.05, 0)
    x_max = max(max(r2_means) + abs(max(r2_means))*0.1 + 0.05, 0)
    axis.set_xlim(x_min, x_max)

    return _finish_figure(figure, output_path)


def _seed_averaged_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average retained seeds for each model and locked endpoint."""

    endpoint_columns = [
        "outer_fold",
        "scenario",
        "sample_id",
        "uav_id",
        "cutoff",
        "y_true",
    ]
    averaged = (
        predictions.groupby(
            ["model_family", *endpoint_columns],
            as_index=False,
            dropna=False,
            sort=False,
        )["y_pred"]
        .mean()
        .copy()
    )
    averaged["residual"] = averaged["y_pred"] - averaged["y_true"]
    return averaged


def _offset_diagnostics(
    seed_averaged: pd.DataFrame,
    prediction_minimum: float,
) -> pd.DataFrame:
    """Evaluate fixed, predeclared offsets without choosing a winner."""

    records: list[dict[str, float | str]] = []
    for family, rows in seed_averaged.groupby("model_family", sort=False):
        observed = rows["y_true"].to_numpy(float)
        predicted = rows["y_pred"].to_numpy(float)
        denominator = float(np.sum(np.square(observed - np.mean(observed))))
        for offset in OFFSET_CYCLES:
            adjusted = np.maximum(predicted - offset, prediction_minimum)
            residual = adjusted - observed
            records.append(
                {
                    "model_family": family,
                    "offset_cycles": offset,
                    "r2": (
                        float("nan")
                        if denominator <= 0.0
                        else 1.0
                        - float(np.sum(np.square(residual))) / denominator
                    ),
                    "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                    "mae": float(np.mean(np.abs(residual))),
                    "bias": float(np.mean(residual)),
                    "overprediction_rate": float(np.mean(residual > 0.0)),
                }
            )
    return pd.DataFrame.from_records(records)


def _overprediction_tail_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize positive-residual tails overall and by true-RUL band."""

    working = predictions.copy()
    working["rul_band"] = pd.cut(
        working["y_true"],
        bins=[-np.inf, 25.0, 50.0, 100.0, np.inf],
        labels=list(RUL_BAND_LABELS),
    )
    records: list[dict[str, float | int | str]] = []
    for family, family_rows in working.groupby("model_family", sort=False):
        groups = [("overall", family_rows)]
        groups.extend(
            (band, family_rows.loc[family_rows["rul_band"] == band])
            for band in RUL_BAND_LABELS
        )
        for group_name, rows in groups:
            positive = rows.loc[rows["residual"] > 0.0, "residual"]
            records.append(
                {
                    "model_family": family,
                    "group_value": group_name,
                    "positive_count": int(len(positive)),
                    "positive_p90": (
                        float(positive.quantile(0.90)) if len(positive) else np.nan
                    ),
                    "positive_p95": (
                        float(positive.quantile(0.95)) if len(positive) else np.nan
                    ),
                    "positive_maximum": (
                        float(positive.max()) if len(positive) else np.nan
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _residual_ecdf_figure(
    seed_averaged: pd.DataFrame,
    plan: ArchitectureComparisonPlan,
    output_path: Path,
) -> Path:
    """Show residual coverage and how fixed offsets move the safety threshold."""

    families = plan.enabled_families
    colors = _family_colors(families)
    figure, axis = plt.subplots(figsize=(13, 7))
    for family in families:
        residuals = np.sort(
            seed_averaged.loc[
                seed_averaged["model_family"] == family,
                "residual",
            ].to_numpy(float)
        )
        coverage = np.arange(1, len(residuals) + 1, dtype=float) / len(residuals)
        axis.plot(
            residuals,
            coverage,
            color=colors[family],
            linewidth=1.8,
            label=_display_name(family),
        )
    for offset_index, offset in enumerate(OFFSET_CYCLES):
        axis.axvline(
            offset,
            color="black" if offset == 0.0 else "#6b7280",
            linewidth=1.2,
            linestyle="-" if offset == 0.0 else "--",
            alpha=0.8,
        )
        axis.text(
            offset,
            0.015 + 0.035 * offset_index,
            f"{offset:g}",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
        )
    axis.set_xlabel("Residual threshold (predicted - observed RUL cycles)")
    axis.set_ylabel("Cumulative fraction at or below threshold")
    axis.set_xscale("symlog", linthresh=15.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(ncol=4, fontsize=9, loc="lower right")
    axis.set_title(
        "Seed-averaged residual distributions and candidate safety offsets\n"
        "the ECDF value at an offset is the resulting non-overprediction rate"
    )
    return _finish_figure(figure, output_path)


def _overprediction_metrics_figure(
    grouped: pd.DataFrame,
    plan: ArchitectureComparisonPlan,
    output_path: Path,
) -> Path:
    """Show asymmetric error frequency and magnitude overall and by RUL band."""

    families = plan.enabled_families
    colors = _family_colors(families)
    labels = [_display_name(family) for family in families]
    positions = np.arange(len(families))
    overall = (
        grouped.loc[grouped["group_type"] == "overall"]
        .set_index("model_family")
        .loc[list(families)]
    )
    by_band = grouped.loc[grouped["group_type"] == "rul_band"]
    figure, axes = plt.subplots(2, 2, figsize=(17, 11))

    for axis, column, title in (
        (axes[0, 0], "overprediction_rate_mean", "Overall overprediction rate"),
        (
            axes[0, 1],
            "root_mean_squared_overprediction_mean",
            "Overall RMS overprediction",
        ),
    ):
        bars = axis.bar(
            positions,
            overall[column].to_numpy(float),
            color=[colors[family] for family in families],
            edgecolor="black",
            linewidth=0.5,
        )
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(
            bars,
            fmt="%.3f" if column == "overprediction_rate_mean" else "%.1f",
            padding=3,
            fontsize=8,
        )
    axes[0, 0].set_ylabel("Fraction of predictions")
    axes[0, 1].set_ylabel("RUL cycles")

    for family in families:
        rows = by_band.loc[by_band["model_family"] == family].sort_values(
            "group_position",
            kind="stable",
        )
        x_values = rows["group_position"].to_numpy(int)
        axes[1, 0].plot(
            x_values,
            rows["overprediction_rate_mean"].to_numpy(float),
            marker="o",
            color=colors[family],
            label=_display_name(family),
        )
        axes[1, 1].plot(
            x_values,
            rows["root_mean_squared_overprediction_mean"].to_numpy(float),
            marker="o",
            color=colors[family],
            label=_display_name(family),
        )
    band_positions = np.arange(len(RUL_BAND_LABELS))
    for axis in axes[1]:
        axis.set_xticks(band_positions, list(RUL_BAND_DISPLAY_LABELS))
        axis.grid(alpha=0.25)
        axis.set_xlabel("True RUL band")
    axes[1, 0].set_title("Overprediction rate by true-RUL band")
    axes[1, 0].set_ylabel("Fraction of predictions")
    axes[1, 1].set_title("RMS overprediction by true-RUL band")
    axes[1, 1].set_ylabel("RUL cycles")
    axes[1, 0].legend(ncol=4, fontsize=9, loc="best")
    figure.suptitle(
        "Asymmetric locked-evaluation diagnostics\n"
        "positive residuals mean predicted RUL exceeds observed RUL",
        fontsize=14,
    )
    return _finish_figure(figure, output_path)


def _offset_tradeoff_figure(
    diagnostics: pd.DataFrame,
    plan: ArchitectureComparisonPlan,
    output_path: Path,
) -> Path:
    """Keep accuracy, bias, and safety visible across fixed offsets."""

    families = _diagnostic_families(plan)
    colors = _family_colors(plan.enabled_families)
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    panels = (
        ("r2", "R2"),
        ("rmse", "RMSE (RUL cycles)"),
        ("bias", "Bias (RUL cycles)"),
        ("overprediction_rate", "Overprediction rate"),
    )
    for family in families:
        rows = diagnostics.loc[diagnostics["model_family"] == family].sort_values(
            "offset_cycles"
        )
        for axis, (column, label) in zip(axes.flat, panels, strict=True):
            axis.plot(
                rows["offset_cycles"],
                rows[column],
                marker="o",
                color=colors[family],
                label=_display_name(family),
            )
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
    axes[1, 0].axhline(0.0, color="#6b7280", linewidth=1.0, linestyle="--")
    for axis in axes[1]:
        axis.set_xlabel("RUL cycles subtracted from prediction")
    axes[0, 0].legend(ncol=4, fontsize=9, loc="best")
    figure.suptitle(
        "Accuracy-safety tradeoff for fixed prediction offsets\n"
        "diagnostic only: offsets are predeclared and no best value is selected",
        fontsize=14,
    )
    return _finish_figure(figure, output_path)


def _positive_tail_figure(
    tails: pd.DataFrame,
    plan: ArchitectureComparisonPlan,
    output_path: Path,
) -> Path:
    """Show positive-residual P90, P95, and maximum values by family."""

    families = plan.enabled_families
    colors = _family_colors(families)
    labels = [_display_name(family) for family in families]
    overall = (
        tails.loc[tails["group_value"] == "overall"]
        .set_index("model_family")
        .loc[list(families)]
    )
    positions = np.arange(len(families))
    figure, axes = plt.subplots(1, 3, figsize=(19, 6), sharex=True)
    for axis, column, title in zip(
        axes,
        ("positive_p90", "positive_p95", "positive_maximum"),
        ("P90", "P95", "Maximum"),
        strict=True,
    ):
        bars = axis.bar(
            positions,
            overall[column].to_numpy(float),
            color=[colors[family] for family in families],
            edgecolor="black",
            linewidth=0.5,
        )
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.set_title(title)
        axis.set_ylabel("Positive residual (RUL cycles)")
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    figure.suptitle(
        "Positive-residual tail magnitude\n"
        "percentiles are conditional on predictions that overestimate RUL",
        fontsize=14,
    )
    return _finish_figure(figure, output_path)


def _prediction_offset_scatter_figure(
    seed_averaged: pd.DataFrame,
    family: str,
    prediction_minimum: float,
    output_path: Path,
) -> Path:
    """Compare observed/predicted alignment before and after a six-cycle offset."""

    rows = seed_averaged.loc[seed_averaged["model_family"] == family]
    observed = rows["y_true"].to_numpy(float)
    original = rows["y_pred"].to_numpy(float)
    adjusted = np.maximum(original - 6.0, prediction_minimum)
    lower = float(min(np.min(observed), np.min(adjusted), np.min(original)))
    upper = float(max(np.max(observed), np.max(adjusted), np.max(original)))
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for axis, predicted, title in (
        (axes[0], original, "Unadjusted"),
        (axes[1], adjusted, "Minus 6 RUL cycles"),
    ):
        axis.scatter(observed, predicted, s=12, alpha=0.25, color="#2563eb")
        axis.plot([lower, upper], [lower, upper], color="black", linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("Observed RUL")
        axis.grid(alpha=0.20)
    axes[0].set_ylabel("Predicted RUL")
    figure.suptitle(
        f"{_display_name(family)} prediction alignment\n"
        "the fixed offset is diagnostic and is not selected from this figure",
        fontsize=14,
    )
    return _finish_figure(figure, output_path)


def create_comparison_figures(
    tables: "ComparisonTables",
    plan: ArchitectureComparisonPlan,
    figure_dir: Path,
    predictions: pd.DataFrame,
) -> list[Path]:
    """Create all figures promised by the architecture-study documentation."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    seed_averaged = _seed_averaged_predictions(predictions)
    prediction_minimum = float(plan.settings["evaluation"]["prediction_minimum"])
    offset_diagnostics = _offset_diagnostics(seed_averaged, prediction_minimum)
    tail_diagnostics = _overprediction_tail_diagnostics(predictions)
    paths = [
        _overall_metrics_figure(
            tables.architecture_comparison,
            plan,
            figure_dir / "overall_metrics_with_uncertainty.png",
        ),
        _reliability_figure(
            tables.grouped_architecture_metrics,
            plan,
            "outer_fold",
            "Reliability across held-out outer UAV folds",
            figure_dir / "outer_fold_reliability.png",
        ),
        _reliability_figure(
            tables.grouped_architecture_metrics,
            plan,
            "scenario",
            "Reliability across locked validation scenarios",
            figure_dir / "scenario_reliability.png",
        ),
        _reliability_figure(
            tables.grouped_architecture_metrics,
            plan,
            "age_band",
            "Reliability across flight-cycle age bands",
            figure_dir / "age_band_reliability.png",
        ),
        _reliability_figure(
            tables.grouped_architecture_metrics,
            plan,
            "lifetime_quantile",
            "Reliability across UAV lifetime groups",
            figure_dir / "lifetime_group_reliability.png",
        ),
        _seed_stability_figure(
            tables.seed_metrics,
            plan,
            figure_dir / "seed_stability.png",
        ),
        _efficiency_figure(
            tables.architecture_comparison,
            tables.efficiency_summary,
            plan,
            figure_dir / "performance_and_efficiency.png",
        ),
        _paired_rmse_figure(
            tables.paired_metric_differences,
            plan,
            figure_dir / "paired_rmse_differences.png",
        ),
        _r2_bar_figure(
            tables.architecture_comparison,
            plan,
            figure_dir / "r2_comparison.png",
        ),
        _residual_ecdf_figure(
            seed_averaged,
            plan,
            figure_dir / "residual_ecdf_with_offsets.png",
        ),
        _overprediction_metrics_figure(
            tables.grouped_architecture_metrics,
            plan,
            figure_dir / "overprediction_diagnostics.png",
        ),
        _offset_tradeoff_figure(
            offset_diagnostics,
            plan,
            figure_dir / "offset_tradeoff.png",
        ),
        _positive_tail_figure(
            tail_diagnostics,
            plan,
            figure_dir / "positive_residual_tails.png",
        ),
    ]
    paths.extend(
        _prediction_offset_scatter_figure(
            seed_averaged,
            family,
            prediction_minimum,
            figure_dir / f"prediction_scatter_{family}.png",
        )
        for family in plan.enabled_families
    )
    return paths
