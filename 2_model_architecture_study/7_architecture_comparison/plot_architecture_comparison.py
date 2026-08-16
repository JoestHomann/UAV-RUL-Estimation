"""Create readable comparison figures without sorting families by performance."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
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


def _display_name(family: str) -> str:
    """Turn a registry key into a compact plot label."""

    special = {
        "cycle_only_baseline": "Cycle-only",
        "mean_baseline": "Mean",
        "regularized_linear": "Regularized linear",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "mlp": "MLP",
        "tcn": "TCN",
        "lstm": "LSTM",
        "transformer": "Transformer",
        "rbf_svr": "RBF-SVR",
    }
    return special.get(family, family.replace("_", " ").title())


def _family_colors(families: tuple[str, ...]) -> dict[str, tuple[float, ...]]:
    """Assign stable, visibly distinct colors in settings order."""

    palette = plt.get_cmap("tab10")
    return {family: palette(index % 10) for index, family in enumerate(families)}


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
    colors = _family_colors(families)
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
    """Show every pairwise RMSE difference in one symmetric matrix."""

    families = plan.enabled_families
    family_index = {family: index for index, family in enumerate(families)}
    matrix = np.zeros((len(families), len(families)), dtype=float)
    rmse_rows = paired.loc[paired["metric"] == "rmse"]
    for row in rmse_rows.itertuples(index=False):
        first = family_index[row.family_a]
        second = family_index[row.family_b]
        matrix[first, second] = float(row.difference_a_minus_b)
        matrix[second, first] = -float(row.difference_a_minus_b)

    maximum = float(np.max(np.abs(matrix)))
    if maximum <= 0.0:
        maximum = 1.0
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(
        matrix,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
    )
    labels = [_display_name(family) for family in families]
    positions = np.arange(len(families))
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Column architecture")
    axis.set_ylabel("Row architecture")
    for row_index in positions:
        for column_index in positions:
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(matrix[row_index, column_index]) > 0.55 * maximum else "black",
            )
    figure.colorbar(image, ax=axis, label="Row RMSE minus column RMSE (RUL cycles)")
    axis.set_title(
        "Paired RMSE differences\nnegative values mean the row architecture has lower RMSE"
    )
    return _finish_figure(figure, output_path)


def create_comparison_figures(
    tables: "ComparisonTables",
    plan: ArchitectureComparisonPlan,
    figure_dir: Path,
) -> list[Path]:
    """Create all figures promised by the architecture-study documentation."""

    figure_dir.mkdir(parents=True, exist_ok=True)
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
    ]
    return paths
