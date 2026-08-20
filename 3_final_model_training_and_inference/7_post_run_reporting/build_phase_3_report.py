"""Build model-agnostic Phase 3 figures from one completed run."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
PHASE_DIR = STEP_DIR.parent
if str(PHASE_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE_DIR))

from phase_3_common import (  # noqa: E402
    Phase3Error,
    complete_manifest,
    read_json,
    require_current_settings,
    run_root,
    selected_configuration_path,
    step_directory,
    test_predictions_path,
    write_json,
)
from phase_3_run_layout import SETTINGS_PATH, tensorboard_log_root  # noqa: E402


REPORT_VERSION = 1
REPORT_DIRECTORY_NAME = "7_post_run_reporting"

CANDIDATE_COLUMNS = {
    "candidate_number",
    "feature_set",
    "lookback",
    "mean_fold_rmse",
    "fold_rmse_standard_deviation",
    "mean_fold_r2",
    "mean_fold_mae",
    "mean_fold_bias",
    "mean_training_seconds",
    "total_training_seconds",
    "final_training_iterations",
}

FOLD_COLUMNS = {
    "candidate_number",
    "outer_fold",
    "rmse",
    "r2",
    "mae",
    "bias",
    "training_seconds",
    "best_epoch_or_iteration",
}


class Phase3ReportingError(Phase3Error):
    """Explain an incomplete or inconsistent reporting input."""


def _display_name(family: str) -> str:
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


def _read_csv(path: Path, description: str) -> pd.DataFrame:
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise Phase3ReportingError(f"Cannot read {description} at {path}: {error}") from error
    if table.empty:
        raise Phase3ReportingError(f"{description} is empty")
    return table


def _require_columns(
    table: pd.DataFrame,
    columns: set[str],
    description: str,
) -> None:
    missing = sorted(columns - set(table.columns))
    if missing:
        raise Phase3ReportingError(f"{description} is missing columns {missing}")


def _finish_figure(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _input_choices(candidates: pd.DataFrame) -> tuple[pd.Series, str]:
    feature_sets = candidates["feature_set"].dropna().astype(str)
    if not feature_sets.empty and feature_sets.str.strip().ne("").any():
        return candidates["feature_set"].fillna("fixed input").astype(str), "Feature set"
    lookbacks = pd.to_numeric(candidates["lookback"], errors="coerce")
    if lookbacks.notna().any():
        labels = lookbacks.map(
            lambda value: (
                f"Lookback {int(value)}" if pd.notna(value) else "fixed input"
            )
        )
        return labels, "Lookback"
    return pd.Series("Fixed input", index=candidates.index), "Input"


def _choice_colors(choices: pd.Series) -> dict[str, tuple[float, ...]]:
    labels = list(dict.fromkeys(choices.astype(str)))
    palette = plt.get_cmap("tab10")
    return {label: palette(index % 10) for index, label in enumerate(labels)}


def _optimization_history(
    candidates: pd.DataFrame,
    selected_number: int,
    family: str,
    path: Path,
) -> Path:
    table = candidates.sort_values("candidate_number").copy()
    choices, _ = _input_choices(table)
    colors = _choice_colors(choices)
    numbers = table["candidate_number"].to_numpy(dtype=int)
    scores = table["mean_fold_rmse"].to_numpy(dtype=float)
    running_best = np.minimum.accumulate(scores)

    figure, axis = plt.subplots(figsize=(12, 6.5))
    axis.plot(
        numbers,
        scores,
        color="#9ca3af",
        linewidth=1.0,
        alpha=0.75,
        zorder=1,
    )
    for choice in colors:
        mask = choices.eq(choice).to_numpy()
        axis.scatter(
            numbers[mask],
            scores[mask],
            s=42,
            color=colors[choice],
            edgecolor="white",
            linewidth=0.5,
            label=choice,
            zorder=2,
        )
    axis.plot(
        numbers,
        running_best,
        color="#111827",
        linewidth=2.0,
        label="Running best",
        zorder=3,
    )
    selected = table.loc[table["candidate_number"].eq(selected_number)].iloc[0]
    axis.scatter(
        [selected_number],
        [float(selected["mean_fold_rmse"])],
        marker="*",
        s=230,
        color="#e11d48",
        edgecolor="black",
        linewidth=0.7,
        label="Selected",
        zorder=4,
    )
    axis.set_title(f"{_display_name(family)} final-search trajectory")
    axis.set_xlabel("Candidate number")
    axis.set_ylabel("Mean five-fold RMSE (RUL cycles)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=min(4, len(colors) + 2))
    return _finish_figure(figure, path)


def _top_candidate_robustness(
    candidates: pd.DataFrame,
    folds: pd.DataFrame,
    selected_number: int,
    family: str,
    path: Path,
) -> Path:
    top = candidates.nsmallest(min(10, len(candidates)), "mean_fold_rmse").copy()
    top = top.sort_values("mean_fold_rmse")
    numbers = top["candidate_number"].astype(int).tolist()
    labels = [
        f"C{number:03d}" + (" selected" if number == selected_number else "")
        for number in numbers
    ]
    fold_labels = sorted(folds["outer_fold"].astype(int).unique())
    palette = plt.get_cmap("tab10")

    figure, axis = plt.subplots(figsize=(12, max(6.5, len(top) * 0.62)))
    for y_position, number in enumerate(numbers):
        rows = folds.loc[folds["candidate_number"].eq(number)].sort_values(
            "outer_fold"
        )
        for color_index, (_, row) in enumerate(rows.iterrows()):
            axis.scatter(
                float(row["rmse"]),
                y_position,
                s=34,
                color=palette(color_index % 10),
                alpha=0.85,
                zorder=2,
            )
        mean_rmse = float(
            top.loc[top["candidate_number"].eq(number), "mean_fold_rmse"].iloc[0]
        )
        axis.scatter(
            mean_rmse,
            y_position,
            marker="D",
            s=58,
            color="#111827" if number != selected_number else "#e11d48",
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("Fold RMSE (RUL cycles)")
    axis.set_title(f"{_display_name(family)} top-candidate fold robustness")
    axis.grid(axis="x", alpha=0.25)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=palette(index % 10),
            label=f"Fold {fold}",
        )
        for index, fold in enumerate(fold_labels)
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                color="#111827",
                label="Candidate mean",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                color="#e11d48",
                label="Selected mean",
            ),
        ]
    )
    axis.legend(
        handles=handles,
        frameon=False,
        ncol=min(7, len(handles)),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
    )
    return _finish_figure(figure, path)


def _fold_heatmap(
    candidates: pd.DataFrame,
    folds: pd.DataFrame,
    selected_number: int,
    family: str,
    path: Path,
) -> Path:
    ordering = candidates.sort_values("mean_fold_rmse")["candidate_number"].astype(int)
    matrix = folds.pivot(
        index="candidate_number",
        columns="outer_fold",
        values="rmse",
    ).reindex(ordering)
    if matrix.isna().any().any():
        raise Phase3ReportingError("Candidate/fold RMSE matrix is incomplete")
    labels = [
        f"C{int(number):03d}" + (" *" if int(number) == selected_number else "")
        for number in matrix.index
    ]
    height = max(7.0, min(18.0, 2.5 + 0.36 * len(matrix)))
    figure, axis = plt.subplots(figsize=(10, height))
    image = axis.imshow(matrix.to_numpy(float), aspect="auto", cmap="cividis")
    axis.set_xticks(
        np.arange(len(matrix.columns)),
        [f"Fold {int(fold)}" for fold in matrix.columns],
    )
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    axis.set_xlabel("Development fold")
    axis.set_ylabel("Candidate, ordered by mean RMSE")
    axis.set_title(f"{_display_name(family)} candidate-by-fold RMSE")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("RMSE (RUL cycles)")
    return _finish_figure(figure, path)


def _selected_fold_metrics(
    folds: pd.DataFrame,
    selected_number: int,
    family: str,
    path: Path,
) -> Path:
    table = folds.loc[folds["candidate_number"].eq(selected_number)].sort_values(
        "outer_fold"
    )
    if table.empty:
        raise Phase3ReportingError("Selected candidate has no fold results")
    labels = [f"Fold {int(value)}" for value in table["outer_fold"]]
    colors = [plt.get_cmap("tab10")(index % 10) for index in range(len(table))]
    metrics = (
        ("rmse", "RMSE (RUL cycles)"),
        ("mae", "MAE (RUL cycles)"),
        ("r2", "R2"),
        ("bias", "Bias (predicted - observed RUL)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis, (column, label) in zip(axes.flat, metrics, strict=True):
        values = table[column].to_numpy(dtype=float)
        axis.bar(labels, values, color=colors, edgecolor="white", linewidth=0.6)
        axis.axhline(
            float(np.mean(values)),
            color="#111827",
            linestyle="--",
            linewidth=1.4,
            label="Fold mean",
        )
        if column == "bias":
            axis.axhline(0.0, color="#9ca3af", linewidth=1.0)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle(
        f"{_display_name(family)} selected-configuration fold metrics",
        fontsize=14,
    )
    return _finish_figure(figure, path)


def _input_alternative_performance(
    candidates: pd.DataFrame,
    selected_number: int,
    family: str,
    path: Path,
) -> Path:
    choices, choice_name = _input_choices(candidates)
    table = candidates.assign(input_choice=choices.astype(str))
    labels = list(dict.fromkeys(table["input_choice"]))
    colors = _choice_colors(table["input_choice"])
    groups = [
        table.loc[table["input_choice"].eq(label), "mean_fold_rmse"].to_numpy(float)
        for label in labels
    ]
    figure, axis = plt.subplots(figsize=(max(9.0, 2.2 * len(labels)), 6.5))
    boxes = axis.boxplot(
        groups,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
    )
    for box, label in zip(boxes["boxes"], labels, strict=True):
        box.set_facecolor(colors[label])
        box.set_alpha(0.35)
    for position, label in enumerate(labels, start=1):
        rows = table.loc[table["input_choice"].eq(label)]
        jitter = ((rows["candidate_number"].to_numpy(int) % 7) - 3) * 0.035
        axis.scatter(
            position + jitter,
            rows["mean_fold_rmse"],
            s=36,
            color=colors[label],
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
    selected = table.loc[table["candidate_number"].eq(selected_number)].iloc[0]
    selected_position = labels.index(str(selected["input_choice"])) + 1
    axis.scatter(
        selected_position,
        float(selected["mean_fold_rmse"]),
        marker="*",
        s=240,
        color="#e11d48",
        edgecolor="black",
        linewidth=0.7,
        label="Selected",
        zorder=3,
    )
    axis.set_xlabel(choice_name)
    axis.set_ylabel("Mean five-fold RMSE (RUL cycles)")
    axis.set_title(
        f"{_display_name(family)} input-alternative performance\n"
        "candidate hyperparameters vary within each group"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    return _finish_figure(figure, path)


def _performance_efficiency(
    candidates: pd.DataFrame,
    selected_number: int,
    family: str,
    path: Path,
) -> Path:
    choices, _ = _input_choices(candidates)
    colors = _choice_colors(choices)
    figure, axis = plt.subplots(figsize=(11, 6.5))
    for choice in colors:
        rows = candidates.loc[choices.eq(choice)]
        axis.scatter(
            rows["mean_training_seconds"],
            rows["mean_fold_rmse"],
            s=48,
            color=colors[choice],
            edgecolor="white",
            linewidth=0.5,
            label=choice,
        )
    selected = candidates.loc[candidates["candidate_number"].eq(selected_number)].iloc[0]
    axis.scatter(
        float(selected["mean_training_seconds"]),
        float(selected["mean_fold_rmse"]),
        marker="*",
        s=240,
        color="#e11d48",
        edgecolor="black",
        linewidth=0.7,
        label="Selected",
        zorder=3,
    )
    positive_times = candidates.loc[
        candidates["mean_training_seconds"].gt(0), "mean_training_seconds"
    ]
    if (
        len(positive_times) > 1
        and float(positive_times.max() / positive_times.min()) > 20.0
    ):
        axis.set_xscale("log")
    axis.set_xlabel("Mean training time per fold (seconds)")
    axis.set_ylabel("Mean five-fold RMSE (RUL cycles)")
    axis.set_title(f"{_display_name(family)} search performance and cost")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=min(4, len(colors) + 1))
    return _finish_figure(figure, path)


def _final_training_curve(
    run_number: int,
    family: str,
    path: Path,
) -> Path | None:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return None

    event_root = tensorboard_log_root(run_number) / "step_6" / family
    event_directories = sorted(
        {event.parent for event in event_root.rglob("events.out.tfevents.*")}
    )
    rows: list[dict[str, float | int]] = []
    for directory in event_directories:
        accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            if not tag.endswith("/train/loss"):
                continue
            rows.extend(
                {
                    "step": int(event.step),
                    "value": float(event.value),
                    "wall_time": float(event.wall_time),
                }
                for event in accumulator.Scalars(tag)
            )
    if not rows:
        return None
    table = (
        pd.DataFrame(rows)
        .sort_values("wall_time")
        .drop_duplicates("step", keep="last")
        .sort_values("step")
    )
    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.plot(
        table["step"],
        table["value"],
        color="#2563eb",
        linewidth=2.0,
        marker="o" if len(table) < 40 else None,
        markersize=3,
    )
    axis.set_xlabel("Training iteration or epoch")
    axis.set_ylabel("Training loss")
    axis.set_title(f"{_display_name(family)} final all-UAV training curve")
    axis.grid(alpha=0.25)
    return _finish_figure(figure, path)


def _test_prediction_diagnostics(
    predictions: pd.DataFrame,
    family: str,
    path: Path,
) -> Path:
    values = pd.to_numeric(predictions["RUL"], errors="coerce").to_numpy(float)
    cutoffs = pd.to_numeric(predictions["cutoff"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or not np.isfinite(cutoffs).all():
        raise Phase3ReportingError("Test prediction diagnostics contain non-finite values")
    bins = min(15, max(5, int(math.ceil(math.sqrt(len(values))))))
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].hist(values, bins=bins, color="#2563eb", alpha=0.82, edgecolor="white")
    axes[0].axvline(
        float(np.median(values)),
        color="#111827",
        linestyle="--",
        linewidth=1.5,
        label="Median",
    )
    axes[0].set_xlabel("Predicted RUL (cycles)")
    axes[0].set_ylabel("Test UAV count")
    axes[0].set_title("Prediction distribution")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].scatter(
        cutoffs,
        values,
        s=42,
        color="#0f766e",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].set_xlabel("Final observed cutoff (cycles)")
    axes[1].set_ylabel("Predicted RUL (cycles)")
    axes[1].set_title("Prediction versus available history")
    axes[1].grid(alpha=0.25)
    figure.suptitle(
        f"{_display_name(family)} test predictions\n"
        "descriptive only; test targets and test metrics are unavailable",
        fontsize=14,
    )
    return _finish_figure(figure, path)


def build_report(run_number: int) -> dict[str, Any]:
    settings = read_json(
        step_directory(1, run_number=run_number)
        / "artifacts"
        / "resolved_phase_3_settings.json",
        "resolved Phase 3 settings",
    )["settings"]
    settings_version = int(settings["settings_version"])
    for prerequisite in (2, 4, 5, 6):
        if complete_manifest(prerequisite, run_number, settings_version) is None:
            raise Phase3ReportingError(
                f"Phase 3 Step {prerequisite} must be complete before reporting"
            )

    search_artifacts = step_directory(2, run_number=run_number) / "artifacts"
    candidates = _read_csv(
        search_artifacts / "final_search_candidate_results.csv",
        "final-search candidate results",
    )
    folds = _read_csv(
        search_artifacts / "final_search_fold_results.csv",
        "final-search fold results",
    )
    predictions = _read_csv(test_predictions_path(run_number), "test predictions")
    _require_columns(candidates, CANDIDATE_COLUMNS, "Candidate results")
    _require_columns(folds, FOLD_COLUMNS, "Fold results")
    _require_columns(predictions, {"uav_id", "cutoff", "RUL"}, "Test predictions")
    forbidden_test_columns = {"target", "actual_rul", "observed_rul", "residual"}
    if forbidden_test_columns & set(predictions.columns):
        raise Phase3ReportingError("Test prediction table unexpectedly contains targets")

    selection = read_json(
        selected_configuration_path(run_number),
        "selected Phase 3 configuration",
    )
    family = str(selection["model_family"])
    selected_number = int(selection["candidate_number"])
    if selected_number not in set(candidates["candidate_number"].astype(int)):
        raise Phase3ReportingError("Selected candidate is absent from candidate results")
    candidate_numbers = set(candidates["candidate_number"].astype(int))
    fold_candidate_numbers = set(folds["candidate_number"].astype(int))
    if candidate_numbers != fold_candidate_numbers:
        raise Phase3ReportingError("Candidate and fold result identifiers disagree")

    report_dir = run_root(run_number) / REPORT_DIRECTORY_NAME
    figure_dir = report_dir / "figures"
    figures = {
        "optimization_history": _optimization_history(
            candidates,
            selected_number,
            family,
            figure_dir / "search_optimization_history.png",
        ),
        "top_candidate_robustness": _top_candidate_robustness(
            candidates,
            folds,
            selected_number,
            family,
            figure_dir / "top_candidate_fold_robustness.png",
        ),
        "candidate_fold_heatmap": _fold_heatmap(
            candidates,
            folds,
            selected_number,
            family,
            figure_dir / "candidate_fold_rmse_heatmap.png",
        ),
        "selected_fold_metrics": _selected_fold_metrics(
            folds,
            selected_number,
            family,
            figure_dir / "selected_configuration_fold_metrics.png",
        ),
        "input_alternative_performance": _input_alternative_performance(
            candidates,
            selected_number,
            family,
            figure_dir / "input_alternative_performance.png",
        ),
        "performance_efficiency": _performance_efficiency(
            candidates,
            selected_number,
            family,
            figure_dir / "search_performance_efficiency.png",
        ),
        "test_prediction_diagnostics": _test_prediction_diagnostics(
            predictions,
            family,
            figure_dir / "test_prediction_diagnostics.png",
        ),
    }
    skipped: dict[str, str] = {}
    training_curve = _final_training_curve(
        run_number,
        family,
        figure_dir / "final_training_curve.png",
    )
    if training_curve is None:
        skipped["final_training_curve"] = "No shared train/loss scalar was recorded"
    else:
        figures["final_training_curve"] = training_curve

    selected = candidates.loc[
        candidates["candidate_number"].astype(int).eq(selected_number)
    ].iloc[0]
    summary = {
        "report_version": REPORT_VERSION,
        "settings_version": settings_version,
        "phase_3_run_number": run_number,
        "model_family": family,
        "configuration_id": selection["configuration_id"],
        "candidate_count": len(candidates),
        "fold_count": int(folds["outer_fold"].nunique()),
        "selected_candidate_number": selected_number,
        "selected_mean_fold_rmse": float(selected["mean_fold_rmse"]),
        "selected_fold_rmse_standard_deviation": float(
            selected["fold_rmse_standard_deviation"]
        ),
        "selected_mean_fold_r2": float(selected["mean_fold_r2"]),
        "selected_mean_fold_mae": float(selected["mean_fold_mae"]),
        "selected_mean_fold_bias": float(selected["mean_fold_bias"]),
        "test_prediction_rows": len(predictions),
        "test_prediction_minimum": float(predictions["RUL"].min()),
        "test_prediction_median": float(predictions["RUL"].median()),
        "test_prediction_maximum": float(predictions["RUL"].max()),
        "test_targets_loaded": False,
        "test_metrics_calculated": False,
        "selection_changed": False,
    }
    write_json(summary, report_dir / "report_summary.json")
    manifest = {
        "report_version": REPORT_VERSION,
        "settings_version": settings_version,
        "phase_3_run_number": run_number,
        "status": "complete",
        "model_family": family,
        "configuration_id": selection["configuration_id"],
        "model_agnostic": True,
        "source_artifacts": {
            "candidate_results": (
                "../2_final_configuration_search/artifacts/"
                "final_search_candidate_results.csv"
            ),
            "fold_results": (
                "../2_final_configuration_search/artifacts/"
                "final_search_fold_results.csv"
            ),
            "test_predictions": "../5_test_inference/artifacts/test_predictions.csv",
        },
        "figures": {
            name: str(path.relative_to(report_dir).as_posix())
            for name, path in figures.items()
        },
        "skipped_figures": skipped,
        "artifacts": {"summary": "report_summary.json"},
        "test_targets_loaded": False,
        "test_metrics_calculated": False,
        "selection_changed": False,
    }
    write_json(manifest, report_dir / "report_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=SETTINGS_PATH)
    args = parser.parse_args()
    try:
        run_number = require_current_settings(args.settings)
        manifest = build_report(run_number)
    except (Phase3ReportingError, Phase3Error, OSError, ValueError) as error:
        print(f"Phase 3 reporting failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("Phase 3 report complete")
    print(f"Model family: {manifest['model_family']}")
    print(f"Figures: {len(manifest['figures'])}")
    print(f"Report: {run_root(run_number) / REPORT_DIRECTORY_NAME}")


if __name__ == "__main__":
    main()
