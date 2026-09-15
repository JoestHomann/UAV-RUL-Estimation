"""Split every Run 10 evaluation into model fitting and the rest of the cell.

The training-time chart answers only half of the cost question. It records the
seconds spent inside the stopping-stage fit and the refit, which is the whole
cost for a parametric model but essentially none of the cost for an
instance-based one: Trajectory DTW-kNN stores its reference trajectories and
does the work at prediction time. This script reads the same per-cell audits and
reports the complete wall clock of each evaluation, split into the fitted part
and everything else the cell did -- feature preparation, prediction and file
I/O. The second segment is not pure inference and is not labelled as such.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_REPORTING = HERE / "runs" / "run_10" / "reporting"

DARK_BLUE = "#0072B2"
YELLOW = "#E69F00"
INK = "#30343B"

DISPLAY = {
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "multiscale_cnn": "Multi-scale CNN",
    "trajectory_dtw_knn": "Trajectory DTW-kNN",
    "mlp": "MLP",
}


def collect_cells(reporting: Path) -> pd.DataFrame:
    """One row per fold/seed cell: fitted seconds and total wall clock."""
    cells = reporting.parent / "cells"
    rows: list[dict[str, object]] = []
    for audit_path in sorted(cells.glob("*/fit_audit.json")):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        stopping = (audit.get("stopping") or {}).get("summary") or {}
        refit = audit.get("refit_summary") or {}
        fitting = float(stopping.get("training_seconds") or 0.0) + float(
            refit.get("training_seconds") or 0.0
        )
        elapsed = float(audit["elapsed_seconds"])
        rows.append(
            {
                "model_family": audit_path.parent.name.split("__", maxsplit=1)[0],
                "cell": audit_path.parent.name,
                "fitting_seconds": fitting,
                "cell_elapsed_seconds": elapsed,
                "other_seconds": elapsed - fitting,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or set(frame.model_family) != set(DISPLAY):
        raise ValueError("The cell audits do not cover every Run 10 model family")
    if (frame.other_seconds < 0).any():
        raise ValueError("A cell reports more fitting time than total elapsed time")
    return frame


def summarise(cells: pd.DataFrame) -> pd.DataFrame:
    summary = (
        cells.groupby("model_family", as_index=False)
        .agg(
            evaluations=("cell_elapsed_seconds", "size"),
            median_fitting_seconds=("fitting_seconds", "median"),
            median_other_seconds=("other_seconds", "median"),
            median_cell_elapsed_seconds=("cell_elapsed_seconds", "median"),
            q1_cell_elapsed_seconds=("cell_elapsed_seconds", lambda v: v.quantile(0.25)),
            q3_cell_elapsed_seconds=("cell_elapsed_seconds", lambda v: v.quantile(0.75)),
            total_cell_elapsed_seconds=("cell_elapsed_seconds", "sum"),
        )
        .sort_values("median_cell_elapsed_seconds", ascending=False)
        .reset_index(drop=True)
    )
    summary["other_share"] = (
        summary.median_other_seconds / summary.median_cell_elapsed_seconds
    )
    return summary


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.frameon": False,
        }
    )


def plot(summary: pd.DataFrame, output: Path) -> Path:
    configure_style()
    figure, axis = plt.subplots(figsize=(11, 4.2))
    figure.subplots_adjust(left=0.24, right=0.985, top=0.985, bottom=0.2)
    y = np.arange(len(summary))
    fitting = summary.median_fitting_seconds.to_numpy(float)
    other = summary.median_other_seconds.to_numpy(float)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#D9D9D9", alpha=0.45, linewidth=0.7)
    axis.barh(y, fitting, height=0.62, color=DARK_BLUE, label="Model fitting")
    axis.barh(
        y,
        other,
        left=fitting,
        height=0.62,
        color=YELLOW,
        label="Rest of the evaluation (data preparation, prediction, I/O)",
    )
    axis.set_yticks(y, [DISPLAY[value] for value in summary.model_family])
    axis.set_ylim(len(summary) - 0.5, -0.5)
    axis.set_xlabel("Median wall-clock time per fold and seed evaluation (seconds)", labelpad=12)
    axis.tick_params(axis="x", length=3, width=0.7, pad=6)
    axis.tick_params(axis="y", length=0, pad=10)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_linewidth(0.7)
    axis.legend(loc="lower right", bbox_to_anchor=(0.995, 0.06), fontsize=10.5,
                handlelength=1.4, handleheight=1.0, borderpad=0.2, labelspacing=0.55)
    path = output / "run10_model_cost_split.png"
    figure.savefig(path, dpi=240, facecolor="white", bbox_inches="tight", pad_inches=0.025)
    figure.savefig(
        path.with_suffix(".svg"), facecolor="white", bbox_inches="tight", pad_inches=0.025
    )
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reporting", type=Path, default=DEFAULT_REPORTING)
    arguments = parser.parse_args()
    output = arguments.reporting / "figures_2"
    output.mkdir(parents=True, exist_ok=True)
    cells = collect_cells(arguments.reporting)
    summary = summarise(cells)
    summary.to_csv(output / "model_cost_split_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(plot(summary, output))


if __name__ == "__main__":
    main()
