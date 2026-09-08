"""Assemble every figure used in UAV_RUL_Project_20min.pptx.

Policy, after the 8 September 2026 review:

1. Figures that the repository has already generated are used **unchanged**.
   They are copied into ``figures/`` and listed in ``figure_manifest.csv``
   with their source path.
2. Where an existing figure is a stacked multi-panel image that is unreadable
   when scaled onto a 10 x 5.625 in slide, the *same* generated image is
   cropped to the panel that is shown. Nothing is re-plotted or restyled.
3. Where the repository has no figure for a needed piece of evidence, the
   figure is produced with the repository's own plotting conventions: the
   Phase 0 palette and axis styling from
   ``0_data_analysis/broad_data_review/plotting_common.py``, matplotlib,
   constrained layout, 160 dpi.
4. One repository script is re-run rather than copied: the Phase 0 temporal /
   RUL summary, restricted to the eight channels shown on the slide. See
   README.md for the exact command.

    python build_figures.py --repo-root <path to UAV-RUL-Estimation>
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "chart_data"
FIG = HERE / "figures"
REPO_FIG = HERE / "figures_repo"
FIG.mkdir(parents=True, exist_ok=True)

# --- repository Phase 0 palette (0_data_analysis/.../plotting_common.py) ----
DARK_BLUE = "#0072B2"
LIGHT_BLUE = "#56B4E9"
ORANGE = "#D55E00"
GREEN = "#009E73"
YELLOW = "#E69F00"
GRAY = "#7A7A7A"
LIGHT_GRAY = "#D9D9D9"
RED = "#C44E52"

DPI = 160

MANIFEST: list[dict] = []


def style_axis(axis):
    """Same helper as the repository's plotting_common.style_axis."""
    axis.grid(True, alpha=0.2)
    axis.tick_params(labelsize=8)


PLACEMENT = {
    "prediction_timeline.png": "slide 1",
    "public_score_chain.png": "slide 2",
    "temporal_rul_summary_subset.png": "slide 3",
    "history_length_ecdf_panel.png": "slide 4",
    "pe2_target_scenario_2x2_rmse_panel.png": "slide 5",
    "pe2_signal_family_ablation_rmse_panel.png": "slide 6",
    "r2_comparison.png": "slide 7",
    "bagging_residual_comparison.png": "slide 8",
    "calibration_tradeoff.png": "slide 9",
    "development_prediction_scatter.png": "slide 10",
    "pe15_comparison.png": "slide 11",
    "development_versus_public.png": "slide 12",
    "hybrid_architecture_comparison.png": "backup B2",
}


def record(name, kind, source, note=""):
    MANIFEST.append(dict(figure=name, provenance=kind, source=source,
                         placed_on=PLACEMENT.get(name, "not placed (available)"),
                         note=note))


# ------------------------------------------------------- 1. copy as-is -----
COPY_AS_IS = [
    ("history_length_distributions.png",
     "0_data_analysis/broad_data_review/figures/history_length_distributions/"
     "history_length_distributions.png",
     "Phase 0 broad review: train vs test history lengths"),
    ("r2_comparison.png",
     "2_architecture_experiments/2_model_architecture_study/runs/run_5/"
     "7_architecture_comparison/figures/r2_comparison.png",
     "Locked architecture comparison, architecture study run 5"),
    ("hybrid_architecture_comparison.png",
     "2_architecture_experiments/2_model_architecture_study/runs/run_8/figures/"
     "hybrid_architecture_comparison.png",
     "Hybrid study, architecture study run 8"),
    ("temporal_architecture_comparison.png",
     "2_architecture_experiments/2_model_architecture_study/runs/run_7/"
     "7_architecture_comparison/temporal_architecture_comparison.png",
     "Dedicated temporal study, architecture study run 7"),
    ("bagging_residual_comparison.png",
     "2_architecture_experiments/1_pipeline_experiments/experiments/PE_11/runs/"
     "run_1/figures/bagging_residual_comparison.png",
     "PE_11 bagging and residual correction"),
    ("calibration_tradeoff.png",
     "2_architecture_experiments/1_pipeline_experiments/experiments/PE_4/runs/"
     "run_1/figures/calibration_tradeoff.png",
     "PE_4 conditional safety calibration trade-off"),
    ("pe15_comparison.png",
     "2_architecture_experiments/1_pipeline_experiments/experiments/PE_15/runs/"
     "run_1/figures/comparison.png",
     "PE_15 causal filtering and forecast history, screening"),
    ("pe20_comparison.png",
     "2_architecture_experiments/1_pipeline_experiments/experiments/PE_20/runs/"
     "run_1/figures/comparison.png",
     "PE_20 complete nested refit of the forecast-history recipe"),
    ("development_prediction_scatter.png",
     "3_final_model_training_and_inference/runs/run_7/7_post_run_reporting/"
     "figures/development_prediction_scatter.png",
     "Phase 3 Run 7 development out-of-fold prediction alignment"),
    ("development_overprediction_diagnostics.png",
     "3_final_model_training_and_inference/runs/run_7/7_post_run_reporting/"
     "figures/development_overprediction_diagnostics.png",
     "Phase 3 Run 7 development overprediction diagnostics"),
]


def copy_repo_figures(repo: Path):
    for name, rel, note in COPY_AS_IS:
        src = repo / rel
        if not src.is_file():
            print(f"  MISSING {rel}")
            continue
        shutil.copyfile(src, FIG / name)
        record(name, "repository figure, unchanged", rel, note)
        print("copied", name)


# ------------------------------------------------- 2. crop one panel -------
# The PE_2 paired-comparison figures stack a paired-R2 panel above a paired
# RMSE panel on an 11.9 x 7.9 in canvas. At slide width that is ~5 pt type.
# The lower panel carries the category labels, so it is the panel shown.
CROPS = [
    ("pe2_target_scenario_2x2_rmse_panel.png",
     "2_architecture_experiments/1_pipeline_experiments/experiments/PE_2/runs/"
     "run_1/figures/PE_target_scenario_2x2__reporting__paired_comparison.png",
     0.462,
     "Lower panel (paired RMSE improvement) of the generated PE_2 figure"),
    ("pe2_signal_family_ablation_rmse_panel.png",
     "2_architecture_experiments/1_pipeline_experiments/experiments/PE_2/runs/"
     "run_1/figures/PE_signal_family_ablation__reporting__paired_comparison.png",
     0.462,
     "Lower panel (paired RMSE improvement) of the generated PE_2 figure"),
]

# The history-length figure is a 15 x 4.8 in three-panel strip. On a half
# slide only the empirical-CDF panel stays legible, and it is the panel the
# cutoff decision rests on.
SIDE_CROPS = [
    ("history_length_ecdf_panel.png",
     "0_data_analysis/broad_data_review/figures/history_length_distributions/"
     "history_length_distributions.png",
     (0.352, 0.0, 0.668, 1.0),
     "Empirical-CDF panel of the generated Phase 0 history-length figure"),
]


def side_crop_repo_figures(repo: Path):
    for name, rel, box, note in SIDE_CROPS:
        src = repo / rel
        if not src.is_file():
            print(f"  MISSING {rel}")
            continue
        image = Image.open(src)
        width, height = image.size
        dpi = image.info.get("dpi", (DPI, DPI))
        left, top, right, bottom = box
        image.crop((int(left * width), int(top * height),
                    int(right * width), int(bottom * height))).save(
            FIG / name, dpi=dpi)
        record(name, f"repository figure, cropped to x={box[0]:.3f}-{box[2]:.3f}W",
               rel, note)
        print("cropped", name)


def crop_repo_figures(repo: Path):
    for name, rel, top_fraction, note in CROPS:
        src = repo / rel
        if not src.is_file():
            print(f"  MISSING {rel}")
            continue
        image = Image.open(src)
        width, height = image.size
        dpi = image.info.get("dpi", (DPI, DPI))
        top = int(round(height * top_fraction))
        image.crop((0, top, width, height)).save(FIG / name, dpi=dpi)
        record(name, f"repository figure, cropped from y={top_fraction:.3f}H",
               rel, note)
        print("cropped", name)


# ------------------------------- 3. re-run one repository script -----------
def collect_rerun():
    """Copy in the Phase 0 temporal/RUL summary re-run for eight channels."""
    src = REPO_FIG / "temporal_rul_subset" / "temporal_rul_summary.png"
    if not src.is_file():
        print("  MISSING temporal_rul_subset/temporal_rul_summary.png "
              "(see README.md for the command)")
        return
    shutil.copyfile(src, FIG / "temporal_rul_summary_subset.png")
    record("temporal_rul_summary_subset.png",
           "repository script re-run on a channel subset",
           "0_data_analysis/core_data_analysis/temporal_rul_analysis.py",
           "Same script, same statistics and colours; --channels limited to the "
           "eight channels shown and the fixed 19x10 in canvas scaled to the "
           "channel count so the labels survive projection")
    print("collected temporal_rul_summary_subset.png")


# --------------------------------- 4. figures the repository lacks ---------
def fig_timeline(repo: Path):
    """The prediction problem. No repository figure shows the cutoff itself."""
    train = pd.read_csv(repo / "data" / "train.csv")
    lives = train.groupby("uav_id")["flight_cycle"].max()
    uav = lives.sub(230).abs().idxmin()
    history = train[train.uav_id == uav].sort_values("flight_cycle")
    life = int(history.flight_cycle.max())
    cutoff = 148

    figure, axis = plt.subplots(figsize=(9.0, 3.0), constrained_layout=True)
    observed = history[history.flight_cycle <= cutoff]
    unseen = history[history.flight_cycle >= cutoff]
    axis.plot(observed.flight_cycle, observed.telemetry_21, color=DARK_BLUE,
              lw=1.6, label="observed telemetry (channel 21)")
    axis.plot(unseen.flight_cycle, unseen.telemetry_21, color=GRAY, lw=1.2,
              ls=":", label="not available at prediction time")
    axis.axvspan(cutoff, life, color=LIGHT_GRAY, alpha=0.45)
    axis.axvline(cutoff, color="black", lw=1.3)
    axis.axvline(life, color=ORANGE, lw=2.0)

    low, high = axis.get_ylim()
    axis.set_ylim(low, high + 0.42 * (high - low))
    marker = high + 0.10 * (high - low)
    axis.annotate("", xy=(life, marker), xytext=(cutoff, marker),
                  arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    axis.text((cutoff + life) / 2, marker + 0.05 * (high - low),
              f"remaining useful life = {life - cutoff} cycles "
              "(the quantity to predict)", ha="center", va="bottom",
              fontsize=10)
    axis.text(cutoff - 5, high, "prediction cutoff\n(last observed cycle)",
              ha="right", va="top", fontsize=9)
    axis.text(life + 3, low, "failure", ha="left", va="bottom", fontsize=9,
              color=ORANGE)
    axis.set_xlabel("flight cycle", fontsize=9)
    axis.set_ylabel("telemetry_21", fontsize=9)
    axis.set_xlim(0, life + 46)
    axis.set_title(f"{uav}: complete life {life} cycles. A test UAV is observed "
                   "only up to its cutoff.", fontsize=10)
    axis.legend(loc="upper left", fontsize=8, frameon=False)
    style_axis(axis)
    figure.savefig(FIG / "prediction_timeline.png", dpi=DPI)
    plt.close(figure)
    record("prediction_timeline.png", "generated for the talk",
           "data/train.csv",
           "Repository Phase 0 palette and axis styling; no repository figure "
           "shows the prediction cutoff")
    print("wrote prediction_timeline.png")


def fig_score_ladder():
    """The answer: recorded public score after each retained decision."""
    steps = pd.read_csv(DATA / "s02_public_score_chain.csv")
    x = np.arange(len(steps))
    figure, axis = plt.subplots(figsize=(10.4, 4.3), constrained_layout=True)

    axis.plot(x, steps.public_r2, color=GRAY, lw=1.4, zorder=1)
    axis.scatter(x, steps.public_r2, s=150, color=DARK_BLUE, zorder=3,
                 edgecolor="white", lw=1.4)
    axis.scatter([x[-1]], [steps.public_r2.iloc[-1]], s=210, color=ORANGE,
                 zorder=4, edgecolor="white", lw=1.4)

    for xi, row in zip(x, steps.itertuples()):
        axis.annotate(f"{row.public_r2:.5f}", (xi, row.public_r2),
                      textcoords="offset points", xytext=(0, 13),
                      ha="center", fontsize=10)
    for xi in range(1, len(steps)):
        delta = steps.public_r2.iloc[xi] - steps.public_r2.iloc[xi - 1]
        mid = (x[xi] + x[xi - 1]) / 2
        height = (steps.public_r2.iloc[xi] + steps.public_r2.iloc[xi - 1]) / 2
        axis.annotate(f"{delta:+.5f}", (mid, height),
                      textcoords="offset points", xytext=(0, -20),
                      ha="center", fontsize=10, color=ORANGE)

    axis.axhline(0.9, color="black", ls="--", lw=1)
    axis.text(1.6, 0.912, "R² = 0.9 objective", fontsize=9, ha="center")
    axis.set_xticks(x)
    axis.set_xticklabels(
        [f"{r.submission}\n" + "\n".join(textwrap.wrap(r.decision, 24))
         for r in steps.itertuples()],
        fontsize=8.5)
    axis.set_ylim(0.50, 0.945)
    axis.set_ylabel("recorded Kaggle public R²\n(about 30% of the test set)",
                    fontsize=9)
    style_axis(axis)
    figure.savefig(FIG / "public_score_chain.png", dpi=DPI)
    plt.close(figure)
    record("public_score_chain.png", "generated for the talk",
           "literature_and_planning/development_documentation/"
           "r2_research_2026_09_07/kaggle_scores.csv",
           "Repository Phase 0 palette; decisions from pipeline_experiments.md")
    print("wrote public_score_chain.png")


def fig_dev_vs_public():
    """Development and public performance, kept on separate axes."""
    table = pd.read_csv(DATA / "s12_development_and_public.csv")
    development = table[table.scope == "development"]
    public = table[table.scope == "public"]

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.9),
                                constrained_layout=True)

    axis = axes[0]
    x = np.arange(len(development))
    axis.bar(x, development.r2, color=[GRAY, DARK_BLUE, LIGHT_BLUE], width=0.55)
    for xi, row in zip(x, development.itertuples()):
        axis.text(xi, row.r2 + 0.0015, f"{row.r2:.5f}", ha="center", fontsize=9)
    axis.axhline(0.9, color="black", ls="--", lw=1)
    axis.text(2.45, 0.9015, "R² = 0.9 objective", fontsize=8, ha="right")
    axis.set_xticks(x)
    axis.set_xticklabels(["Run 6\nmean-fold", "Run 7\nmean-fold",
                          "Run 7\npooled"], fontsize=9)
    axis.set_ylim(0.85, 0.915)
    axis.set_ylabel("R²", fontsize=9)
    axis.set_title("Development: 5 held-out UAV folds, 500 rows", fontsize=10)
    style_axis(axis)

    axis = axes[1]
    x = np.arange(len(public))
    axis.bar(x, public.r2, color=[GRAY, GRAY, GRAY, ORANGE], width=0.55)
    for xi, row in zip(x, public.itertuples()):
        axis.text(xi, row.r2 + 0.0015, f"{row.r2:.5f}", ha="center", fontsize=9)
    axis.axhline(0.9, color="black", ls="--", lw=1)
    axis.text(-0.45, 0.9015, "R² = 0.9 objective", fontsize=8)
    axis.set_xticks(x)
    axis.set_xticklabels([r.label for r in public.itertuples()], fontsize=9)
    axis.set_ylim(0.82, 0.915)
    axis.set_xlim(-0.6, 3.9)
    axis.annotate("", xy=(3.55, 0.8995), xytext=(3.55, 0.87752),
                  arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2))
    axis.text(-0.5, 0.8955,
              "remaining gap 0.02348 R²:\nabout 10.0% lower RMSE on the "
              "same scored set", fontsize=8.5, va="top")
    axis.set_ylabel("R²", fontsize=9)
    axis.set_title("Recorded Kaggle public score (about 30% of the test set)",
                   fontsize=10)
    style_axis(axis)

    figure.savefig(FIG / "development_versus_public.png", dpi=DPI)
    plt.close(figure)
    record("development_versus_public.png", "generated for the talk",
           "3_final_model_training_and_inference/runs/run_6|run_7/"
           "7_post_run_reporting/report_summary.json; kaggle_scores.csv",
           "Repository Phase 0 palette; the two scopes are never joined")
    print("wrote development_versus_public.png")


# ----------------------------------------------------------------- main ----
def main():
    parser = argparse.ArgumentParser()
    default_root = os.environ.get("UAV_RUL_ROOT",
                                  "/mnt/user-data/uploads/UAV-RUL-Estimation")
    parser.add_argument("--repo-root", default=default_root)
    args = parser.parse_args()
    repo = Path(args.repo_root)

    copy_repo_figures(repo)
    crop_repo_figures(repo)
    side_crop_repo_figures(repo)
    collect_rerun()
    fig_timeline(repo)
    fig_score_ladder()
    fig_dev_vs_public()

    with (HERE / "figure_manifest.csv").open("w", newline="",
                                             encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=["figure", "provenance", "source",
                                            "placed_on", "note"])
        writer.writeheader()
        writer.writerows(MANIFEST)
    print(f"\n{len(MANIFEST)} figures; manifest written")
    kinds = pd.Series([m["provenance"].split(",")[0] for m in MANIFEST])
    print(kinds.value_counts().to_string())


if __name__ == "__main__":
    main()
