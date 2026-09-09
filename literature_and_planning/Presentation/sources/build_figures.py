"""Assemble every figure used in UAV_RUL_Project_20min.pptx.

Policy, after the 9 September 2026 review:

1. Figures that the repository has already generated are used **unchanged**.
   They are copied into ``figures/`` and listed in ``figure_manifest.csv``
   with their source path.
2. The Phase 0 figures are drawn on 11 x 9 in to 19 x 10 in canvases. Scaled
   onto the 10 x 5.625 in template their axis labels reach the slide at three
   to four points. Those figures are therefore produced by re-running the
   **unmodified repository script** through ``run_repo_figure.py``, which
   overrides only the canvas size and the tick-label and title sizes so the
   same plot is placed on the slide at 1:1. Every number, colour, ordering and
   label still comes from the repository code; ``build_figures.py`` checks the
   re-run CSV against the repository CSV and reports the largest difference.
3. Where an existing figure is a stacked multi-panel image that is unreadable
   at slide size and has no script available here, the *same* generated image
   is cropped to the panel that is shown. Nothing is re-plotted or restyled.
4. Where the repository has no figure for a needed piece of evidence, the
   figure is produced with the repository's own plotting conventions: the
   Phase 0 palette and axis styling from
   ``0_data_analysis/broad_data_review/plotting_common.py``, matplotlib,
   constrained layout, 160 dpi.

    python build_figures.py --repo-root <path to UAV-RUL-Estimation>
"""

from __future__ import annotations

import argparse
import csv
import json
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
    "constant_features.png": "slide 3",
    "temporal_rul_summary_subset.png": "slide 4",
    "within_between_variance.png": "slide 5",
    "train_test_drift.png": "slide 6",
    "channel_classification.png": "slide 7",
    "history_length_distributions.png": "slide 9",
    "cycle_only_baseline.png": "slide 11",
    "pe2_target_scenario_2x2_rmse_panel.png": "slide 12",
    "r2_comparison.png": "slide 13",
    "calibration_tradeoff.png": "slide 14",
    "bagging_residual_comparison.png": "slide 15",
    "development_prediction_scatter.png": "slide 16",
    "correlation_heatmaps.png": "not placed (available)",
    "hybrid_architecture_comparison.png": "backup B6",
    "pe15_comparison.png": "backup B8",
    "development_versus_public.png": "backup B9",
}


def record(name, kind, source, note=""):
    MANIFEST.append(dict(figure=name, provenance=kind, source=source,
                         placed_on=PLACEMENT.get(name, "not placed (available)"),
                         note=note))


# ------------------------------------------------------- 1. copy as-is -----
COPY_AS_IS = [
    ("correlation_heatmaps.png",
     "0_data_analysis/core_data_analysis/figures/feature_redundancy/"
     "correlation_heatmaps.png",
     "Phase 0 core review: row- and UAV-level redundancy heatmaps"),
    ("anomaly_summary.png",
     "0_data_analysis/core_data_analysis/figures/anomalies/anomaly_summary.png",
     "Phase 0 core review: extreme readings, jumps and persistent shifts"),
    ("grouped_permutation_importance.png",
     "0_data_analysis/model_guided_feature_analysis/runs/FE_run_1/current20/"
     "grouped_permutation_importance.png",
     "FE_run_1 model-guided grouped permutation importance"),
    ("representative_trajectories.png",
     "0_data_analysis/core_data_analysis/figures/representative_trajectories/"
     "representative_trajectories.png",
     "Phase 0 core review: six representative UAV trajectories"),
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


# ------------------------- 3. re-run repository scripts at slide size ------
# (name in figures/, subdirectory of figures_repo/, repository script,
#  csv written by that script, the same csv in the repository, note)
RERUNS = [
    ("constant_features.png", "constant_features",
     "0_data_analysis/broad_data_review/plot_constant_features.py",
     "constant_features.csv",
     "0_data_analysis/broad_data_review/figures/constant_features/"
     "constant_features.csv",
     "Unique-value count and numeric range for all 28 channels"),
    ("temporal_rul_summary_subset.png", "temporal_rul_subset",
     "0_data_analysis/core_data_analysis/temporal_rul_analysis.py",
     "temporal_channel_summary.csv",
     "0_data_analysis/core_data_analysis/figures/temporal_rul/"
     "temporal_channel_summary.csv",
     "--channels restricted to the ten degradation candidates plus "
     "telemetry_18 and telemetry_26"),
    ("within_between_variance.png", "within_between",
     "0_data_analysis/broad_data_review/plot_within_between_variance.py",
     "within_between_variance.csv",
     "0_data_analysis/broad_data_review/figures/within_between_variance/"
     "within_between_variance.csv",
     "Within-UAV and between-UAV variance shares, all 28 channels"),
    ("train_test_drift.png", "train_test_drift",
     "0_data_analysis/core_data_analysis/train_test_drift.py",
     "matched_endpoint_drift_summary.csv",
     "0_data_analysis/core_data_analysis/figures/train_test_drift/"
     "matched_endpoint_drift_summary.csv",
     "Row-level, UAV-level and age-matched train/test drift"),
    ("channel_classification.png", "channel_classification",
     "0_data_analysis/core_data_analysis/channel_classification.py",
     "channel_classification.csv",
     "0_data_analysis/core_data_analysis/figures/channel_classification/"
     "channel_classification.csv",
     "Screening matrix for all 28 channels against the documented thresholds"),
    ("history_length_distributions.png", "history_lengths",
     "0_data_analysis/broad_data_review/plot_history_length_distributions.py",
     "history_lengths.csv",
     "0_data_analysis/broad_data_review/figures/history_length_distributions/"
     "history_lengths.csv",
     "Train and test per-UAV history lengths"),
]


def largest_numeric_difference(left: Path, right: Path):
    """Compare a re-run table with the repository's own table."""
    if not left.is_file() or not right.is_file():
        return None
    first, second = pd.read_csv(left), pd.read_csv(right)
    if list(first.columns) != list(second.columns):
        return "columns differ"
    key = first.columns[0]
    # A re-run restricted to a channel subset has fewer rows; compare the rows
    # it does contain against the repository's values for the same channels.
    first = first.set_index(key).sort_index()
    second = second.set_index(key).sort_index()
    shared = first.index.intersection(second.index)
    if not len(shared):
        return "no shared rows"
    numeric = first.select_dtypes("number").columns
    if not len(numeric):
        return 0.0
    difference = (first.loc[shared, numeric]
                  - second.loc[shared, numeric]).abs().max().max()
    return float(difference)


def collect_reruns(repo: Path):
    for name, directory, script, csv_name, repo_csv, note in RERUNS:
        src = REPO_FIG / directory
        produced = sorted(src.glob("*.png")) if src.is_dir() else []
        if not produced:
            print(f"  MISSING figures_repo/{directory}/*.png "
                  "(see README.md for the command)")
            continue
        shutil.copyfile(produced[0], FIG / name)
        difference = largest_numeric_difference(src / csv_name, repo / repo_csv)
        if difference is None:
            check = "re-run table not compared"
        elif isinstance(difference, str):
            check = f"WARNING: {difference}"
        else:
            check = ("re-run table matches the repository table "
                     f"(largest numeric difference {difference:g})")
        record(name, "repository script re-run at slide size", script,
               f"{note}. {check}")
        print(f"collected {name}: {check}")


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


def fig_cycle_only_baseline(repo: Path):
    """Phase 1 step 9. The repository stores the baseline's coefficients and
    metrics but no figure, so the benchmark is drawn here from those files."""
    prefixes = pd.read_csv(
        repo / "1_dataset_construction" / "4_training_prefixes" / "artifacts"
        / "training_prefixes.csv")
    coefficients = json.loads(
        (repo / "1_dataset_construction" / "9_cycle_only_baseline"
         / "artifacts" / "full_training_coefficients.json").read_text())
    metrics = json.loads(
        (repo / "1_dataset_construction" / "9_cycle_only_baseline"
         / "artifacts" / "metrics" / "overall_metrics.json").read_text())
    bands = pd.read_csv(
        repo / "1_dataset_construction" / "9_cycle_only_baseline" / "artifacts"
        / "metrics" / "age_band_metrics.csv")

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.30),
                                constrained_layout=True,
                                gridspec_kw={"width_ratios": [1.35, 1.0]})

    axis = axes[0]
    axis.scatter(prefixes.cutoff, prefixes.RUL, s=5, alpha=0.30,
                 color=LIGHT_BLUE, linewidths=0,
                 label="2,000 training prefixes")
    grid = np.linspace(prefixes.cutoff.min(), prefixes.cutoff.max(), 200)
    axis.plot(grid,
              np.maximum(0, coefficients["intercept"]
                         + coefficients["slope"] * grid),
              color=ORANGE, lw=2.0,
              label=(f"predicted RUL = max(0, {coefficients['intercept']:.3f} "
                     f"− {abs(coefficients['slope']):.5f} × cycle)"))
    axis.set_xlabel("prefix cutoff (flight cycle)", fontsize=8.5)
    axis.set_ylabel("true RUL at the cutoff (cycles)", fontsize=8.5)
    axis.set_title("Age alone cannot separate the UAVs", fontsize=9.5)
    axis.legend(loc="upper right", fontsize=7.5, frameon=False)
    style_axis(axis)

    axis = axes[1]
    order = ["1-50", "51-100", "101-200", ">200"]
    bands = bands.set_index("age_band").loc[order].reset_index()
    x = np.arange(len(bands))
    axis.bar(x, bands.rmse, width=0.6, color=DARK_BLUE)
    for xi, row in zip(x, bands.itertuples()):
        axis.text(xi, row.rmse + 1.4, f"R² {row.r2:+.3f}", ha="center",
                  fontsize=8)
    axis.axhline(metrics["rmse"], color=ORANGE, ls="--", lw=1.3)
    axis.text(-0.42, metrics["rmse"] + 5.6,
              f"overall RMSE {metrics['rmse']:.3f}, R² {metrics['r2']:+.3f}",
              ha="left", fontsize=8, color=ORANGE)
    axis.set_xticks(x, [f"{band}\ncycles" for band in order], fontsize=8)
    axis.set_ylim(0, 90)
    axis.set_ylabel("locked-scenario RMSE (cycles)", fontsize=8.5)
    axis.set_title("Held-out benchmark by cutoff band", fontsize=9.5)
    style_axis(axis)

    figure.savefig(FIG / "cycle_only_baseline.png", dpi=DPI)
    plt.close(figure)
    record("cycle_only_baseline.png", "generated for the talk",
           "1_dataset_construction/9_cycle_only_baseline/artifacts/"
           "{full_training_coefficients.json, metrics/*} and "
           "4_training_prefixes/artifacts/training_prefixes.csv",
           "Repository Phase 0 palette; Phase 1 records the baseline's "
           "coefficients and metrics but stores no figure")
    print("wrote cycle_only_baseline.png")


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
    collect_reruns(repo)
    fig_timeline(repo)
    fig_score_ladder()
    fig_cycle_only_baseline(repo)
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
