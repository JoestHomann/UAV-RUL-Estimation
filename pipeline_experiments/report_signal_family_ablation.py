"""Build paired development-fold reporting for an explicit experiment group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_paths import artifact_directory


REPOSITORY_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "pipeline_experiments.toml"
RESULT_COLUMNS = {
    "mean_inner_r2": "r2",
    "mean_inner_rmse": "rmse",
    "mean_inner_bias": "bias",
    "mean_inner_overprediction_rate": "overprediction_rate",
    "mean_inner_root_mean_squared_overprediction": "rms_overprediction",
}
DISPLAY_NAMES = {
    "signal_control": "control",
    "signal_family_13_16_22_25_28": "13/16/22/25/28",
    "signal_family_19_21": "19/21",
    "signal_family_15_23": "15/23",
    "signal_family_07": "07",
    "signal_all_families": "all families",
}


class AblationReportError(ValueError):
    """Explain missing or inconsistent ablation results."""


def _read_config(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            with path.open("rb") as stream:
                payload = tomllib.load(stream)
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        raise AblationReportError(f"Cannot read experiment catalog: {error}") from error
    if not isinstance(payload, dict):
        raise AblationReportError("Experiment catalog must contain an object")
    return payload


def _selected_rows(path: Path) -> pd.DataFrame:
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise AblationReportError(f"Cannot read selection results at {path}: {error}") from error
    required = {
        "model_family",
        "outer_fold",
        "feature_set",
        "selected_within_family",
        *RESULT_COLUMNS,
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise AblationReportError(f"Selection results at {path} are missing {missing}")
    selected_flag = table["selected_within_family"]
    if selected_flag.dtype != bool:
        selected_flag = selected_flag.astype(str).str.strip().str.lower() == "true"
    selected = table.loc[selected_flag, list(required)].copy()
    if selected.empty:
        raise AblationReportError(f"Selection results at {path} have no selected rows")
    duplicates = selected.duplicated(["model_family", "outer_fold"])
    if duplicates.any():
        raise AblationReportError(
            f"Selection results at {path} select multiple candidates for one model/fold"
        )
    return selected.rename(columns=RESULT_COLUMNS)


def collect_results(
    config: dict[str, Any],
    group_name: str,
) -> tuple[pd.DataFrame, str]:
    groups = config.get("experiment_groups", {})
    experiments = config.get("experiments", {})
    group = groups.get(group_name) if isinstance(groups, dict) else None
    if not isinstance(group, dict):
        raise AblationReportError(f"Unknown experiment group {group_name!r}")
    members = group.get("experiments")
    control = group.get("control")
    if not isinstance(members, list) or control not in members:
        raise AblationReportError(f"Invalid experiment group {group_name!r}")

    frames: list[pd.DataFrame] = []
    for experiment_name in members:
        experiment = experiments.get(experiment_name)
        if not isinstance(experiment, dict):
            raise AblationReportError(f"Unknown component experiment {experiment_name!r}")
        feature_set = experiment.get("feature_set")
        path = (
            artifact_directory(
                SCRIPT_DIR / "runs",
                experiment_name,
                experiment,
            )
            / "phase2"
            / "5_inner_model_selection"
            / "candidate_results.csv"
        )
        selected = _selected_rows(path)
        observed_sets = set(selected["feature_set"].astype(str))
        if observed_sets != {feature_set}:
            raise AblationReportError(
                f"{experiment_name} selected feature sets {sorted(observed_sets)}, "
                f"expected {feature_set!r}"
            )
        selected.insert(0, "experiment", experiment_name)
        selected.insert(2, "target_profile", experiment.get("target_profile", "raw"))
        selected.insert(3, "prefix_variant", experiment.get("prefix_variant"))
        selected.insert(
            4,
            "fault_mode_strategy",
            experiment.get("fault_mode_strategy", "none"),
        )
        selected.insert(
            5,
            "signal_compression_strategy",
            experiment.get("signal_compression_strategy", "none"),
        )
        selected.insert(
            6,
            "prediction_profile",
            experiment.get("prediction_profile", "symmetric"),
        )
        frames.append(selected)
    return pd.concat(frames, ignore_index=True), str(control)


def pair_with_control(results: pd.DataFrame, control: str) -> pd.DataFrame:
    keys = ["model_family", "outer_fold"]
    metric_names = list(RESULT_COLUMNS.values())
    control_rows = results.loc[results["experiment"] == control, [*keys, *metric_names]]
    if control_rows.empty:
        raise AblationReportError(f"Control experiment {control!r} has no selected results")
    control_rows = control_rows.rename(
        columns={name: f"control_{name}" for name in metric_names}
    )
    paired = results.merge(control_rows, on=keys, how="left", validate="many_to_one")
    if paired[[f"control_{name}" for name in metric_names]].isna().any().any():
        raise AblationReportError("A treatment model/fold has no matching control result")

    paired["r2_improvement"] = paired["r2"] - paired["control_r2"]
    paired["rmse_improvement"] = paired["control_rmse"] - paired["rmse"]
    paired["absolute_bias_improvement"] = (
        paired["control_bias"].abs() - paired["bias"].abs()
    )
    paired["overprediction_rate_improvement"] = (
        paired["control_overprediction_rate"] - paired["overprediction_rate"]
    )
    paired["rms_overprediction_improvement"] = (
        paired["control_rms_overprediction"] - paired["rms_overprediction"]
    )
    paired["r2_fold_win"] = paired["r2_improvement"] > 0.0
    paired["rmse_fold_win"] = paired["rmse_improvement"] > 0.0
    return paired.sort_values(["experiment", "model_family", "outer_fold"])


def summarize_pairs(paired: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "target_profile": "raw",
        "prefix_variant": "unknown",
        "fault_mode_strategy": "none",
        "signal_compression_strategy": "none",
        "prediction_profile": "symmetric",
    }
    paired = paired.copy()
    for column, value in defaults.items():
        if column not in paired.columns:
            paired[column] = value
    group_columns = [
        "experiment",
        "feature_set",
        "target_profile",
        "prefix_variant",
        "fault_mode_strategy",
        "signal_compression_strategy",
        "prediction_profile",
        "model_family",
    ]
    summary = paired.groupby(group_columns, as_index=False).agg(
        folds=("outer_fold", "count"),
        mean_r2=("r2", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_bias=("bias", "mean"),
        mean_overprediction_rate=("overprediction_rate", "mean"),
        mean_rms_overprediction=("rms_overprediction", "mean"),
        mean_r2_improvement=("r2_improvement", "mean"),
        sd_r2_improvement=("r2_improvement", "std"),
        mean_rmse_improvement=("rmse_improvement", "mean"),
        sd_rmse_improvement=("rmse_improvement", "std"),
        mean_absolute_bias_improvement=("absolute_bias_improvement", "mean"),
        mean_overprediction_rate_improvement=(
            "overprediction_rate_improvement",
            "mean",
        ),
        mean_rms_overprediction_improvement=(
            "rms_overprediction_improvement",
            "mean",
        ),
        r2_fold_wins=("r2_fold_win", "sum"),
        rmse_fold_wins=("rmse_fold_win", "sum"),
    )
    summary[["sd_r2_improvement", "sd_rmse_improvement"]] = summary[
        ["sd_r2_improvement", "sd_rmse_improvement"]
    ].fillna(0.0)
    return summary.sort_values(
        ["model_family", "mean_r2_improvement", "mean_rmse_improvement"],
        ascending=[True, False, False],
    )


def _plot(summary: pd.DataFrame, control: str, output_path: Path) -> None:
    treatments = summary.loc[summary["experiment"] != control].copy()
    experiment_order = list(dict.fromkeys(treatments["experiment"].astype(str)))
    model_order = sorted(treatments["model_family"].astype(str).unique())
    x = np.arange(len(experiment_order), dtype=float)
    width = 0.8 / max(len(model_order), 1)
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for index, model in enumerate(model_order):
        model_rows = treatments.loc[treatments["model_family"] == model].set_index(
            "experiment"
        )
        positions = x + (index - (len(model_order) - 1) / 2) * width
        axes[0].bar(
            positions,
            model_rows.reindex(experiment_order)["mean_r2_improvement"],
            width,
            label=model,
        )
        axes[1].bar(
            positions,
            model_rows.reindex(experiment_order)["mean_rmse_improvement"],
            width,
            label=model,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Paired R2 improvement")
    axes[1].set_ylabel("Paired RMSE improvement")
    feature_counts = treatments.groupby("feature_set")["experiment"].nunique()
    labels = []
    for experiment in experiment_order:
        row = model_rows.loc[experiment]
        feature_set = str(row["feature_set"])
        if feature_counts.get(feature_set, 0) > 1:
            labels.append(experiment.removeprefix("PE3_").removeprefix("PE_"))
        else:
            labels.append(DISPLAY_NAMES.get(feature_set, experiment.removeprefix("PE_")))
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_xlabel("Treatment experiment")
    axes[0].legend(frameon=False, ncol=max(len(model_order), 1))
    figure.suptitle("Development-fold paired experiment comparison")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(
    config: dict[str, Any],
    group_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    results, control = collect_results(config, group_name)
    paired = pair_with_control(results, control)
    summary = summarize_pairs(paired)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "paired_fold_results.csv"
    summary_path = output_dir / "paired_summary.csv"
    figure_path = output_dir / "paired_comparison.png"
    paired.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    _plot(summary, control, figure_path)

    treatment_summary = summary.loc[summary["experiment"] != control]
    best_by_model: dict[str, dict[str, Any]] = {}
    for model, rows in treatment_summary.groupby("model_family", sort=True):
        best = rows.sort_values(
            ["mean_r2_improvement", "mean_rmse_improvement"],
            ascending=False,
        ).iloc[0]
        best_by_model[str(model)] = {
            "experiment": str(best["experiment"]),
            "feature_set": str(best["feature_set"]),
            "mean_r2_improvement": float(best["mean_r2_improvement"]),
            "mean_rmse_improvement": float(best["mean_rmse_improvement"]),
            "r2_fold_wins": int(best["r2_fold_wins"]),
            "folds": int(best["folds"]),
        }
    manifest = {
        "status": "complete",
        "group": group_name,
        "control_experiment": control,
        "uses_locked_evaluation": False,
        "interpretation": (
            "Positive paired improvements favor the treatment. A treatment should be "
            "retained only when gains are consistent across outer folds and models."
        ),
        "best_mean_r2_treatment_by_model": best_by_model,
        "artifacts": {
            "fold_results": fold_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "summary": summary_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "figure": figure_path.relative_to(REPOSITORY_ROOT).as_posix(),
        },
    }
    manifest_path = output_dir / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--group", default="PE_signal_family_ablation")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = _read_config(args.config.resolve())
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = SCRIPT_DIR / "runs" / args.group / "reporting"
    try:
        manifest = write_report(config, args.group, output_dir.resolve())
    except AblationReportError as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
