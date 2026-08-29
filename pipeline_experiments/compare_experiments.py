"""Compare development selections or completed locked pipeline experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib
from typing import Any

import pandas as pd


MANAGER_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MANAGER_DIR.parent
DEFAULT_CONFIG_PATH = MANAGER_DIR / "pipeline_experiments.toml"
OUTPUT_PATH = MANAGER_DIR / "experiment_comparison.csv"
SELECTION_OUTPUT_PATH = MANAGER_DIR / "selection_experiment_comparison.csv"
COMPARISON_SCOPES = ("selection", "locked")


def _read_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Cannot read experiment catalog: {error}") from error


def _repo_path(value: str) -> Path:
    path = (REPOSITORY_ROOT / value).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes the repository: {value}") from error
    return path


def _phase2_root(name: str) -> Path:
    return MANAGER_DIR / "runs" / name / "phase2"


def _not_complete_row(
    name: str,
    run_number: Any,
    scope: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment": name,
                "status": "not_complete",
                "comparison_scope": scope,
                "phase_2_run_number": run_number,
            }
        ]
    )


def _selection_table(
    name: str,
    experiment: dict[str, Any],
) -> pd.DataFrame | None:
    path = _phase2_root(name) / "5_inner_model_selection" / "candidate_results.csv"
    manifest = _phase2_root(name) / "5_inner_model_selection" / "selection_manifest.json"
    if not path.is_file() or not manifest.is_file():
        return None

    table = pd.read_csv(path)
    required = {
        "model_family",
        "outer_fold",
        "selected_within_family",
        "mean_inner_rmse",
        "mean_inner_r2",
        "mean_inner_bias",
        "mean_inner_overprediction_rate",
        "mean_inner_root_mean_squared_overprediction",
        "mean_inner_underprediction_rate",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(
            f"Selection results for {name!r} are missing columns: {missing}"
        )

    selected_values = table["selected_within_family"]
    if selected_values.dtype == bool:
        selected_mask = selected_values
    else:
        normalized = selected_values.astype(str).str.strip().str.lower()
        invalid = sorted(set(normalized) - {"true", "false"})
        if invalid:
            raise ValueError(
                f"Selection results for {name!r} contain invalid selected flags: {invalid}"
            )
        selected_mask = normalized == "true"
    selected = table.loc[selected_mask].copy()
    if selected.empty:
        raise ValueError(f"Selection results for {name!r} have no selected candidates")
    if selected.duplicated(["model_family", "outer_fold"]).any():
        raise ValueError(
            f"Selection results for {name!r} contain multiple selected candidates "
            "for one family/fold"
        )

    expected_families = set(experiment.get("architectures", []))
    selected_families = set(selected["model_family"])
    if selected_families != expected_families:
        raise ValueError(
            f"Selection results for {name!r} have families "
            f"{sorted(selected_families)}, expected {sorted(expected_families)}"
        )

    result = (
        selected.groupby("model_family", sort=True)
        .agg(
            outer_fold_studies=("outer_fold", "nunique"),
            inner_rmse_mean=("mean_inner_rmse", "mean"),
            inner_rmse_fold_sd=("mean_inner_rmse", "std"),
            inner_rmse_min=("mean_inner_rmse", "min"),
            inner_rmse_max=("mean_inner_rmse", "max"),
            inner_r2_mean=("mean_inner_r2", "mean"),
            inner_r2_fold_sd=("mean_inner_r2", "std"),
            inner_bias_mean=("mean_inner_bias", "mean"),
            inner_overprediction_rate_mean=(
                "mean_inner_overprediction_rate",
                "mean",
            ),
            inner_rms_overprediction_mean=(
                "mean_inner_root_mean_squared_overprediction",
                "mean",
            ),
            inner_underprediction_rate_mean=(
                "mean_inner_underprediction_rate",
                "mean",
            ),
        )
        .reset_index()
    )
    result.insert(0, "experiment", name)
    result.insert(1, "status", "complete")
    result.insert(2, "comparison_scope", "selection")
    result.insert(3, "phase_2_run_number", experiment.get("phase_2_run_number"))
    result.insert(4, "scenario_profile", experiment.get("scenario_profile"))
    result.insert(5, "target_profile", experiment.get("target_profile"))
    result.insert(6, "prediction_profile", experiment.get("prediction_profile"))
    return result


def _locked_table(
    name: str,
    experiment: dict[str, Any],
) -> pd.DataFrame | None:
    path = _phase2_root(name) / "7_architecture_comparison" / "architecture_comparison.csv"
    if not path.is_file():
        return None
    table = pd.read_csv(path)
    table.insert(0, "experiment", name)
    table.insert(1, "status", "complete")
    table.insert(2, "comparison_scope", "locked")
    table.insert(3, "phase_2_run_number", experiment.get("phase_2_run_number"))

    leaderboard_path = MANAGER_DIR / "runs" / name / "leaderboard_result.json"
    if leaderboard_path.is_file():
        record = json.loads(leaderboard_path.read_text(encoding="utf-8"))
        table["leaderboard_metric"] = record.get("metric")
        table["leaderboard_score"] = record.get("public_score")
    return table


def compare(
    config_path: Path,
    names: list[str] | None = None,
    *,
    scope: str = "locked",
    output_path: Path | None = None,
) -> pd.DataFrame:
    if scope not in COMPARISON_SCOPES:
        raise ValueError(f"scope must be one of: {', '.join(COMPARISON_SCOPES)}")
    config = _read_config(config_path)
    experiments = config.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ValueError("The catalog has no experiments table")
    selected = sorted(names or experiments)
    rows: list[pd.DataFrame] = []
    for name in selected:
        experiment = experiments.get(name)
        if not isinstance(experiment, dict):
            raise ValueError(f"Unknown experiment {name!r}")
        table = (
            _selection_table(name, experiment)
            if scope == "selection"
            else _locked_table(name, experiment)
        )
        if table is None:
            rows.append(
                _not_complete_row(
                    name,
                    experiment.get("phase_2_run_number"),
                    scope,
                )
            )
            continue
        rows.append(table)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True, sort=False)
    destination = output_path or (
        SELECTION_OUTPUT_PATH if scope == "selection" else OUTPUT_PATH
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run", action="append", dest="names")
    parser.add_argument("--scope", choices=COMPARISON_SCOPES, default="locked")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output_path = args.output or (
        SELECTION_OUTPUT_PATH if args.scope == "selection" else OUTPUT_PATH
    )
    try:
        result = compare(
            args.config.resolve(),
            args.names,
            scope=args.scope,
            output_path=output_path.resolve(),
        )
    except (ValueError, OSError, pd.errors.ParserError, json.JSONDecodeError) as error:
        print(f"Experiment comparison failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"Compared {len(result)} rows")
    print(f"Saved {output_path.resolve()}")


if __name__ == "__main__":
    main()
