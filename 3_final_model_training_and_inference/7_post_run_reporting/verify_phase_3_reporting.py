"""Verify model-agnostic Phase 3 reporting with synthetic shared artifacts."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from build_phase_3_report import (  # noqa: E402
    _display_name,
    _fold_heatmap,
    _input_alternative_performance,
    _input_choices,
    _optimization_history,
    _performance_efficiency,
    _selected_fold_metrics,
    _test_prediction_diagnostics,
    _top_candidate_robustness,
)


FAMILIES = (
    "mean_baseline",
    "cycle_only_baseline",
    "regularized_linear",
    "random_forest",
    "extra_trees",
    "xgboost",
    "catboost",
    "mlp",
    "tcn",
    "multiscale_cnn",
    "sensor_graph_tcn",
    "lstm",
    "transformer",
    "rbf_svr",
    "trajectory_dtw_knn",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _baseline_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_number": [1],
            "feature_set": [np.nan],
            "lookback": [np.nan],
            "mean_fold_rmse": [38.0],
            "fold_rmse_standard_deviation": [2.0],
            "mean_fold_r2": [0.4],
            "mean_fold_mae": [30.0],
            "mean_fold_bias": [1.0],
            "mean_training_seconds": [0.01],
            "total_training_seconds": [0.05],
            "final_training_iterations": [np.nan],
        }
    )


def _baseline_folds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_number": [1] * 5,
            "outer_fold": list(range(5)),
            "rmse": [35.0, 37.0, 38.0, 39.0, 41.0],
            "r2": [0.5, 0.4, 0.45, 0.35, 0.3],
            "mae": [27.0, 29.0, 30.0, 31.0, 33.0],
            "bias": [-2.0, -1.0, 0.0, 2.0, 6.0],
            "training_seconds": [0.01] * 5,
            "best_epoch_or_iteration": [np.nan] * 5,
        }
    )


def _verify_baseline_plots() -> int:
    candidates = _baseline_candidates()
    folds = _baseline_folds()
    predictions = pd.DataFrame(
        {
            "uav_id": ["test_1", "test_2", "test_3"],
            "cutoff": [50, 100, 150],
            "RUL": [120.0, 80.0, 40.0],
        }
    )
    output = STEP_DIR / ".verification_outputs"
    output.mkdir(exist_ok=False)
    paths: list[Path] = []
    try:
        paths = [
            _optimization_history(candidates, 1, "mean_baseline", output / "a.png"),
            _top_candidate_robustness(
                candidates,
                folds,
                1,
                "mean_baseline",
                output / "b.png",
            ),
            _fold_heatmap(candidates, folds, 1, "mean_baseline", output / "c.png"),
            _selected_fold_metrics(folds, 1, "mean_baseline", output / "d.png"),
            _input_alternative_performance(
                candidates,
                1,
                "mean_baseline",
                output / "e.png",
            ),
            _performance_efficiency(
                candidates,
                1,
                "mean_baseline",
                output / "f.png",
            ),
            _test_prediction_diagnostics(
                predictions,
                "mean_baseline",
                output / "g.png",
            ),
        ]
        _require(
            all(path.is_file() and path.stat().st_size > 0 for path in paths),
            "A fixed-input single-candidate baseline plot was not written",
        )
        return len(paths)
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
        output.rmdir()


def _verify_sequence_choices() -> None:
    candidates = pd.concat([_baseline_candidates()] * 3, ignore_index=True)
    candidates["candidate_number"] = [1, 2, 3]
    candidates["lookback"] = [50, 100, 200]
    choices, choice_name = _input_choices(candidates)
    _require(choice_name == "Lookback", "Sequence alternatives are not lookbacks")
    _require(
        choices.tolist() == ["Lookback 50", "Lookback 100", "Lookback 200"],
        "Sequence lookback labels changed",
    )


def main() -> None:
    try:
        for family in FAMILIES:
            _require(bool(_display_name(family)), f"Family {family} has no label")
        plot_count = _verify_baseline_plots()
        _verify_sequence_choices()
    except (RuntimeError, OSError, ValueError) as error:
        print(f"Phase 3 reporting verification failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("Phase 3 reporting verified")
    print(f"Registered families checked: {len(FAMILIES)}")
    print(f"Single-candidate baseline plots checked: {plot_count}")
    print("Sequence lookback alternatives checked: 3")


if __name__ == "__main__":
    main()
