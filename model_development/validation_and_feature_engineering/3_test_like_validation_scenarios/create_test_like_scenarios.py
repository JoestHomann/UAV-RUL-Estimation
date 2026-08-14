"""Create frozen development and locked scenarios matching test history lengths."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PHASE_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from common import (  # noqa: E402
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
    ID_COLUMN,
    TARGET_COLUMN,
    STEP_1_ARTIFACT_DIR,
    STEP_2_ARTIFACT_DIR,
    STEP_3_ARTIFACT_DIR,
    load_dataset,
    save_csv,
    save_json,
)


def target_at_cutoff(history: pd.DataFrame, cutoff: int) -> float:
    match = history.loc[history["flight_cycle"] == cutoff, TARGET_COLUMN]
    if len(match) != 1:
        raise ValueError(f"Expected exactly one target at cutoff {cutoff}")
    return float(match.iloc[0])


def make_validation_scenarios(
    train: pd.DataFrame,
    outer_folds: pd.DataFrame,
    test_lengths: np.ndarray,
    *,
    scenario_count: int,
    seed: int,
    scenario_prefix: str,
) -> pd.DataFrame:
    grouped = {str(key): value for key, value in train.groupby(ID_COLUMN, sort=True)}
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    rows = list(outer_folds.sort_values(ID_COLUMN).itertuples(index=False))
    if len(test_lengths) != len(rows):
        raise ValueError(
            "Exact test-length matching requires equal train/test UAV counts"
        )
    for scenario_number in range(1, scenario_count + 1):
        scenario = f"{scenario_prefix}_{scenario_number:02d}"
        available = rows.copy()
        assignments: list[tuple[Any, int]] = []
        for cutoff in sorted(test_lengths.astype(int), reverse=True):
            eligible = [
                row for row in available if int(row.final_cycle) - 1 >= int(cutoff)
            ]
            if not eligible:
                raise ValueError(
                    f"No training UAV can support test-like cutoff {int(cutoff)}"
                )
            selected = eligible[int(rng.integers(0, len(eligible)))]
            available.remove(selected)
            assignments.append((selected, int(cutoff)))
        for row, cutoff in assignments:
            uav_id = str(getattr(row, ID_COLUMN))
            records.append(
                {
                    "sample_id": f"{scenario}::{uav_id}::{cutoff:04d}",
                    "scenario": scenario,
                    "outer_fold": int(row.outer_fold),
                    ID_COLUMN: uav_id,
                    "cutoff": cutoff,
                    TARGET_COLUMN: target_at_cutoff(grouped[uav_id], cutoff),
                    "terminal_lifetime": float(row.terminal_lifetime),
                    "lifetime_quantile": int(row.lifetime_quantile),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        ["scenario", "outer_fold", ID_COLUMN]
    )


def make_test_endpoints(test_histories: pd.DataFrame) -> pd.DataFrame:
    endpoints = test_histories[[ID_COLUMN, "final_cycle"]].copy()
    endpoints = endpoints.rename(columns={"final_cycle": "cutoff"})
    endpoints.insert(
        0,
        "sample_id",
        [f"test::{uav_id}::{cutoff:04d}" for uav_id, cutoff in endpoints.values],
    )
    return endpoints


def verify_scenarios(
    table: pd.DataFrame,
    test_lengths: np.ndarray,
    expected_scenarios: int,
) -> None:
    if table["scenario"].nunique() != expected_scenarios:
        raise AssertionError("Scenario count is incorrect")
    for scenario, group in table.groupby("scenario"):
        if group[ID_COLUMN].duplicated().any():
            raise AssertionError(f"{scenario} contains duplicate UAVs")
        if not np.array_equal(
            np.sort(group["cutoff"].to_numpy(dtype=int)),
            np.sort(test_lengths),
        ):
            raise AssertionError(f"{scenario} does not reproduce test history lengths")
        if not (group[TARGET_COLUMN] > 0).all():
            raise AssertionError(f"{scenario} contains terminal RUL=0 samples")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--audit-dir", type=Path, default=STEP_1_ARTIFACT_DIR)
    parser.add_argument("--fold-dir", type=Path, default=STEP_2_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=STEP_3_ARTIFACT_DIR)
    parser.add_argument("--locked-scenarios", type=int, default=20)
    parser.add_argument("--development-scenarios", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    train = load_dataset(args.train_csv, require_target=True)
    test = load_dataset(args.test_csv, require_target=False)
    outer = pd.read_csv(args.fold_dir / "outer_folds.csv")
    test_histories = pd.read_csv(
        args.audit_dir / "test_fligh_cycles_cut_offs.csv"
    )
    test_lengths = test_histories["final_cycle"].to_numpy(dtype=int)

    locked = make_validation_scenarios(
        train,
        outer,
        test_lengths,
        scenario_count=args.locked_scenarios,
        seed=args.seed + 3,
        scenario_prefix="locked",
    )
    development = make_validation_scenarios(
        train,
        outer,
        test_lengths,
        scenario_count=args.development_scenarios,
        seed=args.seed + 4,
        scenario_prefix="development",
    )
    verify_scenarios(locked, test_lengths, args.locked_scenarios)
    verify_scenarios(development, test_lengths, args.development_scenarios)

    paths = [
        save_csv(locked, args.output_dir / "locked_validation_scenarios.csv"),
        save_csv(
            development,
            args.output_dir / "development_validation_scenarios.csv",
        ),
        save_csv(
            make_test_endpoints(test_histories),
            args.output_dir / "test_endpoints.csv",
        ),
        save_json(
            {
                "seed": args.seed,
                "locked_scenarios": args.locked_scenarios,
                "development_scenarios": args.development_scenarios,
                "cutoff_source": "exact empirical test UAV history lengths",
                "terminal_training_cutoffs_excluded": True,
            },
            args.output_dir / "scenario_config.json",
        ),
    ]
    print("\n".join(f"Saved {path}" for path in paths))


if __name__ == "__main__":
    main()
