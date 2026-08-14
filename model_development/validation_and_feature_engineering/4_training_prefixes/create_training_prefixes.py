"""Create equally weighted, test-like training prefixes for every UAV."""

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
    DEFAULT_TRAIN_CSV,
    ID_COLUMN,
    TARGET_COLUMN,
    STEP_1_ARTIFACT_DIR,
    STEP_4_ARTIFACT_DIR,
    load_dataset,
    save_csv,
    save_json,
)


def empirical_cutoffs(
    test_lengths: np.ndarray,
    *,
    maximum_cutoff: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    eligible = test_lengths[test_lengths <= maximum_cutoff]
    unique, frequencies = np.unique(eligible.astype(int), return_counts=True)
    if len(unique) < count:
        raise ValueError(
            f"Only {len(unique)} distinct test-like cutoffs are available at "
            f"maximum cutoff {maximum_cutoff}; requested {count}"
        )
    probabilities = frequencies.astype(float) / frequencies.sum()
    return rng.choice(
        unique,
        size=count,
        replace=False,
        p=probabilities,
    ).astype(int)


def target_at_cutoff(history: pd.DataFrame, cutoff: int) -> float:
    match = history.loc[history["flight_cycle"] == cutoff, TARGET_COLUMN]
    if len(match) != 1:
        raise ValueError(f"Expected exactly one target at cutoff {cutoff}")
    return float(match.iloc[0])


def make_training_prefixes(
    train: pd.DataFrame,
    histories: pd.DataFrame,
    test_lengths: np.ndarray,
    *,
    cutoffs_per_uav: int,
    seed: int,
) -> pd.DataFrame:
    grouped = {str(key): value for key, value in train.groupby(ID_COLUMN, sort=True)}
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for row in histories.sort_values(ID_COLUMN).itertuples(index=False):
        uav_id = str(getattr(row, ID_COLUMN))
        cutoffs = empirical_cutoffs(
            test_lengths,
            maximum_cutoff=int(row.final_cycle) - 1,
            count=cutoffs_per_uav,
            rng=rng,
        )
        for prefix_number, cutoff in enumerate(sorted(cutoffs), start=1):
            records.append(
                {
                    "sample_id": f"train::{uav_id}::{cutoff:04d}",
                    ID_COLUMN: uav_id,
                    "prefix_number": prefix_number,
                    "cutoff": int(cutoff),
                    TARGET_COLUMN: target_at_cutoff(grouped[uav_id], int(cutoff)),
                    "sample_weight": 1.0 / cutoffs_per_uav,
                }
            )
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--audit-dir", type=Path, default=STEP_1_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=STEP_4_ARTIFACT_DIR)
    parser.add_argument("--cutoffs-per-uav", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    train = load_dataset(args.train_csv, require_target=True)
    histories = pd.read_csv(args.audit_dir / "train_flight_cycles.csv")
    test_histories = pd.read_csv(
        args.audit_dir / "test_fligh_cycles_cut_offs.csv"
    )
    training = make_training_prefixes(
        train,
        histories,
        test_histories["final_cycle"].to_numpy(dtype=int),
        cutoffs_per_uav=args.cutoffs_per_uav,
        seed=args.seed + 2,
    )

    if not training.groupby(ID_COLUMN).size().eq(args.cutoffs_per_uav).all():
        raise AssertionError("Training cutoff counts are not equal by UAV")
    if training.duplicated([ID_COLUMN, "cutoff"]).any():
        raise AssertionError("Training prefixes contain duplicate UAV/cutoff pairs")
    if not (training[TARGET_COLUMN] > 0).all():
        raise AssertionError("Training prefixes contain terminal RUL=0 samples")
    if not training.groupby(ID_COLUMN)["sample_weight"].sum().eq(1.0).all():
        raise AssertionError("Each UAV must have equal total sample weight")

    paths = [
        save_csv(training, args.output_dir / "training_prefixes.csv"),
        save_json(
            {
                "seed": args.seed,
                "cutoffs_per_uav": args.cutoffs_per_uav,
                "cutoff_source": "empirical test UAV history lengths",
                "equal_total_weight_per_uav": True,
            },
            args.output_dir / "training_prefix_config.json",
        ),
    ]
    print("\n".join(f"Saved {path}" for path in paths))


if __name__ == "__main__":
    main()
