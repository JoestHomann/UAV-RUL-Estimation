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


def stratified_empirical_cutoffs(
    test_lengths: np.ndarray,
    *,
    maximum_cutoff: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample distinct cutoffs across test age bands with empirical weights."""

    eligible = test_lengths[test_lengths <= maximum_cutoff].astype(int)
    unique, frequencies = np.unique(eligible, return_counts=True)
    if not len(unique):
        raise ValueError(f"No test-like cutoff is eligible below {maximum_cutoff}")
    sample_count = min(count, len(unique))
    band_edges = np.array([50, 100, 200], dtype=int)
    unique_bands = np.digitize(unique, band_edges, right=True)
    eligible_bands = np.digitize(eligible, band_edges, right=True)
    active_bands = sorted(np.unique(unique_bands).tolist())
    band_frequencies = np.array(
        [np.count_nonzero(eligible_bands == band) for band in active_bands],
        dtype=float,
    )
    desired = sample_count * band_frequencies / band_frequencies.sum()
    allocations = np.floor(desired).astype(int)
    if sample_count >= len(active_bands):
        allocations = np.maximum(allocations, 1)
    capacities = np.array(
        [np.count_nonzero(unique_bands == band) for band in active_bands],
        dtype=int,
    )
    allocations = np.minimum(allocations, capacities)
    while allocations.sum() < sample_count:
        available = allocations < capacities
        if not available.any():
            break
        deficits = np.where(available, desired - allocations, -np.inf)
        allocations[int(np.argmax(deficits))] += 1
    while allocations.sum() > sample_count:
        removable = allocations > 0
        if sample_count >= len(active_bands):
            removable &= allocations > 1
        excess = np.where(removable, allocations - desired, -np.inf)
        allocations[int(np.argmax(excess))] -= 1

    selected: list[int] = []
    for band, allocation in zip(active_bands, allocations, strict=True):
        if allocation == 0:
            continue
        mask = unique_bands == band
        values = unique[mask]
        weights = frequencies[mask].astype(float)
        weights /= weights.sum()
        selected.extend(
            rng.choice(
                values,
                size=int(allocation),
                replace=False,
                p=weights,
            ).astype(int).tolist()
        )
    return np.asarray(selected, dtype=int)


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
    strategy: str = "empirical",
) -> pd.DataFrame:
    grouped = {str(key): value for key, value in train.groupby(ID_COLUMN, sort=True)}
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for row in histories.sort_values(ID_COLUMN).itertuples(index=False):
        uav_id = str(getattr(row, ID_COLUMN))
        sampler = (
            empirical_cutoffs
            if strategy == "empirical"
            else stratified_empirical_cutoffs
        )
        if strategy not in {"empirical", "stratified_empirical"}:
            raise ValueError(f"Unknown prefix strategy {strategy!r}")
        cutoffs = sampler(
            test_lengths,
            maximum_cutoff=int(row.final_cycle) - 1,
            count=cutoffs_per_uav,
            rng=rng,
        )
        prefix_weight = 1.0 / len(cutoffs)
        for prefix_number, cutoff in enumerate(sorted(cutoffs), start=1):
            records.append(
                {
                    "sample_id": f"train::{uav_id}::{cutoff:04d}",
                    ID_COLUMN: uav_id,
                    "prefix_number": prefix_number,
                    "cutoff": int(cutoff),
                    TARGET_COLUMN: target_at_cutoff(grouped[uav_id], int(cutoff)),
                    "sample_weight": prefix_weight,
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
    parser.add_argument(
        "--strategy",
        choices=("empirical", "stratified_empirical"),
        default="empirical",
    )
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
        strategy=args.strategy,
    )

    prefix_counts = training.groupby(ID_COLUMN).size()
    if args.strategy == "empirical" and not prefix_counts.eq(
        args.cutoffs_per_uav
    ).all():
        raise AssertionError("Empirical training cutoff counts are not equal by UAV")
    if (prefix_counts > args.cutoffs_per_uav).any():
        raise AssertionError("Training cutoff count exceeds the configured maximum")
    if training.duplicated([ID_COLUMN, "cutoff"]).any():
        raise AssertionError("Training prefixes contain duplicate UAV/cutoff pairs")
    if not (training[TARGET_COLUMN] > 0).all():
        raise AssertionError("Training prefixes contain terminal RUL=0 samples")
    if not np.allclose(
        training.groupby(ID_COLUMN)["sample_weight"].sum().to_numpy(),
        1.0,
    ):
        raise AssertionError("Each UAV must have equal total sample weight")

    paths = [
        save_csv(training, args.output_dir / "training_prefixes.csv"),
        save_json(
            {
                "seed": args.seed,
                "strategy": args.strategy,
                "cutoffs_per_uav": args.cutoffs_per_uav,
                "actual_prefixes_per_uav_minimum": int(prefix_counts.min()),
                "actual_prefixes_per_uav_maximum": int(prefix_counts.max()),
                "cutoff_source": "empirical test UAV history lengths",
                "equal_total_weight_per_uav": True,
            },
            args.output_dir / "training_prefix_config.json",
        ),
    ]
    print("\n".join(f"Saved {path}" for path in paths))


if __name__ == "__main__":
    main()
