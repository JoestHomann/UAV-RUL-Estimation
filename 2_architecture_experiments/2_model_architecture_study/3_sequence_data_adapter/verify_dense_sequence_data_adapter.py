"""Verify dense-ready causal sequence views and grouped weighting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sequence_data_adapter import SequenceDataAdapter, SequenceAdapterError


def verify(
    manifest: Path,
    requested: list[int] | None = None,
    output: Path | None = None,
) -> None:
    adapter = SequenceDataAdapter(manifest)
    lookbacks = tuple(requested or (20, 30, 50))
    if any(value <= 0 for value in lookbacks):
        raise SequenceAdapterError("Verification lookbacks must be positive")
    # The adapter algorithm is lookback-agnostic. Its normal public contract
    # restricts values to the active run manifest; this verifier deliberately
    # widens that in-memory allow-list to cover the planned experiment values.
    adapter.lookbacks = lookbacks

    records = []
    for lookback in lookbacks:
        training = adapter.load_training(lookback)
        if training.sequences.shape[1] != lookback:
            raise SequenceAdapterError("Materialized lookback changed")
        if np.any(training.sequences[training.padding_mask] != 0.0):
            raise SequenceAdapterError("Raw padded sequence positions are not zero")
        if training.sample_weights is None:
            raise SequenceAdapterError("Training sequence weights are missing")
        weight_table = training.metadata[["uav_id"]].copy()
        weight_table["sample_weight"] = training.sample_weights.to_numpy()
        sums = weight_table.groupby("uav_id")["sample_weight"].sum().to_numpy()
        if not np.allclose(sums, 1.0, rtol=1e-10, atol=1e-10):
            raise SequenceAdapterError("Sequence UAV weight totals differ from one")

        split = adapter.get_inner_selection_split(0, 0, lookback)
        training_ids = set(split.training.metadata["uav_id"].astype(str))
        validation_ids = set(split.validation.metadata["uav_id"].astype(str))
        if training_ids & validation_ids:
            raise SequenceAdapterError("Inner sequence split leaks UAV identifiers")
        for dataset in (split.training, split.validation):
            if not dataset.scaled:
                raise SequenceAdapterError("Fold sequence view was not scaled")
            if np.any(dataset.sequences[dataset.padding_mask] != 0.0):
                raise SequenceAdapterError("Scaled padded positions are not zero")
            if not np.isfinite(dataset.sequences).all():
                raise SequenceAdapterError("Scaled sequences contain non-finite values")
        records.append(
            {
                "lookback": lookback,
                "training_rows": len(training),
                "training_uavs": int(training.metadata["uav_id"].nunique()),
                "minimum_uav_weight_total": float(sums.min()),
                "maximum_uav_weight_total": float(sums.max()),
                "short_history_rows": int(training.padding_mask.any(axis=1).sum()),
                "padding_remains_zero": True,
                "inner_fold_uav_overlap": 0,
            }
        )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "manifest": str(manifest),
                    "lookbacks": records,
                    "equal_total_weight_per_uav": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    print("Dense sequence adapter verification passed")
    print(f"Lookbacks: {list(lookbacks)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "sequence_dataset_manifest.json",
    )
    parser.add_argument("--lookback", type=int, action="append")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "artifacts"
        / "dense_sequence_verification.json",
    )
    args = parser.parse_args()
    verify(args.manifest.resolve(), args.lookback, args.output.resolve())


if __name__ == "__main__":
    main()
