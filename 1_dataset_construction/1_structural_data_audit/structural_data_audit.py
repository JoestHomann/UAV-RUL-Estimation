"""Validate raw train/test structure and save per-UAV history summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PHASE_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from common import (  # noqa: E402
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
    ID_COLUMN,
    TELEMETRY_COLUMNS,
    STEP_1_ARTIFACT_DIR,
    file_sha256,
    load_dataset,
    save_csv,
    save_json,
    summarize_histories,
)


def dataset_audit(
    train_path: Path,
    test_path: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_histories: pd.DataFrame,
    test_histories: pd.DataFrame,
) -> dict[str, Any]:
    overlap = sorted(set(train[ID_COLUMN]) & set(test[ID_COLUMN]))
    if overlap:
        raise ValueError(f"Train/test UAV IDs overlap: {overlap}")
    return {
        "train": {
            "path": str(train_path.resolve()),
            "sha256": file_sha256(train_path),
            "rows": int(len(train)),
            "columns": int(train.shape[1]),
            "column_order": train.columns.tolist(),
            "dtypes": {column: str(dtype) for column, dtype in train.dtypes.items()},
            "uavs": int(train[ID_COLUMN].nunique()),
            "minimum_history_length": int(train_histories["final_cycle"].min()),
            "maximum_history_length": int(train_histories["final_cycle"].max()),
        },
        "test": {
            "path": str(test_path.resolve()),
            "sha256": file_sha256(test_path),
            "rows": int(len(test)),
            "columns": int(test.shape[1]),
            "column_order": test.columns.tolist(),
            "dtypes": {column: str(dtype) for column, dtype in test.dtypes.items()},
            "uavs": int(test[ID_COLUMN].nunique()),
            "minimum_history_length": int(test_histories["final_cycle"].min()),
            "maximum_history_length": int(test_histories["final_cycle"].max()),
        },
        "assertions": {
            "required_columns": True,
            "test_has_no_target": True,
            "no_missing_or_nonfinite_values": True,
            "no_duplicate_rows_or_keys": True,
            "train_test_uav_ids_disjoint": True,
            "histories_start_at_cycle_1": True,
            "cycles_ordered_and_consecutive": True,
            "rul_plus_cycle_constant_within_training_uav": True,
            "training_histories_end_at_rul_0": True,
        },
        "telemetry_columns": TELEMETRY_COLUMNS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=STEP_1_ARTIFACT_DIR)
    args = parser.parse_args()

    train = load_dataset(args.train_csv, require_target=True)
    test = load_dataset(args.test_csv, require_target=False)
    train_histories = summarize_histories(train, require_target=True)
    test_histories = summarize_histories(test, require_target=False)

    paths = [
        save_csv(train_histories, args.output_dir / "train_flight_cycles.csv"),
        save_csv(
            test_histories,
            args.output_dir / "test_fligh_cycles_cut_offs.csv",
        ),
        save_json(
            dataset_audit(
                args.train_csv,
                args.test_csv,
                train,
                test,
                train_histories,
                test_histories,
            ),
            args.output_dir / "dataset_audit.json",
        ),
    ]
    print("\n".join(f"Saved {path}" for path in paths))


if __name__ == "__main__":
    main()
