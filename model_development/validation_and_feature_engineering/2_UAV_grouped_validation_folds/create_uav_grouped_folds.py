"""Create balanced, disjoint outer and inner validation folds by UAV ID."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PHASE_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from common import (  # noqa: E402
    ID_COLUMN,
    STEP_1_ARTIFACT_DIR,
    STEP_2_ARTIFACT_DIR,
    save_csv,
    save_json,
)


def balanced_group_folds(
    histories: pd.DataFrame,
    *,
    n_folds: int,
    seed: int,
    fold_column: str,
) -> pd.DataFrame:
    if len(histories) < n_folds * 2:
        raise ValueError("Too few UAVs for the requested fold count")
    table = histories.copy().sort_values(["terminal_lifetime", ID_COLUMN])
    ranks = table["terminal_lifetime"].rank(method="first")
    table["lifetime_quantile"] = pd.qcut(
        ranks,
        q=n_folds,
        labels=False,
        duplicates="raise",
    ).astype(int)
    assignments: dict[str, int] = {}
    rng = np.random.default_rng(seed)
    for quantile, group in table.groupby("lifetime_quantile", sort=True):
        ids = group[ID_COLUMN].astype(str).to_numpy(copy=True)
        rng.shuffle(ids)
        offset = int(quantile) % n_folds
        for position, uav_id in enumerate(ids):
            assignments[str(uav_id)] = int((position + offset) % n_folds)
    table[fold_column] = table[ID_COLUMN].map(assignments).astype(int)
    return table.sort_values(ID_COLUMN).reset_index(drop=True)


def make_inner_folds(
    outer_folds: pd.DataFrame,
    *,
    n_inner_folds: int,
    seed: int,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for outer_fold in sorted(outer_folds["outer_fold"].unique()):
        outer_train = outer_folds.loc[
            outer_folds["outer_fold"] != outer_fold,
            [ID_COLUMN, "row_count", "final_cycle", "terminal_lifetime"],
        ]
        inner = balanced_group_folds(
            outer_train,
            n_folds=n_inner_folds,
            seed=seed + 1009 * int(outer_fold),
            fold_column="inner_fold",
        )
        inner.insert(0, "outer_fold", int(outer_fold))
        records.append(inner)
    return pd.concat(records, ignore_index=True).sort_values(
        ["outer_fold", ID_COLUMN]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=STEP_1_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=STEP_2_ARTIFACT_DIR)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    histories = pd.read_csv(args.audit_dir / "train_flight_cycles.csv")
    outer = balanced_group_folds(
        histories,
        n_folds=args.outer_folds,
        seed=args.seed,
        fold_column="outer_fold",
    )
    inner = make_inner_folds(
        outer,
        n_inner_folds=args.inner_folds,
        seed=args.seed + 1,
    )

    if outer[ID_COLUMN].duplicated().any():
        raise AssertionError("A UAV occurs in more than one outer fold")
    if outer.groupby("outer_fold")[ID_COLUMN].nunique().nunique() != 1:
        raise AssertionError("Outer folds do not contain equal UAV counts")
    for outer_fold, group in inner.groupby("outer_fold"):
        held_out = set(outer.loc[outer["outer_fold"] == outer_fold, ID_COLUMN])
        if held_out & set(group[ID_COLUMN]):
            raise AssertionError(f"Outer fold {outer_fold} leaks into inner folds")

    paths = [
        save_csv(outer, args.output_dir / "outer_folds.csv"),
        save_csv(inner, args.output_dir / "inner_folds.csv"),
        save_json(
            {
                "seed": args.seed,
                "outer_folds": args.outer_folds,
                "inner_folds": args.inner_folds,
                "group_key": ID_COLUMN,
                "balance_variable": "terminal_lifetime",
            },
            args.output_dir / "fold_config.json",
        ),
    ]
    print("\n".join(f"Saved {path}" for path in paths))


if __name__ == "__main__":
    main()
