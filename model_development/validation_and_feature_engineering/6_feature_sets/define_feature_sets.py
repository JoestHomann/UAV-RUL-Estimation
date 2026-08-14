"""Classify generated columns into explicit, reproducible feature sets."""

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
    CONTEXT_CHANNELS,
    DEGRADATION_CHANNELS,
    STEP_5_ARTIFACT_DIR,
    STEP_6_ARTIFACT_DIR,
    channel_role,
    save_csv,
)


FEATURE_PREFIX = "feature__"
CONTEXT_STATISTICS = {
    "last",
    "baseline_mean",
    "baseline_delta",
    "history_mean",
    "history_sd",
    "history_min",
    "history_max",
}


def feature_catalog(feature_names: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for feature_name in feature_names:
        body = feature_name.removeprefix(FEATURE_PREFIX)
        if body in {"flight_cycle", "log1p_flight_cycle"}:
            records.append(
                {
                    "feature_name": feature_name,
                    "channel": "",
                    "channel_role": "age",
                    "statistic": body,
                    "window": "",
                    "age_only": True,
                    "last_values": True,
                    "screened": True,
                    "all_nonconstant": True,
                }
            )
            continue

        channel, statistic = body.split("__", maxsplit=1)
        window = ""
        base_statistic = statistic
        if statistic.startswith("w") and "_" in statistic:
            window, base_statistic = statistic.split("_", maxsplit=1)
        records.append(
            {
                "feature_name": feature_name,
                "channel": channel,
                "channel_role": channel_role(channel),
                "statistic": base_statistic,
                "window": window,
                "age_only": False,
                "last_values": statistic == "last",
                "screened": channel in DEGRADATION_CHANNELS
                or (
                    channel in CONTEXT_CHANNELS
                    and statistic in CONTEXT_STATISTICS
                ),
                "all_nonconstant": True,
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=STEP_5_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=STEP_6_ARTIFACT_DIR)
    args = parser.parse_args()

    feature_path = args.feature_dir / "training_features.csv.gz"
    columns = pd.read_csv(feature_path, nrows=0).columns.tolist()
    feature_names = [
        column for column in columns if column.startswith(FEATURE_PREFIX)
    ]
    if not feature_names:
        raise AssertionError(f"No generated feature columns found in {feature_path}")
    path = save_csv(
        feature_catalog(feature_names),
        args.output_dir / "feature_catalog.csv",
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
