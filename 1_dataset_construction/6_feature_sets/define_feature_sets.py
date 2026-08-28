"""Classify generated columns into explicit, reproducible feature sets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PHASE_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from common import STEP_5_ARTIFACT_DIR, STEP_6_ARTIFACT_DIR, save_csv, save_json
from feature_recipes import feature_catalog
from phase_1_config import DEFAULT_SETTINGS_PATH, load_phase_one_profile


FEATURE_PREFIX = "feature__"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=STEP_5_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=STEP_6_ARTIFACT_DIR)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--profile", default="legacy")
    args = parser.parse_args()

    profile = load_phase_one_profile(args.profile, args.settings)

    feature_path = args.feature_dir / "training_features.csv.gz"
    columns = pd.read_csv(feature_path, nrows=0).columns.tolist()
    feature_names = [
        column for column in columns if column.startswith(FEATURE_PREFIX)
    ]
    if not feature_names:
        raise AssertionError(f"No generated feature columns found in {feature_path}")
    catalog = feature_catalog(
        feature_names,
        feature_profile=profile.feature_profile,
        feature_sets=profile.feature_sets,
    )
    path = save_csv(
        catalog,
        args.output_dir / "feature_catalog.csv",
    )
    print(f"Saved {path}")
    counts = {
        name: int(catalog[name].astype(bool).sum())
        for name in profile.feature_sets
    }
    config_path = save_json(
        {
            "profile": profile.name,
            "feature_profile": profile.feature_profile,
            "generated_features": len(catalog),
            "feature_sets": counts,
        },
        args.output_dir / "feature_set_config.json",
    )
    print(f"Saved {config_path}")


if __name__ == "__main__":
    main()
