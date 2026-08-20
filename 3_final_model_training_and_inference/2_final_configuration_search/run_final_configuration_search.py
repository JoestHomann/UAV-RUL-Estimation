"""Run or resume Phase 3's final selected-family configuration search."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


STEP_DIR = Path(__file__).resolve().parent
PHASE_DIR = STEP_DIR.parent
for dependency_dir in (PHASE_DIR, STEP_DIR):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from final_configuration_search import (  # noqa: E402
    FinalConfigurationSearchError,
    FinalConfigurationSearchRunner,
)
from phase_3_common import require_current_settings  # noqa: E402
from phase_3_run_layout import SETTINGS_PATH  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--settings", type=Path, default=SETTINGS_PATH)
    args = parser.parse_args()
    try:
        run_number = require_current_settings(args.settings)
        runner = FinalConfigurationSearchRunner(run_number)
        selected = runner.run(force=args.force)
    except (FinalConfigurationSearchError, ValueError, OSError) as error:
        print(f"Final configuration search failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("Final configuration search complete")
    print(f"Selected: {selected['configuration_id']}")
    print(f"Mean fold RMSE: {selected['mean_fold_rmse']:.6f}")


if __name__ == "__main__":
    main()
