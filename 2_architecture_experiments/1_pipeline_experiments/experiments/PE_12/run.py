"""Execute PE_12 from its co-located settings file."""

from pathlib import Path
import sys

EXPERIMENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = EXPERIMENT_DIR.parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
from run_experiment_definition import main  # noqa: E402

if __name__ == "__main__":
    main(EXPERIMENT_DIR / "settings.toml", "PE_12")

