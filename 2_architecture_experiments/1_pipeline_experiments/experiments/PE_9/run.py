"""Single execution entry point for PE_9."""

from pathlib import Path
import sys

PIPELINE_DIR = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from run_experiment_definition import main


if __name__ == "__main__":
    main(Path(__file__).with_name("settings.toml"), "PE_9")
