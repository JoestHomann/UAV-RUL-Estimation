"""Build the model registry from the resolved contract and adapter classes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


STEP_DIR = Path(__file__).resolve().parent
DEFAULT_SPECIFICATION_PATH = (
    STEP_DIR.parent
    / "1_architecture_study_settings"
    / "artifacts"
    / "experiment_specification.json"
)
DEFAULT_OUTPUT_DIR = STEP_DIR / "artifacts"
OUTPUT_FILENAME = "model_registry.json"

if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from base import ModelAdapterError  # noqa: E402
from model_registry import build_registry_payload  # noqa: E402


def build_model_registry(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Generate one deterministic registry after importing every adapter."""

    registry = build_registry_payload(specification_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    output_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    """Build the Step 4 registry from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specification",
        type=Path,
        default=DEFAULT_SPECIFICATION_PATH,
        help="Location of Step 1's generated experiment specification.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated model registry.",
    )
    args = parser.parse_args()

    try:
        output_path = build_model_registry(args.specification, args.output_dir)
    except ModelAdapterError as error:
        print(f"Model registry build failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    registry = json.loads(output_path.read_text(encoding="utf-8"))
    enabled = sum(entry["enabled"] for entry in registry["families"].values())
    print("Model registry built successfully")
    print(f"Implemented families: {len(registry['families'])}")
    print(f"Enabled families: {enabled}")
    print(f"Saved {output_path.resolve()}")


if __name__ == "__main__":
    main()
