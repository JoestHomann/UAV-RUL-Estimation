"""Build the deterministic JSON artifact for the Phase 2 experiment contract.

The TOML file is the human-edited source of truth.  This script converts it to
a machine-friendly JSON document only after the contract schema and every
declared Phase 1 dependency pass the shared verifier.  It contains no model
training logic and makes no architecture-selection decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


STEP_DIR = Path(__file__).resolve().parent

# Running this file directly makes Python place only this directory on the
# import path.  The explicit guard documents and preserves that expectation,
# while avoiding duplicate entries when the module is imported elsewhere.
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

# Import the shared verification gate instead of implementing a second, subtly
# different validation path in the builder.  The import follows the path setup,
# hence the local E402 exception.
from verify_experiment_contract import (  # noqa: E402
    ContractError,
    DEFAULT_CONTRACT_PATH,
    load_and_verify_contract,
    repository_relative_path,
)


DEFAULT_OUTPUT_DIR = STEP_DIR / "artifacts"
OUTPUT_FILENAME = "experiment_specification.json"


def resolved_specification(contract_path: Path) -> dict[str, Any]:
    """Create the complete in-memory payload after mandatory verification.

    The result contains three deliberately separate parts:

    * "contract_source" identifies the human-readable source using a portable
      repository-relative path.
    * "contract" is the fully validated configuration converted to ordinary
      JSON-compatible values.
    * "phase_1_verification" records which upstream artifacts were observed
      and confirms that the Phase 1 leakage assertions passed.

    No timestamp, absolute path, hash, or randomly generated identifier is
    added.  The same inputs therefore produce byte-for-byte identical JSON.
    """

    # This call is the single shared gate: schema errors or incompatible Phase 1
    # artifacts stop the build before any output file can be changed.
    contract, verification = load_and_verify_contract(contract_path)
    return {
        "contract_source": repository_relative_path(contract_path),
        "contract": contract.model_dump(mode="json"),
        "phase_1_verification": verification.model_dump(mode="json"),
    }


def write_specification(payload: dict[str, Any], output_dir: Path) -> Path:
    """Serialize a validated payload in a deterministic, readable format.

    "sort_keys=True" removes dictionary insertion order as a source of output
    changes.  A fixed indentation and final newline make diffs readable and
    ensure repeated builds are stable.  Directory creation occurs only here,
    after validation has succeeded.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    """Validate the source contract and build its resolved JSON artifact."""

    # Only input and output locations are configurable from the command line.
    # Experiment values remain exclusively in TOML so a produced result always
    # has one inspectable scientific configuration.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="TOML contract location; experiment values cannot be overridden.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated JSON specification.",
    )
    args = parser.parse_args()

    try:
        payload = resolved_specification(args.contract)
    except ContractError as error:
        print(f"Experiment contract build failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    # Writing is deliberately separated from resolution.  A failed contract or
    # failed Phase 1 check cannot leave behind a newly generated specification.
    output_path = write_specification(payload, args.output_dir)
    verification = payload["phase_1_verification"]
    print(
        "Experiment contract built successfully\n"
        f"Verified Phase 1 artifacts: {verification['checked_artifacts']}\n"
        f"Saved {output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
