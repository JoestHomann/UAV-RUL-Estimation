"""Run the saved alternative script unchanged in an isolated, traceable directory."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reproduce(output):
    import pandas as pd
    output = Path(output).resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    inputs = {"reference.py": ROOT / "other_pipelines/uav_rul_pipeline_v4 (1).py",
              "train.csv": ROOT / "data/train.csv", "test.csv": ROOT / "data/test.csv"}
    contract = {"input_sha256": {name: sha(path) for name, path in inputs.items()},
        "python": sys.version, "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "catboost")}}
    registration = output / "reproduction_contract.json"
    if registration.exists() and json.loads(registration.read_text()) != contract:
        raise ValueError("Reproduction inputs/environment changed; choose a new output directory")
    registration.write_text(json.dumps(contract, indent=2) + "\n")
    manifest_path = output / "reproduction_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if sha(output / "submission.csv") != manifest["submission_sha256"]:
            raise ValueError("Reproduced submission changed")
        print(json.dumps(manifest, indent=2), flush=True)
        return manifest
    for name, path in inputs.items():
        shutil.copy2(path, output / name)
    # Run the exact source (including its original feature order and CPU defaults).
    with (output / "reference_stdout.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", "reference.py"], cwd=output,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                log.write(line); log.flush()
                print(line, end="", flush=True)
            if process.wait():
                raise RuntimeError("Reference script failed; inspect reference_stdout.log")
        finally:
            if process.poll() is None:
                process.terminate(); process.wait()
    submission = pd.read_csv(output / "submission.csv")
    expected = set(pd.read_csv(inputs["test.csv"], usecols=["uav_id"]).uav_id)
    import numpy as np
    if list(submission) != ["id", "RUL"] or submission.id.duplicated().any() or set(submission.id) != expected or not np.isfinite(submission.RUL).all():
        raise ValueError("Reproduced submission failed schema/identity/value validation")
    manifest = {"status": "complete", "source_executed_unchanged": True, "rows": len(submission),
        "submission_sha256": sha(output / "submission.csv"), "source_sha256": contract["input_sha256"]["reference.py"],
        "reported_historical_score": 0.87877, "reproduced_kaggle_score": None,
        "score_status": "awaiting_user_upload", "submitted_to_kaggle": False,
        "local_internal_metrics_are_capped_and_not_unbiased_outer_metrics": True}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    output = args.output
    if output is None and args.config is not None:
        from advanced_r2_utils import load_workflow
        _, _, root = load_workflow(args.config, "campaign_workflows", "PE_31")
        output = root / "reproduction"
    reproduce(output or ROOT / "2_architecture_experiments/1_pipeline_experiments/experiments/PE_31/runs/run_1/reproduction")
