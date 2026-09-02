"""Open locked data once to confirm an already frozen PE_7 stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
STUDY_DIR = SCRIPT_DIR.parent / "2_model_architecture_study"
for path in (STUDY_DIR / "3_sequence_data_adapter", STUDY_DIR / "4_model_adapters"):
    sys.path.insert(0, str(path))
from model_registry import ModelAdapterFactory  # noqa: E402
from no_op_training_monitor import NoOpTrainingMonitor  # noqa: E402
from sequence_data_adapter import SequenceDataAdapter  # noqa: E402


class StackConfirmationError(ValueError):
    """Explain an invalid or repeated locked stack confirmation."""


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y
    return {
        "r2": 1.0 - float(np.square(error).sum()) / float(np.square(y - y.mean()).sum()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "overprediction_rate": float(np.mean(error > 0)),
        "rms_overprediction": float(np.sqrt(np.mean(np.square(np.maximum(error, 0.0))))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--sequence-manifest", type=Path, required=True)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--tree-locked-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    manifest_path = output / "locked_confirmation_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(json.dumps(existing, indent=2, sort_keys=True))
            return
        raise StackConfirmationError("An incomplete locked confirmation already exists")
    contract = json.loads(args.contract.resolve().read_text(encoding="utf-8"))
    if contract.get("family") != "heterogeneous_oof_stack":
        raise StackConfirmationError("Contract is not a heterogeneous stack")
    temporal = contract.get("components", {}).get("temporal")
    if not isinstance(temporal, dict):
        raise StackConfirmationError("Stack contract has no temporal component")
    adapter = SequenceDataAdapter(args.sequence_manifest.resolve())
    factory = ModelAdapterFactory(args.specification.resolve())
    tree = pd.read_csv(args.tree_locked_predictions.resolve())
    meta = joblib.load(REPOSITORY_ROOT / contract["frozen_meta_model"])
    records = []
    for outer_fold in adapter.outer_fold_labels():
        split = adapter.get_locked_outer_evaluation_split(
            int(outer_fold), int(temporal["lookback"])
        )
        for seed in (13, 37, 73):
            model = factory.create(
                str(temporal["family"]),
                temporal["hyperparameters"],
                seed=seed,
                training_iterations=int(temporal["training_iterations"]),
                training_monitor=NoOpTrainingMonitor(),
            )
            model.fit(split.training, split.validation)
            temporal_prediction = model.predict(split.validation)
            temporal_table = split.validation.metadata.copy().reset_index(drop=True)
            temporal_table["outer_fold"] = int(outer_fold)
            temporal_table["seed"] = seed
            temporal_table["y_true"] = split.validation.target.to_numpy(float)
            temporal_table["temporal_prediction"] = temporal_prediction
            keys = ["outer_fold", "seed", "uav_id", "scenario", "cutoff", "y_true"]
            paired = tree.loc[(tree["outer_fold"].eq(outer_fold)) & (tree["seed"].eq(seed)), keys + ["y_pred"]].merge(
                temporal_table[keys + ["temporal_prediction"]],
                on=keys,
                validate="one_to_one",
            )
            if len(paired) != len(temporal_table):
                raise StackConfirmationError(
                    f"Locked component rows do not align for fold {outer_fold}, seed {seed}"
                )
            component = paired[["y_pred", "temporal_prediction"]].to_numpy(float)
            if isinstance(meta, dict) and meta.get("method") == "convex_blend":
                stacked = meta["tree_weight"] * component[:, 0] + meta["temporal_weight"] * component[:, 1]
            elif hasattr(meta, "predict"):
                stacked = meta.predict(
                    pd.DataFrame(
                        component,
                        columns=["prediction__tree", "prediction__temporal"],
                    )
                )
            else:
                raise StackConfirmationError("Frozen OOF meta-model has an unknown format")
            paired["stack_prediction"] = np.maximum(np.asarray(stacked, float), 0.0)
            records.append(paired)
    predictions = pd.concat(records, ignore_index=True)
    seed_metrics = []
    for seed, rows in predictions.groupby("seed", sort=True):
        seed_metrics.append({"seed": int(seed), **_metrics(rows["y_true"].to_numpy(float), rows["stack_prediction"].to_numpy(float))})
    metrics = pd.DataFrame(seed_metrics)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "locked_stack_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(output / "locked_stack_metrics_by_seed.csv", index=False)
    mean_metrics = {column: float(metrics[column].mean()) for column in metrics if column != "seed"}
    manifest = {
        "status": "complete",
        "family": "heterogeneous_oof_stack",
        "development_gate_passed": True,
        "uses_locked_evaluation": True,
        "locked_results_used_for_tuning": False,
        "seed_count": len(metrics),
        "mean_locked_metrics": mean_metrics,
        "contract": args.contract.resolve().relative_to(REPOSITORY_ROOT).as_posix(),
        "artifacts": {
            "predictions": "locked_stack_predictions.csv.gz",
            "metrics_by_seed": "locked_stack_metrics_by_seed.csv",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
