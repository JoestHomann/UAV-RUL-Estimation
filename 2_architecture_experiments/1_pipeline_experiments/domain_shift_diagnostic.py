"""Measure development/test separability and stable feature shift for PE_9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from experiment_paths import repository_path, run_directory
from experiment_config import read_experiment_config


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
STUDY_DIR = SCRIPT_DIR.parent / "2_model_architecture_study"
sys.path.insert(0, str(STUDY_DIR / "2_tabular_data_adapter"))
from tabular_data_adapter import TabularDataAdapter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default="PE_9")
    args = parser.parse_args()
    config = read_experiment_config(args.config.resolve())
    workflow = config["domain_workflows"][args.workflow]
    definition = config["run_definitions"][args.workflow]
    adapter = TabularDataAdapter(repository_path(REPOSITORY_ROOT, workflow["tabular_manifest"]))
    feature_set = str(workflow.get("feature_set", "screened_drift_pruned"))
    development = adapter.load_development(feature_set)
    test = adapter.load_test(feature_set)
    features = pd.concat([development.features, test.features], ignore_index=True)
    labels = np.r_[np.zeros(len(development), dtype=int), np.ones(len(test), dtype=int)]
    groups = pd.concat(
        [
            "development::" + development.metadata["uav_id"].astype(str),
            "test::" + test.metadata["uav_id"].astype(str),
        ],
        ignore_index=True,
    )
    combined_metadata = pd.concat(
        [development.metadata.reset_index(drop=True), test.metadata.reset_index(drop=True)],
        ignore_index=True,
    )
    cutoff = pd.concat([development.metadata["cutoff"], test.metadata["cutoff"]], ignore_index=True).to_numpy(float).reshape(-1, 1)
    propensity_records = []
    auc_records = []
    for seed in workflow.get("seeds", [13, 37, 73]):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(seed))
        for fold, (training, validation) in enumerate(splitter.split(features, labels, groups)):
            for model_name, matrix in (("all_features", features.to_numpy(float)), ("cutoff_only", cutoff)):
                model = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt", random_state=int(seed), n_jobs=1)
                model.fit(matrix[training], labels[training])
                prediction = model.predict_proba(matrix[validation])[:, 1]
                auc_records.append({"seed": int(seed), "fold": fold, "model": model_name, "auc": float(roc_auc_score(labels[validation], prediction))})
                if model_name == "all_features":
                    for index, value in zip(validation, prediction, strict=True):
                        metadata = combined_metadata.iloc[index]
                        propensity_records.append({"seed": int(seed), "row_index": int(index), "domain": "test" if labels[index] else "development", "uav_id": metadata["uav_id"], "scenario": metadata.get("scenario", "test"), "cutoff": metadata["cutoff"], "propensity": float(value)})
    shifts = []
    for name in features.columns:
        left = development.features[name].to_numpy(float)
        right = test.features[name].to_numpy(float)
        pooled = float(np.sqrt((np.var(left) + np.var(right)) / 2.0))
        shifts.append({"feature": name, "ks_statistic": float(ks_2samp(left, right).statistic), "standardized_mean_difference": 0.0 if pooled == 0 else float((np.mean(right) - np.mean(left)) / pooled)})
    shifts = pd.DataFrame(shifts)
    shifts["shift_score"] = shifts["ks_statistic"] + 0.25 * shifts["standardized_mean_difference"].abs()
    shifts = shifts.sort_values(["shift_score", "feature"], ascending=[False, True])
    root = run_directory(EXPERIMENTS_DIR, args.workflow, definition)
    diagnostic = root / "domain_diagnostic"
    diagnostic.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(auc_records).to_csv(diagnostic / "domain_auc.csv", index=False)
    propensity = pd.DataFrame(propensity_records).groupby(["row_index", "domain", "uav_id", "scenario", "cutoff"], as_index=False, dropna=False)["propensity"].mean()
    propensity.to_csv(diagnostic / "domain_propensity.csv", index=False)
    shifts.to_csv(diagnostic / "feature_shift_statistics.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    auc = pd.DataFrame(auc_records)
    auc.boxplot(column="auc", by="model", ax=axis)
    axis.set_title("Repeated OOF development/test separability")
    axis.set_ylabel("ROC AUC")
    figure.suptitle("")
    figure.tight_layout()
    figure.savefig(diagnostic / "domain_auc.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    manifest = {"status": "complete", "mean_all_feature_auc": float(auc.loc[auc["model"].eq("all_features"), "auc"].mean()), "mean_cutoff_only_auc": float(auc.loc[auc["model"].eq("cutoff_only"), "auc"].mean()), "uses_test_labels": False, "seeds": workflow.get("seeds", [13, 37, 73])}
    (diagnostic / "domain_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
