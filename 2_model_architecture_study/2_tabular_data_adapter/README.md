# Step 2: Tabular data adapter

## Purpose

This step creates the tabular input boundary used by every feature-based model
in Phase 2. It copies the required Phase 1 feature and fold files into its own
artifact folder, verifies that the copies remain identical to their sources,
and provides one shared Python adapter for selecting features and UAV folds.

The adapter does not scale, impute, transform, or reorder the data. Scaling is
fitted later using only the training UAVs of the active inner or outer fold.

## Inputs

The build reads the resolved Step 1 specification:

"../1_experiment_contract/artifacts/experiment_specification.json"

It copies these seven Phase 1 artifacts:

- "training_features.csv.gz"
- "development_validation_features.csv.gz"
- "locked_validation_features.csv.gz"
- "test_features.csv.gz"
- "feature_catalog.csv"
- "outer_folds.csv"
- "inner_folds.csv"

## Outputs

Generated files are written below "artifacts/":

- "data/" contains the seven timestamp-preserving copies.
- "tabular_dataset_manifest.json" defines dataset roles, metadata columns,
  label and weight availability, file provenance, and feature-set counts.
- "copy_verification.json" records existence, size, timestamp, and direct
  byte-comparison results for every source/copy pair.

These outputs are reproducible and ignored by Git, but remain available locally
so the exact inputs consumed by Step 2 can be inspected.

## Build and verify

Run from the repository root:

    py 2_model_architecture_study\2_tabular_data_adapter\build_tabular_data_adapter.py

The build automatically refreshes all copies, writes the manifest, and runs the
copy checker. The checker can also be run independently:

    py 2_model_architecture_study\2_tabular_data_adapter\verify_copied_files.py

No hashes are created. The checker compares file existence, size, modification
time, and complete byte content.

## Adapter API

"tabular_data_adapter.py" exposes "TabularDataAdapter", "TabularDataset", and
"TabularSplit". Its public methods are:

- "load_training(feature_set)"
- "load_development(feature_set)"
- "load_locked(feature_set)"
- "load_test(feature_set)"
- "outer_fold_labels()"
- "inner_fold_labels(outer_fold)"
- "get_inner_selection_split(outer_fold, inner_fold, feature_set)"
- "get_locked_outer_evaluation_split(outer_fold, feature_set)"

Every returned dataset keeps the feature matrix, metadata, optional RUL target,
and optional sample weights separate. Requested feature columns follow the
original order in the copied feature catalog.
