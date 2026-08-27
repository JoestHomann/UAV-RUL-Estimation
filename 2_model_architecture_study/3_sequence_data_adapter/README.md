# Step 3: Sequence data adapter

## Purpose

This step creates the fixed-window telemetry input boundary for TCN,
multi-scale CNN, sensor-graph TCN, LSTM, Transformer, and later sequence
architectures. It converts each training, development, locked, or test
endpoint into a causal trailing window using the 22 nonconstant channels from
the experiment contract.

Sequence tensors are created on demand and are not stored as duplicate files.
The same copied histories can therefore produce lookbacks of 50 or 100 cycles
without maintaining materialized datasets for either length.

## Inputs and copied files

The builder reads the resolved Step 1 experiment specification and copies eight
inputs into "artifacts/data/":

- raw "train.csv" and "test.csv" telemetry histories;
- training-prefix endpoints;
- development-validation endpoints;
- locked-validation endpoints;
- test endpoints;
- outer UAV folds;
- inner UAV folds.

The builder preserves modification timestamps and compares every copy with its
source using existence, size, timestamp, and complete byte-content checks. No
hashes are generated.

## Generated metadata

- "sequence_dataset_manifest.json" records the 22 ordered channels, supported
  lookbacks, padding convention, side features, scaling rule, and file roles.
- "copy_verification.json" records the copy-integrity result for all eight
  inputs.

All generated artifacts remain visible locally but are ignored by Git.

## Window construction

For an endpoint at cutoff c, the adapter:

1. locates the same UAV and cycle in the copied raw telemetry;
2. selects at most the configured final 50 or 100 cycles ending at c;
3. never includes a cycle after c;
4. left-pads short histories with zeros;
5. sets the Boolean padding mask to True only at padded positions;
6. returns flight cycle and log(1 + flight cycle) separately as side features.

The source endpoint order and telemetry-channel order are preserved.

## Fold-fitted scaling

Inner and locked-outer split methods fit one median/IQR channel scaler using
only the active training UAVs. The scale is IQR divided by 1.349, followed by a
standard-deviation fallback and then a unit fallback. The same fitted scaler is
applied to the training and validation windows, while padded values remain zero.

Complete histories of the training UAVs are used for fitting. This keeps the
normalization independent of the candidate lookback and excludes every
validation UAV. Side features are not scaled in this step.

## Build

Run from the repository root:

    py 2_model_architecture_study\3_sequence_data_adapter\build_sequence_data_adapter.py

## Adapter API

"sequence_data_adapter.py" exposes "SequenceDataAdapter", "SequenceDataset",
"SequenceSplit", and "RobustChannelScaler". The main methods are:

- "load_training(lookback)"
- "load_development(lookback)"
- "load_locked(lookback)"
- "load_test(lookback)"
- "outer_fold_labels()"
- "inner_fold_labels(outer_fold)"
- "fit_channel_scaler(training_uav_ids)"
- "get_inner_selection_split(outer_fold, inner_fold, lookback)"
- "get_locked_outer_evaluation_split(outer_fold, lookback)"

The direct loading methods return raw windows for inspection. The two split
methods return telemetry-scaled training and validation datasets together with
the training-fold channel scaler that produced them. Age side features remain
raw until the sequence model fits and persists its own training-row scaler.
