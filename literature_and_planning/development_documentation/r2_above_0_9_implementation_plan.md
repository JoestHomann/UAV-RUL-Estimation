# Implementation Plan for Reaching R2 Above 0.9

Last updated: 2026-09-01

## Implementation Status

Implemented on 2026-09-01:

- per-experiment sequence, neural-training, and fixed-hyperparameter overrides;
- dense sequence verification at lookbacks 20, 30, and 50;
- causal GRU adapter and deterministic persistence verification;
- PE_6 two-stage density/lookback workflow and automatic winner propagation;
- development-only temporal architecture Run 7 with three-seed confirmation;
- PE_7 exact OOF alignment, nested stacking, leakage provenance, and promotion
  contract generation;
- the heterogeneous dataset and OOF-frozen production stack adapter core;
- PE_8 fold-fitted personalized-onset targets and banded safety gates;
- PE_9 repeated domain diagnostics, deterministic shift pruning, fold-local
  target-aware pruning, and high-propensity gates;
- backward-compatible Phase 3 named q=0.50/q=0.55 submission policies;
- synthetic contract verification for alignment, duplicate rejection, nested
  UAV isolation, onset targets, and dual-policy settings.

Full scientific runs remain intentionally unexecuted. Locked evaluation and
Phase 3 promotion remain closed until their declared development gates pass.

## Purpose

This document translates the recommendations in
[`r2_above_0_9_strategy.md`](r2_above_0_9_strategy.md) into repository changes,
execution stages, experiment contracts, and promotion gates.

The implementation must preserve the existing separation between:

- pipeline experiments under
  `2_architecture_experiments/1_pipeline_experiments/experiments/PE_X`;
- the standalone model architecture study under
  `2_architecture_experiments/2_model_architecture_study`;
- final model training and inference under
  `3_final_model_training_and_inference`.

Each new `PE_X` must have one user entry point, one user-editable TOML file, a
README, and run-owned artifacts. Shared implementation code remains outside the
individual experiment folders.

## Scientific Invariants

The following settings remain fixed unless a named experiment explicitly tests
them:

| Setting | Fixed value |
| --- | --- |
| Validation | Existing UAV-grouped outer and inner folds |
| Scenario profile | `early_and_middle` |
| Fitting target | Piecewise cap at 125 |
| Evaluation target | Raw RUL |
| Training-unit weighting | Equal total weight per UAV |
| Search seed | 13 |
| Confirmation seeds | 13, 37, 73 |
| Accuracy discovery calibration | None or `q = 0.50` |
| Locked data | Closed until the final promotion gate passes |
| External NASA test RUL | Never read or used |

All comparisons use development predictions first. A setting may enter locked
evaluation only through the explicit final gate in this plan.

## Experiment Sequence

| Order | Identifier | Type | Dependency | Opens locked data |
| ---: | --- | --- | --- | --- |
| 1 | `PE_6` | Pipeline experiment | None | No |
| 2 | Architecture `run_7` | Model architecture study | `PE_6` winner | No |
| 3 | `PE_7` | Pipeline experiment | Architecture `run_7` winner | No |
| 4 | `PE_8` | Pipeline experiment | None; may run alongside `PE_6` | No |
| 5 | `PE_9` | Pipeline experiment | None; may run alongside `PE_6` | No |
| 6 | Final promotion | Gate | Best development candidate | Once |
| 7 | Phase 3 policy variants | Final pipeline | Locked-confirmed winner | No additional tuning |

`PE_8` and `PE_9` are independent supporting experiments. Their winners may be
incorporated into `PE_7` only if they pass their own development gates.

## Shared Enabling Work

### 1. Per-experiment sequence settings

The pipeline experiment manager currently overrides architectures, feature
sets, target profiles, and candidate budgets, but sequence lookbacks remain in
the shared Phase 2 base TOML. Extend the experiment materialization path in:

- `2_architecture_experiments/1_pipeline_experiments/run_experiments.py`;
- `2_architecture_experiments/1_pipeline_experiments/experiment_config.py`;
- `2_architecture_experiments/2_model_architecture_study/1_architecture_study_settings/verify_architecture_study_settings.py`.

Add validated experiment fields:

```toml
sequence_lookbacks = [30]
sequence_channels = ["telemetry_01", "..."]

[neural_training]
batch_size = 128
maximum_epochs = 250
early_stopping_patience = 20
gradient_clip_global_norm = 1.0
```

The resolved Phase 2 JSON must contain these values. The standalone
`architecture_study_settings.toml` remains independent and must not be edited by
pipeline experiments.

### 2. Dense prefix definitions

Phase 1 already supports `dense_all` and `dense_stride`; no new prefix creation
algorithm is required. Define experiment-owned variants in `PE_6/settings.toml`:

```toml
[prefix_variants.dense_stride_1]
strategy = "dense_all"
minimum_cutoff = 1
seed = 20260814

[prefix_variants.dense_stride_2]
strategy = "dense_stride"
stride = 2
minimum_cutoff = 1
seed = 20260814
```

Verify that `create_training_prefixes.py` gives every UAV the same total sample
weight under both variants. The generator already asserts that the per-UAV
weight sums equal one; retain that assertion and expose its result in the
verification artifact.

### 3. Sequence adapter verification

The sequence adapter already provides causal windows, left padding, masks, and
fold-fitted telemetry scaling. Extend its verification to cover:

- lookbacks 20, 30, and 50;
- histories shorter than the selected lookback;
- dense endpoint row counts;
- zero values in padded positions after scaling;
- disjoint training and validation UAV identifiers;
- equal per-UAV training weight totals.

Edit:

- `2_architecture_experiments/2_model_architecture_study/3_sequence_data_adapter/build_sequence_data_adapter.py`;
- `2_architecture_experiments/2_model_architecture_study/3_sequence_data_adapter/sequence_data_adapter.py`;
- create `2_architecture_experiments/2_model_architecture_study/3_sequence_data_adapter/verify_dense_sequence_data_adapter.py`.

### 4. GRU adapter

CNN, TCN, and LSTM adapters already exist. Add the missing GRU family:

- create `2_architecture_experiments/2_model_architecture_study/4_model_adapters/models/neural/gru.py`;
- register `GRUAdapter` in `model_registry.py`;
- add exact expected hyperparameters in `EXPECTED_HYPERPARAMETERS`;
- include GRU in the neural-family factory branch;
- add its architecture table to both standalone and pipeline-owned Phase 2 base
  settings;
- extend settings verification and registry verification.

The GRU must be causal and unidirectional. It should use packed valid sequence
lengths like the LSTM adapter and reuse `SequenceNeuralAdapter` for all scaling
and input checks.

Initial search space:

```toml
[architectures.gru]
status = "included"
representation = "sequence"
lookbacks = [20, 30, 50]
variants = ["unidirectional"]

[architectures.gru.search.layers]
kind = "categorical"
values = [1, 2]

[architectures.gru.search.hidden_units]
kind = "categorical"
values = [32, 64, 128]

[architectures.gru.search.dropout]
kind = "categorical"
values = [0.0, 0.1, 0.2]

[architectures.gru.search.learning_rate]
kind = "log_uniform"
low = 1e-4
high = 3e-3

[architectures.gru.search.weight_decay]
kind = "log_uniform"
low = 1e-6
high = 1e-2
```

Add `verify_gru.py` with fit, predict, persistence, padding, and deterministic
same-seed checks.

## PE_6: Dense Temporal Sample Construction

### Question

Does dense sequence supervision expose degradation information that the
existing 20-prefix training set misses, and which causal lookback is best?

### Planned directory

```text
2_architecture_experiments/1_pipeline_experiments/experiments/PE_6/
|-- README.md
|-- run.py
|-- settings.toml
|-- figures/
`-- runs/
    `-- run_1/
        |-- sampling_comparison/
        |-- lookback_comparison/
        `-- reporting/
```

### Entry point and settings

`run.py` must delegate to `run_experiment_definition.main`, matching PE_1 to
PE_5. `settings.toml` is the only user-edited file and declares both stages.

Stage A isolates endpoint density with a fixed lookback and fixed anchor model:

| Cell | Prefix variant | Lookback | Model |
| --- | --- | ---: | --- |
| `PE6_sparse_l30` | `current20` | 30 | Fixed multiscale CNN |
| `PE6_dense2_l30` | `dense_stride_2` | 30 | Same fixed multiscale CNN |
| `PE6_dense1_l30` | `dense_stride_1` | 30 | Same fixed multiscale CNN |

Stage B isolates lookback using the Stage A density winner:

| Cell | Prefix variant | Lookback | Model |
| --- | --- | ---: | --- |
| `PE6_w20` | Propagated winner | 20 | Same fixed multiscale CNN |
| `PE6_w30` | Propagated winner | 30 | Same fixed multiscale CNN |
| `PE6_w50` | Propagated winner | 50 | Same fixed multiscale CNN |

All anchor hyperparameters must be fixed in the resolved architecture table.
Do not let separate Optuna searches confound the sampling comparison. The
automatic workflow propagates the density winner into Stage B using only mean
development RMSE. For an exact tie, prefer the least expensive source in this
order: `current20`, stride 2, then stride 1.

Use this predeclared anchor configuration in all six cells:

```toml
branch_channels = 32
kernel_sizes = [3, 7, 15]
dropout = 0.20
learning_rate = 0.001
weight_decay = 0.0001
```

### New shared reporter

Create:

- `2_architecture_experiments/1_pipeline_experiments/report_temporal_sampling.py`.

It must produce:

- `sampling_fold_results.csv`;
- `sampling_summary.csv`;
- `lookback_fold_results.csv`;
- `lookback_summary.csv`;
- `temporal_sampling_comparison.png`;
- `temporal_lookback_comparison.png`;
- `winner_manifest.json`.

Copy the figures into `PE_6/figures` through the existing figure collector.

### PE_6 gate

Proceed to architecture `run_7` only when the dense winner:

- beats `current20` in at least four of five outer folds;
- improves mean RMSE by at least 3%;
- has no fold more than 1.0 RMSE worse than the sparse control; and
- is stable enough to complete without non-finite losses or gradient failures.

PE_6 never runs Step 6 locked evaluation.

## Architecture Study Run 7: Temporal Families

### Question

Which compact temporal architecture best uses the frozen dense sampling and
lookback policy selected by `PE_6`?

### Configuration

Edit the standalone architecture study TOML only after `PE_6` completes:

```text
2_architecture_experiments/2_model_architecture_study/
|-- run_phase_2.py
|-- 1_architecture_study_settings/
|   `-- architecture_study_settings.toml
`-- runs/
    `-- run_7/
```

Set:

- `run_number = 7`;
- enabled families: `tcn`, `multiscale_cnn`, `gru`, and `lstm`;
- `sequence_lookbacks` to the single PE_6 winner;
- the Phase 1 artifacts to the PE_6 winning dense-prefix run;
- target mode to piecewise cap 125;
- calibration to none;
- candidate budget to 8-12 per family;
- one search seed and three confirmation seeds.

Do not enable transformer, graph TCN, CatBoost, DTW, or unrelated tabular
families in this run. Load the frozen Run 6 tree-blend development predictions
as a reporting reference rather than rerunning its component searches.

### Architecture outputs

Extend Step 7 reporting with:

- mean and per-fold metrics for each temporal family;
- seed stability;
- training time and epoch counts;
- residual correlations against the frozen tree blend;
- metrics by RUL band and cutoff band;
- prediction-versus-target and residual-alignment figures.

### Architecture gate

A temporal model may enter `PE_7` only if it satisfies all of:

- mean grouped `R2 >= 0.89`;
- mean grouped `RMSE <= 10.7`;
- no severe fold failure;
- stable confirmation across seeds 13, 37, and 73;
- `abs(corr(tree residual, temporal residual)) < 0.90`.

Do not run architecture Step 6. This run is development-only because the
downstream stack, not an individual temporal family, is the promotion target.

## PE_7: Leakage-Free OOF Stacking

### Question

Can a complementary temporal model correct tree-blend errors sufficiently to
reach the development region required for public `R2 > 0.9`?

### Planned directory

```text
2_architecture_experiments/1_pipeline_experiments/experiments/PE_7/
|-- README.md
|-- run.py
|-- settings.toml
|-- figures/
`-- runs/run_1/
    |-- aligned_oof_predictions/
    |-- stacking/
    `-- reporting/
```

### New shared implementation

Create:

- `2_architecture_experiments/1_pipeline_experiments/align_oof_predictions.py`;
- `2_architecture_experiments/1_pipeline_experiments/stack_oof_predictions.py`;
- `2_architecture_experiments/1_pipeline_experiments/promote_stacked_model.py`.

The alignment script must pair sources on fold, UAV, scenario, cutoff, observed
RUL, and validation-row identity. It must reject duplicate, missing, or
target-disagreeing rows.

Compare these declared methods:

| Method | Implementation |
| --- | --- |
| `tree_control` | Frozen Run 6 tree blend OOF predictions |
| `temporal_control` | Architecture run_7 winner |
| `blend_025` | 25% temporal, 75% tree |
| `blend_050` | 50% temporal, 50% tree |
| `blend_075` | 75% temporal, 25% tree |
| `nonnegative_ridge` | Fold-fitted nonnegative linear stack |
| `shallow_xgboost` | Fixed small XGBoost meta-model |

Meta-model predictions must be nested inside the existing inner folds. No meta
prediction may be produced by a model fitted on that endpoint or UAV. The
promotion script must verify this from saved provenance columns.

### Production stack adapter

An OOF stack is not deployable through the current single-representation model
contract. Add a promoted heterogeneous family that consumes aligned tabular and
sequence datasets:

- create `2_architecture_experiments/2_model_architecture_study/4_model_adapters/models/ensemble/__init__.py`;
- create `2_architecture_experiments/2_model_architecture_study/4_model_adapters/models/ensemble/heterogeneous_oof_stack.py`;
- register `heterogeneous_oof_stack` as an optional promoted family in
  `model_registry.py`;
- create an aligned `HeterogeneousDataset` container in
  `3_final_model_training_and_inference/phase_3_data.py`;
- extend the Phase 3 split repository to return aligned tabular and sequence
  splits when `representation = "heterogeneous"`;
- extend Phase 3 selection verification and contract construction with a
  promoted-stack branch analogous to `calibrated_tree_blend`.

The promotion contract must freeze:

- tree component configuration and feature set;
- temporal family, configuration, lookback, and fixed training epochs;
- meta-model type and fitted OOF meta-model artifact;
- required tabular and sequence manifests;
- row-alignment keys;
- target and prediction policies.

At final fit, the adapter trains both base components on their aligned full
training data and applies the already frozen OOF meta-model to their test
predictions. It must never refit the meta-model on in-sample predictions from
the full-data base models. Adapter persistence must include both fitted base
models, the frozen meta-model, input schemas, and alignment metadata.

Phase 3 Step 2 uses one frozen stack candidate rather than reopening the stack
search. It may estimate the temporal component's fixed epoch count from the
development folds, but it may not change the selected base configurations or
meta-model.

### PE_7 gate

Promote a stack only if it reaches:

- mean grouped `R2 >= 0.91`;
- mean grouped `RMSE <= 9.5`;
- at least four of five fold wins over the tree control;
- at least 3% mean RMSE improvement;
- stable performance across three base-model seeds.

If no method passes, stop before locked evaluation and retain Run 6.

## PE_8: Personalized Degradation Onset

### Question

Does a fold-fitted, UAV-specific degradation onset improve predictions in the
RUL 51-125 region relative to a universal cap of 125?

### Planned directory and scripts

```text
2_architecture_experiments/1_pipeline_experiments/experiments/PE_8/
|-- README.md
|-- run.py
|-- settings.toml
|-- figures/
`-- runs/run_1/
```

Create shared modules:

- `2_architecture_experiments/1_pipeline_experiments/degradation_onset.py`;
- `2_architecture_experiments/1_pipeline_experiments/build_onset_targets.py`;
- `2_architecture_experiments/1_pipeline_experiments/report_onset_experiment.py`.

Implement two fold-fitted detectors:

- temporal-correlation change point;
- monotonic health-index change point.

Each detector is fitted only from the current fold's training UAVs. It may use
complete run-to-failure histories for those training UAVs, but never a held-out
UAV's terminal lifetime or future telemetry. It writes a per-training-row
`fitting_target` column while preserving raw `RUL` for all metrics.

Extend tabular and sequence dataset objects with an optional precomputed
`fitting_target`. Update `ModelAdapter.fitting_target_values` to prefer that
column when present and otherwise apply the existing `TargetPolicy`. Record the
target-builder manifest in every selected configuration.

Cells:

| Cell | Fitting target |
| --- | --- |
| `PE8_cap125` | Existing global cap |
| `PE8_temporal_change_point` | Per-UAV temporal-correlation onset |
| `PE8_health_index_change_point` | Per-UAV monotonic health-index onset |

Use the current XGBoost-ExtraTrees components with fixed configurations so only
the fitting target changes.

### PE_8 gate

Promote an onset target only when it:

- wins at least four folds overall;
- improves RMSE in both RUL 51-75 and 76-125;
- does not worsen RUL 0-50 RMSE by more than 0.25;
- does not increase RMS overprediction near failure; and
- produces plausible onset distributions without fold-specific collapse.

## PE_9: Domain Robustness

### Question

Can features that distinguish development from test endpoints be removed or
controlled without discarding RUL signal?

### Planned directory and scripts

```text
2_architecture_experiments/1_pipeline_experiments/experiments/PE_9/
|-- README.md
|-- run.py
|-- settings.toml
|-- figures/
`-- runs/run_1/
```

Create:

- `2_architecture_experiments/1_pipeline_experiments/domain_shift_diagnostic.py`;
- `2_architecture_experiments/1_pipeline_experiments/build_domain_feature_sets.py`;
- `2_architecture_experiments/1_pipeline_experiments/report_domain_robustness.py`.

The diagnostic must save repeated OOF domain AUC, feature shift statistics,
domain propensity, overlap plots, and model error by propensity band. It must
include the cutoff-only classifier as a negative control.

Initial cells:

| Cell | Change |
| --- | --- |
| `PE9_control` | Current `screened_drift_pruned` |
| `PE9_shift_pruned_5` | Remove five strongest stable shift features |
| `PE9_shift_pruned_10` | Remove ten strongest stable shift features |
| `PE9_target_aware_pruned` | Remove shifted features only when target importance is low |

Do not add a rank/quantile-transform cell for the tree control: monotonic
transforms generally preserve tree split ordering and would not test a useful
mechanism. Keep density-ratio weighting disabled by default because current
domain propensity is negatively correlated with absolute development error.

An optional clipped-weight cell may be enabled only when:

- competition rules permit test-distribution adaptation;
- repeated density-ratio estimates show adequate overlap;
- effective sample size remains at least 50% of the unweighted set; and
- propensity becomes positively associated with model error in the relevant
  development comparison.

### PE_9 gate

Promote a feature set only if it wins at least four folds, improves mean RMSE by
at least 2%, and does not worsen the highest-propensity quintile.

## Final Promotion and Locked Evaluation

After `PE_7`, `PE_8`, and `PE_9` finish, create one development comparison table
containing:

- Run 6 tree control;
- PE_7 stack winner;
- PE_8 target winner, if any;
- PE_9 feature winner, if any;
- combinations only when their effects are orthogonal and revalidated together.

The final candidate must reach approximately `R2 >= 0.91` and `RMSE <= 9.5`,
win at least four folds, and pass the safety regressions. Freeze its model,
features, target, stack, seeds, and calibration mode before running locked Step
6 once. Do not use locked results to reopen the development search.

## Phase 3 Accuracy and Safety Policies

Phase 3 currently resolves one conditional calibrator. Extend it to produce two
verified policies from the same frozen base model without retraining:

```toml
[[submission_policies]]
name = "accuracy_q50"
calibration = "conditional_quantile"
non_overprediction_coverage = 0.50

[[submission_policies]]
name = "conservative_q55"
calibration = "conditional_quantile"
non_overprediction_coverage = 0.55
```

Edit:

- `3_final_model_training_and_inference/2_final_configuration_search/final_configuration_search.py`;
- `3_final_model_training_and_inference/3_final_training_contract/build_final_training_contract.py`;
- `3_final_model_training_and_inference/5_test_inference/run_test_inference.py`;
- `3_final_model_training_and_inference/6_submission_verification/build_submission.py`;
- `3_final_model_training_and_inference/7_post_run_reporting/build_phase_3_report.py`;
- `3_final_model_training_and_inference/1_winning_architecture_selection/verify_phase_3_settings.py`.

Step 2 cross-fits and reports both policies. Step 3 freezes both calibrators.
Step 5 writes `submission_accuracy_q50.csv` and
`submission_conservative_q55.csv`. Step 6 regenerates and verifies each file.
Step 7 plots their accuracy-safety tradeoff. One policy remains marked as the
canonical `submission.csv` through an explicit TOML setting.

Do not select the canonical policy from repeated Kaggle feedback.

## Testing Plan

### Unit and verification tests

Add or extend checks for:

- dense prefix row counts and equal UAV weight totals;
- sequence padding and fold-fitted scaling at lookbacks 20, 30, and 50;
- GRU fit, predict, persistence, and deterministic replay;
- exact OOF key alignment and duplicate rejection;
- nested meta-model leakage prevention;
- fold-fitted onset detector isolation;
- preservation of raw evaluation targets;
- deterministic domain-feature selection;
- multiple Phase 3 submission-policy verification.

### Synthetic integration run

Before using the complete data, run a small temporary configuration with:

- two outer folds;
- two inner folds;
- two temporal candidates;
- one retraining seed;
- a small UAV subset;
- all locked stages disabled.

The integration run must exercise PE_6 winner propagation, architecture data
loading, OOF alignment, onset-target construction, domain reporting, and both
Phase 3 policy files.

### Full-run verification

For every full experiment:

- `run.py --list` prints the exact script chain;
- `run.py --status` reports completed cells and gates;
- interrupted cells resume from their own run folder;
- figures are collected in the experiment-level `figures` folder;
- manifests contain repository-relative paths;
- `git diff --check` and the relevant verification scripts pass.

## Delivery Milestones

| Milestone | Deliverable | Stop condition |
| ---: | --- | --- |
| 1 | Sequence overrides, dense adapter verification, GRU | Synthetic sequence tests pass |
| 2 | `PE_6` implementation | Sampling and lookback winner manifest generated |
| 3 | Architecture run_7 configuration/reporting | Temporal gate evaluated |
| 4 | `PE_7` generic OOF stack | Leakage checks and stack report pass |
| 5 | `PE_8` onset targets | Fold-isolation tests pass |
| 6 | `PE_9` domain robustness | Domain report and feature manifests pass |
| 7 | Final promotion | One frozen development winner or explicit no-promotion result |
| 8 | Phase 3 policy variants | Two verified submissions from one frozen model |

Implement and execute milestones sequentially through Milestone 4. Milestones
5 and 6 may be implemented in parallel after the shared enabling work, but
their results must not alter an already opened locked evaluation.

## Definition of Done

The program is complete when either:

1. one frozen candidate passes the final development gate, completes exactly
   one locked evaluation, and produces verified `q = 0.50` and `q = 0.55`
   Phase 3 submissions; or
2. all development candidates fail the promotion gate and the artifacts record
   that the current Run 6 model remains the evidence-supported winner.

Failure to reach `R2 > 0.9` is not itself an implementation failure. Opening
locked data repeatedly, selecting from public leaderboard feedback, or using
external NASA test labels would be protocol failures.

## Hybrid Representation Extension

The standalone temporal screen exposed a representation bottleneck: recent raw
windows do not contain the full-prefix summaries available to the tree models.
Two development-only stages therefore extend the program without reopening
locked data:

1. `PE_10` compares one fixed hybrid CNN using the latest 20 raw cycles against
   the same network using 20 pooled older-history bins followed by the latest
   20 raw cycles. Both variants also receive `screened_drift_pruned` engineered
   features. The treatment advances only with at least four fold wins and at
   least 1% mean RMSE improvement; otherwise the recent-only control is frozen.
2. Architecture Run 8 reads the PE_10 winner and compares XGBoost, hybrid CNN,
   and hybrid GRU. Promotion is paired against the frozen PE_3 calibrated-tree
   OOF control, rather than only the dense-data XGBoost diagnostic. A hybrid
   advances only with at least four fold wins, at least 2% paired RMSE
   improvement, and no material RMS-overprediction regression.

Entry points and editable settings are:

```text
2_architecture_experiments/1_pipeline_experiments/experiments/PE_10/run.py
2_architecture_experiments/1_pipeline_experiments/experiments/PE_10/settings.toml
2_architecture_experiments/2_model_architecture_study/run_hybrid_architecture_study.py
2_architecture_experiments/2_model_architecture_study/hybrid_run_8_settings.toml
```
