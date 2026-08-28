# Dataset construction

This workflow converts the Phase 0 findings into fixed, leakage-safe datasets and validation artifacts for model development.

Run the complete workflow from the repository root:

```powershell
py 1_dataset_construction\run_all.py
```

The command above preserves the original 606-feature legacy profile. The
feature experiments use versioned outputs instead:

```powershell
py 1_dataset_construction\run_all.py `
  --profile extended_features --run-name FE_run_1
```

This builds `current20` and `prefix40_stratified` below
`1_dataset_construction/runs/FE_run_1/` without replacing the feature artifacts
used by completed model runs.

Each variant writes `phase_2_interface.json`. This is the copy-ready Phase 2
contract: it records the generated feature count, catalog set counts, observed
prefix-count bounds, training-row count, and all Phase 1 artifact paths. To
refresh only these contracts after upgrading the pipeline, without rebuilding
features, run:

```powershell
py 1_dataset_construction\run_all.py `
  --profile extended_features --run-name FE_run_1 --refresh-interface
```

The configurable launcher in
`0_data_analysis/model_guided_feature_analysis/run_experiments.py` can rebuild
these profiles and run their XGBoost/ExtraTrees diagnostics from one tracked
TOML. Use it for new named feature-engineering runs.

Each numbered folder contains its own `artifacts/` directory. Generated data is ignored by Git, while the `.gitignore` and step README remain tracked.

Artifacts are immutable between steps: a later step reads earlier outputs and writes a new file in its own folder. This makes the complete data flow traceable without hiding which step changed the representation.

## Validation design

- Five outer folds contain 20 non-overlapping UAVs each and are balanced across terminal-lifetime quantiles.
- Each outer-training set has four separate inner UAV-group folds.
- Twenty locked scenarios are reserved for final model comparisons.
- Five development scenarios may be used during feature and model iteration.
- Every scenario contains one prefix from every training UAV.
- Every scenario reproduces the exact 100-value test history-length distribution.
- Long test-like cutoffs are assigned only to training UAVs that can support them before failure.
- Legacy and `current20` training use 20 distinct test-like cutoffs per UAV.
- `prefix40_stratified` uses as many as 40 eligible age-band-stratified cutoffs.
- Every prefix policy gives each UAV a total training weight of one.

## Feature sets

All generated model columns start with `feature__`. Identifiers, folds, targets, terminal lifetimes, and scenario fields are metadata and must never be selected by position.

| Feature set | Contents | Feature count |
| --- | --- | ---: |
| `age_only` | Cycle and log-cycle | 2 |
| `last_values` | Age plus the final available value of every nonconstant channel | 24 |
| `screened` | Rich temporal features for degradation candidates plus baseline/context summaries | 310 |
| `all_nonconstant` | All generated features from the 22 nonconstant channels | 606 |

The `extended_features` profile additionally declares `screened_v1`, `screened_robust`,
`screened_acceleration`, `screened_compact`, and `all_generated_v2`. Their
counts come from the versioned catalog rather than a hard-coded implementation
constant.

The `drift_ablation_features` profile is used by `FE_run_2`. It compares the
unchanged `screened_v1` control with a telemetry 15/16 drift-pruned set and a
set that replaces the removed statistics with robust quantile and local-window
features.

For each nonconstant channel, the feature table contains current and initial values, early-life baseline, baseline deviation, whole-prefix summaries, slopes, deltas, and 5/20/50-cycle recent summaries. `telemetry_07` and `telemetry_16` also receive state-transition and dwell-time features.

Use the functions in `7_fold_fitted_preprocessing/preprocessing.py` to obtain an explicit feature list and fit robust scaling only on the outer- or inner-training rows.

## Numbered workflow

| Step folder | Main artifact | Purpose |
| --- | --- | --- |
| `1_structural_data_audit/` | `artifacts/dataset_audit.json` | Raw-data assertions and per-UAV history summaries |
| `2_UAV_grouped_validation_folds/` | `artifacts/outer_folds.csv` | Disjoint, lifetime-balanced outer and inner UAV folds |
| `3_test_like_validation_scenarios/` | `artifacts/locked_validation_scenarios.csv` | Frozen development and locked scenarios matching test history lengths |
| `4_training_prefixes/` | `artifacts/training_prefixes.csv` | Equal-weight, test-like training cutoffs for every UAV |
| `5_prefix_feature_engineering/` | `artifacts/training_features.csv.gz` | Causal features calculated from rows available by each cutoff |
| `6_feature_sets/` | `artifacts/feature_catalog.csv` | Explicit age-only, last-value, screened, and all-nonconstant feature sets |
| `7_fold_fitted_preprocessing/` | `artifacts/fold_scaler_parameters.csv.gz` | Training-fold-only robust-scaling parameters |
| `8_validation_metrics/` | `artifacts/metric_specification.json` | Metric definitions and reporting groups |
| `9_cycle_only_baseline/` | `artifacts/locked_predictions.csv` | Group-held-out cycle-only reference predictions and metrics |
| `10_automated_leakage_checks/` | `artifacts/verification_report.json` | Fold, cutoff, leakage, causality, finite-value, preprocessing, and baseline checks |

`common.py` contains shared definitions. `run_all.py` executes the numbered steps in dependency order. Every step README documents its exact inputs and outputs.

## Locked-scenario rule

Use development scenarios for ordinary iteration. Evaluate locked scenarios only for serious candidate comparisons, and report the distribution of scenario scores rather than selecting the single best scenario.
