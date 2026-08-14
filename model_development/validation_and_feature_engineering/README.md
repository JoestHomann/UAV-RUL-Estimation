# Validation and feature engineering

This module converts the Phase 0 findings into a fixed, leakage-safe model-development contract.

Run the complete workflow from the repository root:

```powershell
py model_development\validation_and_feature_engineering\run_all.py
```

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
- Training uses 20 distinct test-like cutoffs per UAV with equal total UAV weight.

## Feature sets

All generated model columns start with `feature__`. Identifiers, folds, targets, terminal lifetimes, and scenario fields are metadata and must never be selected by position.

| Feature set | Contents | Feature count |
| --- | --- | ---: |
| `age_only` | Cycle and log-cycle | 2 |
| `last_values` | Age plus the final available value of every nonconstant channel | 24 |
| `screened` | Rich temporal features for degradation candidates plus baseline/context summaries | 310 |
| `all_nonconstant` | All generated features from the 22 nonconstant channels | 606 |

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
