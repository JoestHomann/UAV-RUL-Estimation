# Model-guided feature analysis

This follow-up analysis runs after a Phase 1 feature profile has been built. It
fits fixed XGBoost and ExtraTrees models on each outer-training UAV partition
and predicts only the held-out development scenarios. It never loads locked
validation predictions or labelled test outcomes.

For the Run 5 `current20` prefix variant:

```powershell
py 0_data_analysis\model_guided_feature_analysis\run_feature_diagnostics.py `
  --training-features 1_dataset_construction\runs\run_5\current20\5_prefix_feature_engineering\artifacts\training_features.csv.gz `
  --development-features 1_dataset_construction\runs\run_5\current20\5_prefix_feature_engineering\artifacts\development_validation_features.csv.gz `
  --test-features 1_dataset_construction\runs\run_5\current20\5_prefix_feature_engineering\artifacts\test_features.csv.gz `
  --feature-catalog 1_dataset_construction\runs\run_5\current20\6_feature_sets\artifacts\feature_catalog.csv `
  --feature-sets screened_v1 screened_robust screened_acceleration screened_compact `
  --output-dir 0_data_analysis\model_guided_feature_analysis\runs\run_5\current20
```

The output contains cross-fitted residual predictions and grouped metrics,
telemetry-residual correlations, grouped permutation importance, feature drift,
XGBoost/ExtraTrees residual agreement, two summary figures, and a manifest.

Repeat the command with every `current20` path replaced by
`prefix40_stratified` and use
`runs\run_5\prefix40_stratified` as the output directory. This keeps the
feature-recipe comparison paired within each prefix policy and prevents the two
training-row contracts from being mixed.
