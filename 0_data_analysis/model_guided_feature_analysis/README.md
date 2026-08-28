# Model-guided feature analysis

This follow-up analysis runs after a Phase 1 feature profile has been built. It
fits fixed XGBoost and ExtraTrees models on each outer-training UAV partition
and predicts only the held-out development scenarios. It never loads locked
validation predictions or labelled test outcomes.

## Named experiments

`run_experiments.py` executes the named runs in
`feature_engineering_experiments.toml`. This one TOML owns the Phase 1 feature
profile, prefix strategies and counts, named runs, diagnostic settings, and
model parameters. Each run can rebuild or reuse Phase 1 and then run the
fixed-fold diagnostics for every configured prefix variant.

Run all enabled experiments from the repository root:

```powershell
.venv\Scripts\python.exe `
  0_data_analysis\model_guided_feature_analysis\run_experiments.py
```

Inspect the registry or generated commands without starting model fits:

```powershell
.venv\Scripts\python.exe `
  0_data_analysis\model_guided_feature_analysis\run_experiments.py --list
.venv\Scripts\python.exe `
  0_data_analysis\model_guided_feature_analysis\run_experiments.py --dry-run
```

Select one run or force Phase 1 to be rebuilt:

```powershell
.venv\Scripts\python.exe `
  0_data_analysis\model_guided_feature_analysis\run_experiments.py `
  --run FE_run_1 --rebuild-phase1
```

To add a run, duplicate `[runs.FE_run_1]` and its nested model tables under a
new name. Assign a distinct `phase_1_run_name` and `output_root`, then edit its
prefix variants, feature sets, models, model parameters, permutation budget,
seed, or plotting DPI. New cutoff policies are added under
`[prefix_variants.*]` and referenced by the selected profile.

`FE_run_2` is the paired telemetry-drift ablation. It rebuilds only `current20`
and compares `screened_v1`, `screened_drift_pruned`, and
`screened_drift_replaced` with the same XGBoost and ExtraTrees settings:

```powershell
.venv\Scripts\python.exe `
  0_data_analysis\model_guided_feature_analysis\run_experiments.py `
  --run FE_run_2
```

`phase_1_mode = "reuse"` requires an existing `phase_2_interface.json` for
each variant. `phase_1_mode = "rebuild"` runs Phase 1 first. The launcher
checks the cutoff strategy, count, seed, profile feature sets, and requested
diagnostics against the generated interface so stale artifacts cannot be
reused accidentally.

Generated diagnostics and launcher manifests use the configured `output_root`.
The default is `0_data_analysis/model_guided_feature_analysis/runs/FE_run_1/`.

## Direct diagnostics

The direct command below remains useful for one-off diagnostics.

For the `FE_run_1` `current20` prefix variant:

```powershell
py 0_data_analysis\model_guided_feature_analysis\run_feature_diagnostics.py `
  --training-features 1_dataset_construction\runs\FE_run_1\current20\5_prefix_feature_engineering\artifacts\training_features.csv.gz `
  --development-features 1_dataset_construction\runs\FE_run_1\current20\5_prefix_feature_engineering\artifacts\development_validation_features.csv.gz `
  --test-features 1_dataset_construction\runs\FE_run_1\current20\5_prefix_feature_engineering\artifacts\test_features.csv.gz `
  --feature-catalog 1_dataset_construction\runs\FE_run_1\current20\6_feature_sets\artifacts\feature_catalog.csv `
  --feature-sets screened_v1 screened_robust screened_acceleration screened_compact `
  --output-dir 0_data_analysis\model_guided_feature_analysis\runs\FE_run_1\current20
```

The output contains cross-fitted residual predictions and grouped metrics,
telemetry-residual correlations, grouped permutation importance, feature drift,
XGBoost/ExtraTrees residual agreement, two summary figures, and a manifest.

Repeat the command with every `current20` path replaced by
`prefix40_stratified` and use
`runs\FE_run_1\prefix40_stratified` as the output directory. This keeps the
feature-recipe comparison paired within each prefix policy and prevents the two
training-row contracts from being mixed.
