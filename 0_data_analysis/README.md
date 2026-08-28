# Data analysis

The data-review work is organized into two self-contained folders:

- `broad_data_review/` contains descriptive plotting scripts and their
  `figures/` outputs.
- `core_data_analysis/` contains temporal, redundancy, drift, and anomaly
  scripts and their `figures/` outputs.
- `model_guided_feature_analysis/` is a post-Phase-1 feature experiment that
  compares feature recipes with fixed, cross-fitted XGBoost and ExtraTrees
  models on development scenarios only.

Run the broad scripts individually, for example:

```powershell
py 0_data_analysis\broad_data_review\plot_descriptive_statistics.py
```

Run the complete core suite with:

```powershell
py 0_data_analysis\core_data_analysis\run_all.py
```
