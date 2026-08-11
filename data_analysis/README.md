# Data review

The data-review work is organized into two self-contained folders:

- `broad_data_review/` contains descriptive plotting scripts and their
  `figures/` outputs.
- `core_data_analysis/` contains temporal, redundancy, drift, and anomaly
  scripts and their `figures/` outputs.

Run the broad scripts individually, for example:

```powershell
py data_review\broad_data_review\plot_descriptive_statistics.py
```

Run the complete core suite with:

```powershell
py data_review\core_data_analysis\run_all.py
```
