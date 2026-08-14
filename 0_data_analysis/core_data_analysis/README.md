# Core data analysis

This folder contains the analyses needed to move from descriptive plots to
evidence about temporal degradation, redundancy, distribution shift, and
suspicious UAV histories.

Run the complete suite from the repository root:

```powershell
py 0_data_analysis\core_data_analysis\run_all.py
```

By default, results are written to
`0_data_analysis/core_data_analysis/figures/<analysis-name>/`. Every quantitative
plot has one or more companion CSV files so the conclusions do not depend on
visual inspection alone.

| Script | Main question | Main outputs |
| --- | --- | --- |
| `temporal_rul_analysis.py` | Does each channel consistently change with cycle or RUL within UAVs? | Pooled/partial correlations, per-UAV correlations and slopes, early/late changes, endpoint rolling features |
| `representative_trajectories.py` | Are aggregate findings visible in individual short-, median-, and long-history UAVs? | Train-referenced robust-Z trajectory heatmaps and selected-UAV table |
| `feature_redundancy.py` | Which channels contain overlapping information? | Row/UAV Pearson and Spearman matrices, numeric-order heatmap, strong pairs and clusters |
| `train_test_drift.py` | Do train and test differ after accounting for observed age? | Row, UAV, age-band, and exact-age endpoint drift tables |
| `anomaly_analysis.py` | Which readings and UAV histories require review? | MAD flags, jump flags, persistent-shift candidates, similar histories, ranked UAV review list |
| `channel_classification.py` | What initial role does the combined evidence support for each channel? | Transparent threshold-based evidence table and overview plot |

The final classification is a screening summary, not a feature-selection result.
It intentionally keeps anomaly and drift warnings separate from a channel's
primary role. Channels should only be removed or transformed after the relevant
finding is confirmed and later UAV-grouped validation supports the action.

Within-UAV correlations indicate whether a channel changes consistently during
individual histories. The age-controlled pooled association instead asks
whether channel values differ with remaining lifetime among rows observed at a
similar cycle; it should not be interpreted as proof of causality.

Individual scripts accept `--help`. Common options include `--train-csv`,
`--test-csv`, `--channels`, `--output-dir`, and `--dpi`. The implementation uses
NumPy, pandas, Matplotlib, and SciPy; it does not require scikit-learn.
