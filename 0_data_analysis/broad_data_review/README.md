# Broad data review

Each script creates one overview figure. Scripts that calculate exact metrics
also save companion CSV files beside the figure. Input and output defaults are
resolved from the script location, so the commands work from any directory.

| Analysis | Script | Default output directory |
| --- | --- | --- |
| Cycle-wise telemetry overview | `plot_telemetry_over_cycles.py` | `figures/telemetry_over_cycles` |
| Full descriptive statistics | `plot_descriptive_statistics.py` | `figures/descriptive_statistics` |
| Row-level diagnostics | `plot_row_level_statistics.py` | `figures/row_level_statistics` |
| Equal-weight UAV statistics | `plot_uav_level_statistics.py` | `figures/uav_level_statistics` |
| Age-band statistics | `plot_age_band_statistics.py` | `figures/age_band_statistics` |
| History-length distributions | `plot_history_length_distributions.py` | `figures/history_length_distributions` |
| Histograms and box plots | `plot_histograms_boxplots.py` | `figures/histograms_boxplots` |
| Within/between variance | `plot_within_between_variance.py` | `figures/within_between_variance` |
| Constant-feature detection | `plot_constant_features.py` | `figures/constant_features` |
| Flatline duration | `plot_flatline_duration.py` | `figures/flatline_duration` |
| Extreme-reading investigation | `tukeys_range_test.py` | `figures/Tukey_extreme_readings_investigation` |

From the repository root, run a script with:

```powershell
py 0_data_analysis\broad_data_review\plot_descriptive_statistics.py
```

The common options are `--train-csv`, `--output-dir`, `--channels`, and
`--dpi`. The history-length script also accepts `--test-csv`. The flatline
script accepts `--minimum-run`; an exact-value run of at least five cycles is
the default. The extreme-reading script accepts `--iqr-multiplier`; its robust
bounds are fitted on training data only and applied unchanged to test data.

The deeper temporal, redundancy, matched-age drift, and anomaly analyses are
kept separately in `../core_data_analysis`. Run that suite with:

```powershell
py 0_data_analysis\core_data_analysis\run_all.py
```

See `../core_data_analysis/README.md` for its generated plots and CSV tables.

Definitions:

- Row-level calculations give every recorded cycle equal weight, so UAVs with
  longer histories contribute more observations.
- UAV-level calculations first average each channel within a UAV, then
  describe the resulting 100 equally weighted UAV values.
- Age bands are 1–50, 51–100, 101–200, and over 200 flight cycles.
- Within/between variance uses the standard sum-of-squares decomposition with
  UAV as the group.
- A flatline is a run of consecutive exactly equal readings within one UAV.
