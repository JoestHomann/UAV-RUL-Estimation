# UAV Remaining Useful Life Estimation


Based on Lecture Notes:

![Machine learning pipeline](../data_review/broad_data_review/figures/machine_learning_pipeline/machine_learning_pipeline.png)

This file talks about the bullet points 1 to 5. Bullet point to is ignored as the dataset already exists.

## Objective

Complete a reproducible data and validation audit before model development begins.


## Broad Data Review

![Training data preview](../data_review/broad_data_review/figures/data_overview/training_data_preview.png)

The input data (train.csv) consists of 31 columns, the uav_id, the flight cycle of the UAV, 28 telemetry values and the RUL. These telemetry values need to be analyzed to what extent they correlate with the RUL. The training is therefore based on these values and the training cycle number but before that unnecessary telemetry data should be removed from the dataset.

The test dataset obviously does not have the RUL column.

The output data on the other hand, so the value the model should predict is simply a scalar value representing the RUL (Remaining Useful Lifetime).

## Statistical Review


### Cycle-wise fleet-level telemetry trend analysis
To find out which telemetry data is useful for us we can plot each telemetry column over its flight cycle number and check how they evolve.

![Cycle-wise median and mean telemetry trends](../data_review/broad_data_review/figures/telemetry_over_cycles/telemetry_overview.png)

This plot shows the median (blue) and the average value of each telemetry channel (light blue) over the number of flight cycles.
Here we can see that telemetry_03, telemetry_08, telemetry_14, telemetry_17, telemetry_20 and telemetry_27 stay constant over all UAVs and cycles. Therefore these channels are likely to be not useful for training the model.
But important to note is that at the end there are only a limited number of UAVs "surviving" (or data points available), therefore we have a survivor bias here.

**Findings:**
- Channels 03, 08, 14, 17, 20, 27 likely irrelevant

### Descriptive statistics

Count, mean, median, standard deviation, minimum, maximum, quantiles, and IQR show the scale, skewness, spread, and extreme values of each telemetry channel. These results support scaling, transformation, and anomaly-review decisions.

In other words, here we see how much the data points vary in general for each telemetry channel.

- Gray line and × marks: complete minimum-to-maximum range
- Light blue line: central 90% of values, from P05 to P95
- Dark blue line: IQR—the middle 50%, from P25 to P75
- White vertical mark: median
- Black diamond: mean
- Text: observation count, standard deviation, and IQR (Interquartile Range, measures the spread of the middle 50% of the data)


![Telemetry descriptive statistics](../data_review/broad_data_review/figures/descriptive_statistics/descriptive_statistics.png)


**Findings:**
- Again we can see that telemetry_03, 08, 14, 17, 20, and 27 are effectively constant. Therefore they dont really provide useful variation and are removal candidates.
- Channel scales differ enormously. For example, some values are near 10 while others are near 50,000. **Standardization or robust scaling** is therefore highly advisable here.
- telemetry_01, 04, 10, 11, 18, 24, and 26 have min–max ranges much wider than their central distributions. This indicates rare extreme readings, outliers, or distinct operating regimes that should be investigated (see subchapter *Tukey's Range Test*)
- Separation between the mean and median indicates asymmetry. The largest differences occur in channels such as telemetry_07, 16, 23, 18, and 25, although the overall skew is moderate. Use **Yeo–Johnson transformation** for linear or neural models. We need to proof if these channels actually correlate with RUL -> **Turn channel on or off and check if validation results improve**
- A large SD relative to the IQR indicates that extreme values influence the standard deviation. **Robust scaling** (yes, thats the name) may therefore be safer than ordinary mean/SD scaling for some channels. Do not use this scaling for tree-based models (Random Forest, XGBoost, ...)


#### Tukey's Range Test

To investigate the unusually wide min–max ranges, extreme-value boundaries are calculated from the training data as:

$$
\mathrm{lower\ bound} = Q_1 - 3 \times \mathrm{IQR}
$$

$$
\mathrm{upper\ bound} = Q_3 + 3 \times \mathrm{IQR}
$$

The training-derived boundaries are then applied unchanged to both the training and test datasets. Red points are training extremes, orange points are test extremes, and the dashed lines show the lower and upper boundaries. Looking at the extreme readings over flight cycle helps distinguish isolated sensor spikes from sustained UAV-specific operating regimes.

![Extreme telemetry readings by flight cycle](../data_review/broad_data_review/figures/Tukey_extreme_readings_investigation/Tukey_extreme_readings_overview.png)

**Findings:**

- For `telemetry_04`, `telemetry_10`, and `telemetry_11`, approximately 97% of the training extremes are isolated readings. This pattern is more consistent with recurring single-cycle spikes than with sustained degradation. -> **Use individual and synchronized-event flags: Flagging them for the model could improve generalization (check if this actually improves the results)**
- The extremes in `telemetry_01` and `telemetry_18` form long consecutive runs but are concentrated in only 8–9 training UAVs. They are therefore more likely to represent UAV-specific operating regimes or calibration differences.
-> **Use regime flags: Flagging them for the model could improve generalization (check if this actually improves the results)**
- `telemetry_26` also contains mostly sustained extremes and affects 48 training UAVs. -> **Use regime flag: Flagging them for the model could improve generalization (check if this actually improves the results)**
- `telemetry_24` shows mixed behavior and contains extremes in both training and test data. Removing or clipping all of these values would therefore be risky. -> **Don't flag for now but could potentially be flagged as low- and high-extreme flags (check too)**
- Extreme values should not be removed automatically. Any clipping boundaries must be learned from the training portion of each validation fold to avoid **data leakage**.


### Row-level statistics

The plot describes the shape of each telemetry distribution across all recorded cycles.

- Left panel - (mean - median) / SD:
    - Near zero: roughly symmetric distribution.
    - Positive: unusually high values pull the mean upward.
    - Negative: unusually low values pull the mean downward.
    - Larger magnitude means stronger asymmetry.

- Right panel - IQR / SD:
    - Low value: extreme readings or multiple regimes inflate the SD.
    - Around 1.35: approximately normal-shaped distribution.
    - High value: more concentrated, bounded, or discrete distribution.
    - Zero: effectively constant channel.


![Row-level telemetry statistics](../data_review/broad_data_review/figures/row_level_statistics/row_level_statistics.png)

**Findings:**
- Several channels are asymmetric or affected by extreme values, especially telemetry_07, 16, 18, 23, and 25. -> **Robust scaling** 
- Channels such as 18, 24, and 26 have particularly low IQR-to-SD ratios and deserve **anomaly or operating-regime review**.

### UAV-level statistics

The UAV-level analysis first averages each telemetry channel within a UAV and then compares the 100 UAV summaries. This gives every training UAV equal weight and reveals whether variation is associated with UAV identity.

- Light-blue dot: one UAV’s mean telemetry value
- Box: middle 50% of UAV means
- Line inside the box: median UAV
- Whiskers: typical range
- Black diamond: mean across the 100 UAVs
- Isolated dots: UAVs whose average differs substantially from the fleet

![UAV-level telemetry statistics](../data_review/broad_data_review/figures/uav_level_statistics/uav_level_statistics.png)

This plot tells us about the following things:
- Between-UAV variation: A wide distribution means UAVs operate at different average levels -> **Potential context feature**
- Typical fleet value: The median and box describe a representative (median)UAV.
- Skewness: Separation between the mean and median indicates an asymmetric distribution.
- Operating regimes: Separate groups of dots can reveal distinct configurations or conditions.
- Unusual UAVs: Isolated dots identify UAVs requiring investigation.
Constant features: A collapsed box and overlapping dots indicate no useful UAV-level variation.
- Context candidates: Channels that vary strongly between UAVs but little within them may represent operating context or UAV identity.


**Findings:**
- telemetry_03, 08, 14, 17, 20, and 27 are effectively constant across UAVs. -> **Again, removal condidates**
- Channels such as telemetry_01, 06, 12, 18, and 26 show substantial UAV-to-UAV differences. They may represent operating conditions, configuration, or UAV identity rather than degradation alone. -> **Potential context or baseline features, might use Yeo–Johnson transformation and/or robust scaling**
- Distinct groups or isolated UAVs may indicate operating regimes or anomalies. -> **Flagging**
- Narrow boxes suggest UAVs have similar average values; wide boxes indicate stronger between-UAV variation. -> **Seperation of baseline values and deviation value (potential degradation information)**

### Age-band statistics

Telemetry statistics are compared across cycles 1–50, 51–100, 101–200, and over 200. Changes between these bands help distinguish degradation-related signals from operating-context signals.

- Dark-blue line: median telemetry value.
- Light-blue line: mean telemetry value.
- Shaded region: IQR—the middle 50% of readings.
- n below each band: number of contributing rows.
- Each subplot has its own vertical scale.

![Telemetry statistics by flight-cycle age band](../data_review/broad_data_review/figures/age_band_statistics/age_band_statistics.png)


**Findings:**
- Potential age-linked degradation indicators: telemetry_06, 11, 12, 13, 15, 16, 19, 21, 22, 23, 24, 25, 26, and 28.
    - Increasing with age: telemetry_12, 13, 15, 19, 21, and 22.
    - Decreasing with age: telemetry_06, 11, 16, 23, 24, 25, 26, and 28.
- Age-stable channels: telemetry_01, 02, 04, 05, 09, and 10. Effectively constant channels are telemetry_03, 08, 14, 17, 20, and 27.
- Noticeable mean–median separation: telemetry_07, 15, 16, 18, 19, 21, and 23, suggesting skewness or multiple operating regimes. -> **Baseline seperation valuable**
- Clearly widening IQR at later ages: telemetry_06, 07, 11, 12, 15, 16, 19, 21, 22, 23, 24, 25, 26, and 28, suggesting increasing variability or survivor/fleet-composition effects.

-> **Especially the age-linked indicators might be the most important channels**

### History-length distributions

Train and test history-length distributions show how much information is available at prediction time. The difference between the two distributions should guide the selection of test-like validation cutoffs. So it tells us how we must construct validation and training examples. The central point is that the **model must learn to predict from incomplete histories**, not only from complete run-to-failure sequences.

![Train and test UAV history-length distributions](../data_review/broad_data_review/figures/history_length_distributions/history_length_distributions.png)


**Findings:**
**Findings directly supported by the history-length diagram:**

- **Histogram and ECDF:** Test histories are generally shorter than training histories, so training and validation samples should use partial UAV prefixes.
- **ECDF at 145 cycles:** Approximately 47% of test UAVs stop before the shortest complete training lifetime; therefore, validation must emphasize cutoffs between cycles 38 and 144.
- **Test histogram and ECDF:** Sample cutoff lengths from the observed test distribution rather than uniformly across all cycles.
- **Box plots:** Train histories span 145–525 cycles, while test histories span 38–475; therefore, the model must handle variable-length inputs.
- **Sparse histogram right tail and box-plot outliers:** Few UAVs have very long histories, so validation results at long cutoffs are less reliable.
- **Different history lengths across UAVs:** Give every UAV equal total weight so long histories do not dominate training.
- **Full history-length range:** Report performance by cutoff band, such as cycles 1–50, 51–100, 101–200, and over 200.

-> **Findings are here about the design of the training and validation datasets, not which channels to use or how to handle them**


**General training and validation rules—not read directly from the diagram:**

- Split by `uav_id`; all prefixes from one UAV must remain in the same fold.
- Use five fixed folds as a design choice, giving approximately 20 validation UAVs per fold.
- Use a fixed number of training cutoffs per UAV; 20 is a configurable starting point.
- Use 20 fixed test-like validation scenarios as a configurable robustness check.
- Build features using only cycles available up to the selected cutoff.
- Never use complete-history length, final cycle, terminal lifetime, future telemetry, or post-cutoff information as features.
- Fit scaling, transformations, feature selection, and anomaly thresholds inside each training fold.
- Retrain the selected pipeline on all training UAVs.
- Predict once per test UAV using its complete available history.


### Histograms and box plots

Histograms and box plots expose skewness, clusters, operating regimes, spikes, saturation, and outliers. They support transformation choices and the anomaly-handling policy.

![Telemetry histograms and box plots](../data_review/broad_data_review/figures/histograms_boxplots/histograms_boxplots.png)

**Findings:**

- Effectively constant: `telemetry_03`, `08`, `14`, `17`, `20`, and `27` -> **Remove from the model because they contain no useful variation**
- Discrete or regime-like: `telemetry_05`, `07`, `16`, and `18` -> **Investigate as operating-state or context variables rather than continuous degradation signals**
- Approximately smooth and unimodal: `telemetry_02`, `06`, `09`, `13`, and `22` -> **No distribution transformation appears necessary; scale only if required by the model**
- Noticeably skewed: `telemetry_15`, `18`, `21`, `23`, `24`, `25`, `26`, and `28` -> **Consider robust scaling or Yeo–Johnson transformation for linear and neural models**
- Strong tails or extreme readings: `telemetry_01`, `04`, `10`, `11`, `18`, `24`, and `26` -> **Flag and investigate extreme values; use robust scaling and do not remove observations automatically**


### Within-UAV versus between-UAV variance

The variance decomposition separates changes occurring within UAV histories from persistent differences between UAVs. Channels dominated by within-UAV variation are candidates for sequence features, while channels dominated by between-UAV variation may describe operating context or UAV identity.

The plot divides each channel’s total variation into two sources:
- Dark blue: variation within the same UAV over time
- Light blue: variation between different UAVs’ average levels

![Within-UAV versus between-UAV telemetry variance](../data_review/broad_data_review/figures/within_between_variance/within_between_variance.png)

**Findings:**
- Within-UAV dominated: telemetry_02, 04, 05, 07, 09, 10, 11, 13, 15, 16, 19, 21, 22, 23, 24, 25, and 28 -> **Potential sequence features: current value, slope, rolling statistics, recent change, and baseline deviation**
- Mixed within/between variation: telemetry_06 and 12 -> **Use both the UAV baseline and temporal-change features**
- Between-UAV dominated: telemetry_01, 18, and 26 -> **Treat primarily as context or baseline features; verify that their relationship generalizes to unseen UAVs**
- Effectively constant: telemetry_03, 08, 14, 17, 20, and 27 -> **Removal candidates**


**Important caveats:**
- Within-UAV variation may be degradation, operating-state changes, or noise; the plot does not distinguish them.
- Between-UAV variation may be useful context or unwanted UAV identity.
- The graph shows variance proportions, not the absolute amount of variation.
- Final decisions require age/RUL association tests and UAV-grouped validation.


### Flatline duration

Flatline analysis measures consecutive exactly equal readings within each UAV. Long or frequent flatlines can indicate a stuck sensor, a quantized state variable, or an intentionally static operating parameter and should be interpreted before being treated as anomalies.

- Left, light blue: median of each UAV’s longest unchanged run.
- Left, dark blue: longest run observed anywhere.
- The horizontal axis is logarithmic.
- Right, yellow: percentage of UAVs containing a flatline of at least five cycles.
- Right, orange: percentage of all rows belonging to those flatlines.

![Telemetry flatline duration and prevalence](../data_review/broad_data_review/figures/flatline_duration/flatline_duration.png)

**Findings:**
- Fully constant: telemetry_20 and 27 -> **100% of their rows are flatlined; remove them**
- Effectively constant: telemetry_03, 08, 14, and 17 -> **More than 92% of their rows belong to long flatlines; remove them**
- Strongly state-like: telemetry_07 -> **Flatlines occur in every UAV and cover about 97% of rows; treat it as a discrete operating-state variable rather than a continuous sensor**
- Moderately quantized: telemetry_16 -> **90% of UAVs contain a run of at least five cycles, but only about 5% of rows are affected; consider state-level and transition features**
- Minimal flatlining: remaining channels -> **No broad flatline problem is evident**

## Summary

### 1. Channel removal

- Effectively constant: `telemetry_03`, `08`, `14`, `17`, `20`, and `27` -> **Remove them from the model because they contain no meaningful predictive variation. Keep the automated constant checks as data-quality gates.** (Source: [Within-UAV versus between-UAV variance](#within-uav-versus-between-uav-variance), [variance plot](../data_review/broad_data_review/figures/within_between_variance/within_between_variance.png); [Flatline duration](#flatline-duration), [flatline plot](../data_review/broad_data_review/figures/flatline_duration/flatline_duration.png))

### 2. Feature roles and engineering

- Age-linked candidates: increasing `telemetry_12`, `13`, `15`, `19`, `21`, and `22`; decreasing `telemetry_06`, `11`, `16`, `23`, `24`, `25`, `26`, and `28` -> **Prioritize them for degradation-oriented sequence features, subject to within-UAV and grouped-validation confirmation.** (Source: [Age-band statistics](#age-band-statistics), [age-band plot](../data_review/broad_data_review/figures/age_band_statistics/age_band_statistics.png))
- Within-UAV dominated: `telemetry_02`, `04`, `05`, `07`, `09`, `10`, `11`, `13`, `15`, `16`, `19`, `21`, `22`, `23`, `24`, `25`, and `28` -> **Test current value, slope, rolling statistics, recent change, and deviation from an early-life baseline.** (Source: [Within-UAV versus between-UAV variance](#within-uav-versus-between-uav-variance), [variance plot](../data_review/broad_data_review/figures/within_between_variance/within_between_variance.png))
- Mixed within/between variation: `telemetry_06` and `12` -> **Use both prefix-derived UAV baselines and temporal-change features.** (Source: [Within-UAV versus between-UAV variance](#within-uav-versus-between-uav-variance), [variance plot](../data_review/broad_data_review/figures/within_between_variance/within_between_variance.png))
- Between-UAV dominated: `telemetry_01`, `18`, and `26` -> **Treat primarily as context or baseline features and verify that they generalize to unseen UAVs.** (Source: [UAV-level statistics](#uav-level-statistics), [UAV-level plot](../data_review/broad_data_review/figures/uav_level_statistics/uav_level_statistics.png); [Within-UAV versus between-UAV variance](#within-uav-versus-between-uav-variance), [variance plot](../data_review/broad_data_review/figures/within_between_variance/within_between_variance.png))
- Discrete or regime-like: `telemetry_05`, `07`, `16`, and `18` -> **Investigate operating-state, level, transition-count, and dwell-time features rather than assuming continuous degradation.** (Source: [Histograms and box plots](#histograms-and-box-plots), [distribution plot](../data_review/broad_data_review/figures/histograms_boxplots/histograms_boxplots.png); [Flatline duration](#flatline-duration), [flatline plot](../data_review/broad_data_review/figures/flatline_duration/flatline_duration.png))
- Age-stable but nonconstant: `telemetry_01`, `02`, `04`, `05`, `09`, and `10` -> **Do not treat them as primary age-degradation indicators; retain them only if context features or grouped validation add value.** (Source: [Age-band statistics](#age-band-statistics), [age-band plot](../data_review/broad_data_review/figures/age_band_statistics/age_band_statistics.png))

### 3. Scaling, transformations, and anomaly handling

- Strongly different channel scales -> **Use robust scaling for linear, neural, distance-based, and similar scale-sensitive models; tree-based models do not require scaling.** (Source: [Descriptive statistics](#descriptive-statistics), [descriptive-statistics plot](../data_review/broad_data_review/figures/descriptive_statistics/descriptive_statistics.png))
- Skewed continuous channels: `telemetry_15`, `21`, `23`, `24`, `25`, `26`, and `28` -> **Test robust scaling or Yeo-Johnson transformation for linear and neural models; retain the transformation only if grouped validation improves.** (Source: [Histograms and box plots](#histograms-and-box-plots), [distribution plot](../data_review/broad_data_review/figures/histograms_boxplots/histograms_boxplots.png); [Row-level statistics](#row-level-statistics), [row-level plot](../data_review/broad_data_review/figures/row_level_statistics/row_level_statistics.png))
- Mostly isolated extremes: `telemetry_04`, `10`, and `11` -> **Test individual-spike and synchronized-event flags instead of automatically removing the rows.** (Source: [Tukey's Range Test](#tukeys-range-test), [extreme-reading plot](../data_review/broad_data_review/figures/Tukey_extreme_readings_investigation/Tukey_extreme_readings_overview.png))
- Mostly sustained UAV-specific extremes: `telemetry_01`, `18`, and `26` -> **Test regime flags because these patterns are more consistent with context or calibration differences than isolated sensor spikes.** (Source: [Tukey's Range Test](#tukeys-range-test), [extreme-reading plot](../data_review/broad_data_review/figures/Tukey_extreme_readings_investigation/Tukey_extreme_readings_overview.png))
- Mixed extremes: `telemetry_24` -> **Do not clip or remove automatically; test separate low- and high-extreme flags if validation supports them.** (Source: [Tukey's Range Test](#tukeys-range-test), [extreme-reading plot](../data_review/broad_data_review/figures/Tukey_extreme_readings_investigation/Tukey_extreme_readings_overview.png))
- All anomaly boundaries, scalers, and transformations -> **Fit them using only the training portion of each validation fold.** (Source: [Descriptive statistics](#descriptive-statistics); [Tukey's Range Test](#tukeys-range-test))

### 4. Training and validation construction

- Test histories are shorter than complete training histories, with 47% of test UAVs ending before cycle 145 -> **Train and validate on partial UAV prefixes, emphasizing cutoffs from cycles 38-144 and sampling from the observed test-length distribution.** (Source: [History-length distributions](#history-length-distributions), [history-length plot](../data_review/broad_data_review/figures/history_length_distributions/history_length_distributions.png))
- History lengths vary from 145-525 cycles in train and 38-475 in test -> **Support variable-length inputs, give each UAV equal total weight, report metrics by cutoff band, and treat very long-cutoff results cautiously.** (Source: [History-length distributions](#history-length-distributions), [history-length plot](../data_review/broad_data_review/figures/history_length_distributions/history_length_distributions.png))
- UAV grouping -> **Split by `uav_id` before generating prefixes; keep every prefix from one UAV in the same fold.** (Source: [History-length distributions - general validation rules](#history-length-distributions))
- Recommended starting design -> **Use five fixed UAV folds, a fixed number of cutoffs per training UAV (for example 20), and 20 locked test-like validation scenarios. Treat these numbers as configurable design choices.** (Source: [History-length distributions - general validation rules](#history-length-distributions))
- Prefix leakage control -> **Build every feature from cycles available at the cutoff only; never use full-history length, final cycle, terminal lifetime, future telemetry, or other post-cutoff information.** (Source: [History-length distributions - general validation rules](#history-length-distributions))
- Final training and inference -> **Retrain the selected pipeline on all training UAVs and predict once per test UAV from its complete available history.** (Source: [History-length distributions - general validation rules](#history-length-distributions))

### 5. Caveats and required validation

- Late-cycle fleet trends and widening IQRs may partly reflect survivor or fleet-composition effects -> **Confirm apparent degradation using consistent within-UAV trends rather than fleet aggregates alone.** (Source: [Cycle-wise fleet-level telemetry trend analysis](#cycle-wise-fleet-level-telemetry-trend-analysis), [cycle-wise plot](../data_review/broad_data_review/figures/telemetry_over_cycles/telemetry_overview.png); [Age-band statistics](#age-band-statistics), [age-band plot](../data_review/broad_data_review/figures/age_band_statistics/age_band_statistics.png))
- Within-UAV variation may be degradation, noise, or operating-state change, while between-UAV variation may be useful context or unwanted identity -> **Use UAV-grouped ablation tests to decide which raw and engineered features generalize.** (Source: [Within-UAV versus between-UAV variance](#within-uav-versus-between-uav-variance), [variance plot](../data_review/broad_data_review/figures/within_between_variance/within_between_variance.png))
- Skewness, extremes, flatlines, and operating regimes do not prove predictive value -> **Keep transformations, flags, and channels only when locked grouped validation improves RUL performance.** (Source: [Row-level statistics](#row-level-statistics), [row-level plot](../data_review/broad_data_review/figures/row_level_statistics/row_level_statistics.png); [Histograms and box plots](#histograms-and-box-plots), [distribution plot](../data_review/broad_data_review/figures/histograms_boxplots/histograms_boxplots.png); [Flatline duration](#flatline-duration), [flatline plot](../data_review/broad_data_review/figures/flatline_duration/flatline_duration.png))


## Core Data Analysis