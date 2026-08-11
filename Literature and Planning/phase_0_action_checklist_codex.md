# Phase 0 Action Checklist

This is a companion checklist. It does not replace `phase_0_checklist.md`.

Labels:

- **[CODE]** Automate or implement.
- **[REVIEW]** Inspect and interpret manually.
- **[DECISION]** Agree on and freeze a project rule.

## A. Confirm the project contract

- [ ] **[DECISION]** Predict RUL at the final available cycle of every test UAV.
- [ ] **[DECISION]** Use `RUL` as the target column.
- [ ] **[DECISION]** Use `uav_id` only as a grouping key, never as a numeric feature.
- [ ] **[DECISION]** Keep raw competition data, derived row data, models, and submissions private and ignored by Git.
- [ ] **[CODE]** Record file hashes, shapes, column order, data types, package versions, and random seeds.

## B. Run the structural data audit

- [ ] **[CODE]** Verify all expected columns exist.
- [ ] **[CODE]** Verify test data does not contain `RUL`.
- [ ] **[CODE]** Check missing and infinite values.
- [ ] **[CODE]** Check duplicate rows.
- [ ] **[CODE]** Check duplicate `(uav_id, flight_cycle)` keys.
- [ ] **[CODE]** Confirm train and test UAV IDs do not overlap.
- [ ] **[CODE]** Confirm every history starts at cycle 1.
- [ ] **[CODE]** Confirm cycles are ordered and consecutive within every UAV.
- [ ] **[CODE]** Confirm `RUL + flight_cycle` is constant within every training UAV.
- [ ] **[CODE]** Create a per-UAV table with row count, final cycle, and terminal lifetime.
- [ ] **[REVIEW]** Investigate every failed assertion or unusual history length.

Expected findings to reproduce:

- [ ] 100 training UAVs and 100 test UAVs.
- [ ] No missing/non-finite values, duplicate keys, or cycle gaps.
- [ ] Exact terminal-lifetime identity for all training UAVs.

## C. Audit distributions and scales

- [ ] **[CODE]** Calculate count, mean, median, standard deviation, minimum, maximum, quantiles, and IQR for every channel.
- [ ] **[CODE]** Calculate statistics at row level and UAV-summary level.
- [ ] **[CODE]** Repeat statistics for age bands 1-50, 51-100, 101-200, and over 200 cycles.
- [ ] **[CODE]** Plot train/test history-length distributions.
- [ ] **[CODE]** Plot per-channel histograms and robust box plots.
- [ ] **[CODE]** Compare within-UAV variance with between-UAV variance.
- [ ] **[CODE]** Identify constant and near-constant channels.
- [ ] **[CODE]** Measure flatline duration for every UAV/channel pair.
- [ ] **[REVIEW]** Inspect channels with extreme scales, skewness, saturation, or multiple regimes.
- [ ] **[DECISION]** Freeze the low-variance removal rule.
- [ ] **[DECISION]** Do not scale tree-model inputs.
- [ ] **[DECISION]** Fit scaling inside training folds only for linear, distance-based, or neural models.

Expected findings to verify:

- [ ] `telemetry_20` and `telemetry_27` are constant.
- [ ] `telemetry_08` and `telemetry_14` are static within most UAVs.

## D. Audit temporal degradation signals

For each telemetry channel:

- [ ] **[CODE]** Calculate pooled Pearson and Spearman correlations with `RUL`.
- [ ] **[CODE]** Calculate pooled correlations with `flight_cycle`.
- [ ] **[CODE]** Calculate within-UAV correlations.
- [ ] **[CODE]** Calculate per-UAV linear and robust slopes.
- [ ] **[CODE]** Calculate the percentage of UAVs sharing the same trend direction.
- [ ] **[CODE]** Calculate early-to-late changes and effect sizes.
- [ ] **[CODE]** Calculate last-minus-first values.
- [ ] **[CODE]** Calculate rolling means, volatility, deltas, and slopes over 5, 20, and 50 cycles.
- [ ] **[CODE]** Measure association with RUL after accounting for observed age.
- [ ] **[CODE]** Plot representative short-, median-, and long-lived histories.
- [ ] **[REVIEW]** Review approximately 12 representative training UAVs.
- [ ] **[REVIEW]** Review test UAVs at comparable observed ages.
- [ ] **[REVIEW]** Classify every channel as:
  - stable degradation signal;
  - static UAV context;
  - age-confounded signal;
  - noisy or weak;
  - constant;
  - suspicious or unresolved.
- [ ] **[DECISION]** Freeze the initial channel-classification table.

## E. Audit anomalies and suspicious histories

- [ ] **[CODE]** Flag robust Z-score/MAD and IQR extremes.
- [ ] **[CODE]** Flag unusually large cycle-to-cycle jumps.
- [ ] **[CODE]** Flag long flatlines and saturation.
- [ ] **[CODE]** Flag abrupt permanent level shifts.
- [ ] **[CODE]** Search for copied or nearly identical histories across UAVs.
- [ ] **[CODE]** Flag unusually short and long lifetimes.
- [ ] **[CODE]** Create UAV-level summary features.
- [ ] **[CODE]** Rank unusual UAVs with Isolation Forest as a diagnostic.
- [ ] **[CODE]** Use robust Mahalanobis distance only if covariance is stable.
- [ ] **[REVIEW]** Inspect the approximately 10 highest-priority flagged UAVs.
- [ ] **[REVIEW]** Label each reviewed flag as:
  - likely data error;
  - plausible operating regime;
  - possible degradation behavior;
  - unclear.
- [ ] **[DECISION]** Keep observations by default.
- [ ] **[DECISION]** Require a group-held-out sensitivity test before excluding or down-weighting data.
- [ ] **[DECISION]** Do not treat Grubbs' test as a general time-series deletion rule.

## F. Audit redundancy

- [ ] **[CODE]** Calculate Pearson and Spearman feature-correlation matrices.
- [ ] **[CODE]** Create a clustered correlation heatmap.
- [ ] **[CODE]** Identify highly correlated channel groups.
- [ ] **[CODE]** Compare redundancy at row and UAV-summary levels.
- [ ] **[CODE]** Run PCA on UAV summaries for visualization only.
- [ ] **[REVIEW]** Check whether clusters represent age, lifetime, operating regime, anomalies, or train/test membership.
- [ ] **[DECISION]** Do not use t-SNE or UMAP coordinates as production features.
- [ ] **[DECISION]** Do not use PCA in production unless a later group-held-out experiment proves it helps.

## G. Audit train/test drift

- [ ] **[CODE]** Compare train/test channel quantiles.
- [ ] **[CODE]** Calculate standardized mean and median shifts.
- [ ] **[CODE]** Identify test values outside training ranges.
- [ ] **[CODE]** Compare per-UAV summary distributions.
- [ ] **[CODE]** Repeat comparisons within matched age bands.
- [ ] **[CODE]** Compare final observed prefixes at similar ages.
- [ ] **[CODE]** Train a group-aware classifier to distinguish train UAVs from test UAVs.
- [ ] **[CODE]** Rank features contributing most to train/test separation.
- [ ] **[REVIEW]** Inspect the largest drift channels and affected ages.
- [ ] **[DECISION]** Use test data only for covariate-shift diagnosis, not for fitting preprocessing.
- [ ] **[DECISION]** Document how drift changes validation scenarios or inference warnings.

## H. Freeze the validation contract

- [ ] **[CODE]** Create five outer folds with non-overlapping UAV IDs.
- [ ] **[CODE]** Approximately balance folds by terminal-lifetime quantile.
- [ ] **[CODE]** Create separate inner UAV-group folds for model selection.
- [ ] **[CODE]** Create 20 locked test-like cutoff scenarios from test history lengths.
- [ ] **[CODE]** Create separate development cutoff scenarios.
- [ ] **[CODE]** Assert no UAV occurs in both training and validation.
- [ ] **[CODE]** Assert preprocessing is fitted only on training UAVs.
- [ ] **[CODE]** Assert changing rows after cutoff `t` cannot change features at `t`.
- [ ] **[CODE]** Calculate scenario-level `R^2` from joined held-out predictions.
- [ ] **[CODE]** Add age- and lifetime-bucket metrics.
- [ ] **[CODE]** Add UAV-level bootstrap confidence intervals.
- [ ] **[DECISION]** Freeze folds and locked scenarios before serious tuning.
- [ ] **[DECISION]** Use nested group-aware validation for hyperparameter selection.
- [ ] **[DECISION]** Do not tune against the public leaderboard.

## I. Manual UAV review template

Complete this for each representative or suspicious UAV:

- [ ] UAV ID and history length recorded.
- [ ] Lifetime category recorded: short, medium, or long.
- [ ] Abrupt jumps noted.
- [ ] Flat or saturated channels noted.
- [ ] Regime changes noted.
- [ ] Degradation-like trends noted.
- [ ] Unusual early-life values noted.
- [ ] Similar behavior in other UAVs noted.
- [ ] Final interpretation recorded.
- [ ] Action recorded: keep, flag, sensitivity test, or confirmed correction.

## J. Phase 0 completion gate

Do not begin serious model selection until:

- [ ] Structural checks are automated and passing.
- [ ] Data fingerprints and schema are saved.
- [ ] Low-variance and scaling policies are frozen.
- [ ] Every telemetry channel has an initial classification.
- [ ] Suspicious UAVs have been manually reviewed.
- [ ] No data has been removed without validation evidence.
- [ ] Train/test drift is quantified at matched ages.
- [ ] Outer and inner UAV-group folds are frozen.
- [ ] Locked and development cutoff scenarios are separate.
- [ ] Feature-causality and leakage tests pass.
- [ ] Reports contain conclusions, not only plots.
- [ ] The first baseline consumes the frozen Phase 0 artifacts.

## Optional - defer unless a finding justifies it

- [ ] DBSCAN clustering.
- [ ] One-Class SVM anomaly detection.
- [ ] ICA or kernel PCA.
- [ ] t-SNE or UMAP visualization.
- [ ] Autoencoder anomaly detection.
- [ ] Advanced imputation experiments.
- [ ] Automatic removal of anomalous cycles or UAVs.
