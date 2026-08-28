# Phase 1: Dataset Construction

## Structural data audit

Quick and easy. Just checking if the data has:

- The expected columns and data types
- Missing and/or infinite values
- Duplicate rows and UAV/cycle keys
- Train/test UAV separation
- Consecutive histories starting at cycle 1
- Constant RUL + flight_cycle within each training UAV
- Training histories ending at RUL = 0
- File hashes and dataset dimensions

**Results:**
- All good. No structural data issues were found.

## UAV-grouped validation folds

The 100 training UAVs are divided into non-overlapping subsets called folds. This allows the model-selection procedure to be evaluated on several different train/validation partitions instead of relying on one potentially favourable 80/20 split.

A single grouped 80/20 split would be suitable for quick experimentation or if the complete modelling pipeline were fixed beforehand. However, with only 100 training UAVs and substantial UAV-to-UAV differences, grouped nested cross-validation provides a more reliable assessment. Its main disadvantage is increased computation time.

All rows and history prefixes belonging to the same `uav_id` remain in the same fold. This prevents information from one UAV appearing in both training and validation.

```text
All 100 training UAVs
│
├── Outer validation: 20 UAVs
│   └── Used only to evaluate the selected configuration
│
└── Outer training: 80 UAVs
    │
    └── Divided into four inner folds of 20 UAVs
        ├── Inner round 1: 60 train, 20 validate
        ├── Inner round 2: 60 train, 20 validate
        ├── Inner round 3: 60 train, 20 validate
        └── Inner round 4: 60 train, 20 validate
```

Each outer round therefore contains two validation levels:

- **Inner validation:** Selects the features, preprocessing, model type, and hyperparameters.
- **Outer validation:** Evaluates the selected procedure on 20 UAVs that were not used during model selection.

After inner validation selects a configuration, that configuration is retrained using all 80 outer-training UAVs. It is then evaluated on the 20 outer-validation UAVs.

The outer folds rotate as follows:

| Outer round | Outer-training UAVs | Outer-validation UAVs |
| ---: | --- | --- |
| 1 | Folds 2–5: 80 UAVs | Fold 1: 20 UAVs |
| 2 | Folds 1 and 3–5: 80 UAVs | Fold 2: 20 UAVs |
| 3 | Folds 1–2 and 4–5: 80 UAVs | Fold 3: 20 UAVs |
| 4 | Folds 1–3 and 5: 80 UAVs | Fold 4: 20 UAVs |
| 5 | Folds 1–4: 80 UAVs | Fold 5: 20 UAVs |

Thus, there are:

```text
5 outer rounds
× 4 inner rounds per outer round
= 20 inner train/validation rounds
```

Every training UAV is used for outer validation exactly once. The five outer results are combined to estimate how well the complete model-selection procedure generalizes to unseen UAVs.

The folds are also approximately balanced by terminal lifetime, ensuring that every fold contains a mixture of short-, medium-, and long-lived UAVs. The separate test dataset is not used during cross-validation.


## Test-like validation scenarios (Cut-Off definition, validation)

The model must learn to predict from incomplete histories, while long-lived UAVs must not dominate merely because they provide more rows. Therefore we need to truncate the dataset artificially.

To define where we actually cut off the UAV flight cycles, we take a look at the test dataset. As this dataset is cut off already, we can infer how the train dataset needs to be cut off for training.

**WARNING** IDK if this is the best practice here

The idea here is to first check how long each flight cycle is in the training dataset and then randomly attaching/applying the cut-off number from the test dataset onto one of the eligible UAVs (that has a long enough flight cycle number).

By doing this we get a training dataset that actually makes learning the RUL possible.

Note: The dataset is not yet cut-off, it is only defined where to cut it off.

## Training prefixes (prefix = cut-off flight cycle (left side))

A similar process is applied in this step, but instead of assigning one cutoff per UAV for each validation scenario, 20 distinct test-like cutoffs are assigned to every training UAV. The cutoffs are randomly sampled from the observed test history lengths and must occur before the UAV’s final cycle. Each cutoff creates a separate training prefix containing only cycles up to and including that cutoff. All 20 prefixes from one UAV receive a weight of 1/20, giving every UAV equal total influence during model training regardless of its history length. The assignments are generated using a fixed random seed to ensure reproducibility.

The sampling probabilities are based on how frequently each flight-cycle cutoff occurs in the test dataset. For example, if cutoff 45 occurs twice among the 100 test UAVs, it initially receives a weight of 2/100, provided that all test cutoffs are eligible for the training UAV. If some test cutoffs exceed the UAV’s lifetime, they are removed and the probabilities of the remaining eligible cutoffs are normalized again. After every selection, the probabilities change because sampling is performed without replacement.

For each uav_id, 20 distinct cutoff values are selected, resulting in 20 different training prefixes. A cutoff value can appear only once for a particular UAV, but the same cutoff may be assigned to other UAVs. Because 20 values are selected, the overall probability that cutoff 45 is included among a UAV’s prefixes is greater than its initial first-draw probability.

## Prefix feature engineering

In this step, the cutoff assignments created in Steps 3 and 4 are applied to the UAV histories. For every (uav_id, cutoff) combination, only telemetry observations from cycle 1 up to and including the cutoff are used. Cycles occurring after the cutoff are treated as unknown future information and are excluded from all calculations. The original datasets are not shortened or modified; the cutoff is applied only while calculating the features.

Each prefix (cut-off flight cycle) is transformed into one row of model features. These features describe the UAV’s age, current telemetry state, early-life baseline, historical behaviour, and recent changes. They include the first and latest values, baseline mean and deviation, whole-prefix mean, standard deviation, minimum, maximum, slope, cycle-to-cycle changes, and statistics over the most recent 5, 20, and 50 cycles. State-related features, such as transition counts and current run length, are additionally calculated for discrete or quantized channels.

```
UAV history: cycles 1–250
Assigned cutoff: cycle 120

Used for feature calculation: cycles 1–120
Ignored as future information: cycles 121–250
Prediction target: RUL at cycle 120
```

The same procedure is applied to:
- Training prefixes from Step 4.
- Development validation scenarios from Step 3.
- Locked validation scenarios from Step 3.
- The complete available history of every test UAV.

All generated model-feature columns begin with feature__. Constant telemetry channels identified during data analysis are excluded. The script also verifies that the generated features contain no missing or infinite values and checks that modifying telemetry after a cutoff cannot change the corresponding prefix features. This ensures that no future information leaks into the model inputs. Scaling and other fold-fitted preprocessing are not applied in this step; they are handled separately after the feature sets have been defined.

### Feature derivation and purpose

For a telemetry channel with prefix values `x_1, ..., x_c`, the following features are calculated independently:

- **Flight-cycle age:** The cutoff `c` and `log(1 + c)` tell the model how much operating time has accumulated and allow nonlinear age effects.
- **First and latest values:** `x_1` provides the initial channel level, while `x_c` tells the model what the sensor reports at prediction time.
- **Baseline mean:** The mean of the first ten available readings provides a less noisy UAV-specific reference level.
- **Baseline deviation:** `x_c - baseline_mean` measures movement from the UAV's own starting level, helping the model distinguish degradation from natural differences between UAVs.
- **History mean and standard deviation:** Describe the channel's typical operating level and variability across the complete prefix.
- **History minimum and maximum:** Capture extreme conditions and the complete range observed before prediction.
- **History slope:** The linear regression slope of telemetry against flight cycle tells the model whether the channel has been increasing or decreasing over the long term.
- **Latest change:** `x_c - x_(c-1)` identifies an abrupt change immediately before prediction.
- **Mean and maximum absolute change:** Summaries of `|x_t - x_(t-1)|` tell the model about typical volatility and the largest observed jump, which may indicate instability or anomalies.
- **Recent 5-, 20-, and 50-cycle features:** Mean, standard deviation, slope, first-to-last change, and latest-value deviation from the window mean separate short-, medium-, and longer-term behaviour. Comparing these time scales helps the model detect recent changes or accelerating degradation that a whole-history summary may hide. If a prefix is shorter than the requested window, the complete prefix is used.
- **State features:** For `telemetry_07` and `telemetry_16`, the number of unique values, transition count, transition rate, current unchanged-run length, and time since the last change tell the model about operating-mode transitions, dwell times, and possible flatlining.

The 22 retained nonconstant telemetry channels produce:

```text
22 channels x 27 standard features = 594
2 state channels x 5 extra features = 10
2 flight-cycle features             = 2
                                      ---
Total                                = 606 features
```


**Without feature engineering, we have two problematic choices:**
- Use only the latest telemetry row, which discards the preceding history.
- Give the model the complete prefix, whose length differs between UAVs and cutoffs.

Feature engineering preserves important information from the prefix while producing the same number of inputs for every sample.

## Feature sets

Step 6 groups the features created in Step 5 into predefined feature sets.

Step 5 creates 606 candidate features, but using all of them immediately could increase computation, redundancy, and overfitting. Step 6 therefore creates a feature catalog that assigns every generated feature to one or more predefined comparison sets. It does not change feature values or select a final model input set.

The catalog records each feature's name, telemetry channel, channel role, statistic, window length, and set membership. This provides explicit feature lists and prevents identifiers, targets, fold labels, or other metadata from being selected accidentally.

- **`age_only` - 2 features:** Contains `flight_cycle` and `log(1 + flight_cycle)`. This is the minimum baseline and shows how much RUL can be predicted from age alone.
- **`last_values` - 24 features:** Contains the two age features and the latest available value from each of the 22 nonconstant telemetry channels. Comparing it with `age_only` tests whether the current telemetry snapshot adds information beyond age.
- **`screened` - 310 features:** Contains the age features, all temporal features from the ten degradation-candidate channels, and seven level/baseline statistics from the four context channels. It keeps rich degradation information while limiting context channels to current and baseline descriptions, reducing noise and dimensionality.
- **`all_nonconstant` - 606 features:** Contains every generated feature from all 22 nonconstant channels. This is the maximum-information reference and checks whether the initial screening excluded useful information.

The `screened` set is based on the Phase 0 channel roles:

- **Degradation candidates:** `telemetry_07`, `telemetry_13`, `telemetry_15`, `telemetry_16`, `telemetry_19`, `telemetry_21`, `telemetry_22`, `telemetry_23`, `telemetry_25`, and `telemetry_28`. All 27 standard features are retained, with five additional state features for `telemetry_07` and `telemetry_16`.
- **Context channels:** `telemetry_01`, `telemetry_06`, `telemetry_18`, and `telemetry_26`. Only the latest value, baseline mean, baseline deviation, history mean, standard deviation, minimum, and maximum are retained because these channels are treated primarily as operating context or UAV baseline information.
- **Weak channels:** Their latest values appear in `last_values`, and all their features appear in `all_nonconstant`, but they are excluded from `screened`.
- **Constant channels:** `telemetry_03`, `telemetry_08`, `telemetry_14`, `telemetry_17`, `telemetry_20`, and `telemetry_27` were already removed and do not appear in any feature set.

The four sets answer different modelling questions:

```text
age_only       -> Is UAV age alone predictive?
last_values    -> Does the current telemetry snapshot help?
screened       -> Do the Phase 0 degradation and context findings help?
all_nonconstant -> Do the excluded candidate features add useful information?
```

Step 6 saves these definitions in `feature_catalog.csv`. The final feature set is not chosen here; the alternatives must be compared using the inner validation folds. Scaling is also not performed here and is handled separately in Step 7.

## Fold-fitted preprocessing

Step 7 puts features with different units and numerical ranges onto a comparable scale. The scaler is fitted separately for every fold and feature set using only the corresponding training UAVs. Validation UAVs never contribute to the fitted preprocessing values, preventing information from the validation data from leaking into model training.

For each feature, the following values are calculated from the training prefixes:

- **Center - median:** The training-fold median is subtracted from every value. A transformed value of zero therefore represents the typical training value.
- **Scale - IQR:** The interquartile range is calculated as `Q75 - Q25` and divided by `1.349`. Dividing the centered feature by this value gives it a standard-deviation-like scale while remaining less sensitive to extreme observations than ordinary mean/standard-deviation scaling.
- **Fallback scale:** A feature-relative tolerance of `1e-12 * max(maximum absolute training value, 1.0)` distinguishes meaningful variation from floating-point noise. If the IQR-based scale is below this tolerance but the feature still has a meaningful observed range, its training-fold standard deviation is used. If the complete range is also below the tolerance, the feature is treated as effectively constant and its scale is set to `1.0`. The output records `standard_deviation_fallback` or `unit_fallback` for every feature where a fallback is used; ordinary IQR scaling is recorded as `iqr`.
- **Transformation:** Every training, validation, or test value is transformed using `scaled_value = (value - training_median) / training_scale`.

This tolerance prevents floating-point noise from being amplified. For example, `feature__telemetry_07__first` has an observed range of zero but a numerically calculated standard deviation of approximately `2.84e-14`. It is now classified as effectively constant and receives `scale = 1.0` instead of using that tiny standard deviation. After median centering, the training-fold feature therefore remains zero and has no influence on the model. It is retained in the fixed feature table for structural consistency but recorded as a `unit_fallback` candidate for later removal.

For the current five outer folds and four feature sets, the recorded scaling decisions are:

```text
IQR scaling                    4,405
Standard-deviation fallback      293
Unit fallback                     12
```

The 12 unit-fallback records represent repeated fold/feature-set occurrences of two unique features: `feature__telemetry_07__first` and `feature__telemetry_07__baseline_mean`.

The procedure is repeated independently for `age_only`, `last_values`, `screened`, and `all_nonconstant`, because each feature set contains different columns. It is also repeated for every outer fold because each outer-training partition contains a different set of 80 UAVs.

This preprocessing helps the model because:

- Features with large numerical ranges cannot dominate only because of their units.
- Median/IQR scaling reduces the influence of skewness and extreme telemetry readings identified during data analysis.
- Fold-specific fitting prevents validation information from influencing the model and keeps the performance estimate realistic.

The script stores the fitted center, final scale, raw IQR, standard deviation, complete data range, numerical tolerance, and `scale_method` for every `(outer_fold, feature_set, feature)` combination in `fold_scaler_parameters.csv.gz`. This makes every fallback auditable. The total use of each scaling method is also recorded in `preprocessing_config.json`. The script does not overwrite the original feature tables. During inner validation, the same scaler must be fitted again using only the 60 inner-training UAVs and then applied unchanged to the 20 inner-validation UAVs. After a configuration has been selected, the outer scaler is fitted on all 80 outer-training UAVs and applied unchanged to the 20 outer-validation UAVs. For final test prediction, a new scaler must be fitted on all 100 training UAVs and applied unchanged to the test UAVs.

## Validation metrics

Step 8 defines how every model is evaluated. It does not train a model or change the features. Instead, it takes a prediction table containing one RUL prediction for each UAV prefix in each validation scenario and compares the predicted RUL (`y_pred`) with the true RUL at that cutoff (`y_true`). Using the same metrics and groups for every candidate makes model comparisons consistent.

The following complementary regression metrics are calculated:

- **R-squared (`R2`):** `1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)`. It measures improvement over always predicting the mean validation RUL. `1` is perfect, `0` is equal to the mean predictor, and a negative value is worse than that baseline.
- **Root mean squared error (`RMSE`):** `sqrt(mean((y_pred - y_true)^2))`. It reports error in flight cycles and gives large errors extra influence, making serious prediction failures visible.
- **Mean absolute error (`MAE`):** `mean(abs(y_pred - y_true))`. It reports the average absolute error in flight cycles and is less dominated by a few large misses than RMSE.
- **Bias:** `mean(y_pred - y_true)`. It shows the direction of systematic error. Positive bias means RUL is generally overestimated; negative bias means it is underestimated; values near zero indicate balanced errors but do not guarantee small errors.

Metrics are reported at several levels:

- **Overall:** Summarizes all evaluated UAV-prefix predictions.
- **By validation scenario:** Checks whether performance changes under different test-like cutoff assignments.
- **By outer fold:** Shows whether performance is consistent across different groups of unseen UAVs.
- **By age band:** Reports results for cutoffs `1-50`, `51-100`, `101-200`, and `>200` cycles, revealing whether prediction quality depends on how much history is available.
- **By lifetime quantile:** Compares UAVs with shorter and longer terminal lifetimes, revealing whether the model performs unevenly across lifetime groups.

Because every UAV appears in multiple validation scenarios, prediction rows from the same UAV are not independent. The uncertainty calculation therefore resamples complete UAV groups rather than individual rows. By default, 1,000 UAV-level bootstrap samples are generated with replacement using a fixed seed. The median bootstrapped `R2` and its 2.5th and 97.5th percentiles provide a reproducible 95% uncertainty interval. A wide interval indicates that the result depends strongly on which UAVs are evaluated.

Before calculating results, the script verifies that all required columns exist, each `(scenario, uav_id)` pair appears only once, and the true and predicted RUL values are finite. It writes:

- `overall_metrics.json` for the overall metrics and bootstrapped `R2` interval.
- `scenario_metrics.csv` for scenario-specific results.
- `fold_metrics.csv` for outer-fold results.
- `age_band_metrics.csv` for cutoff-age results.
- `lifetime_quantile_metrics.csv` for terminal-lifetime-group results.

The metric definitions are stored in `metric_specification.json`. The model-specific result files are created only after a prediction table is supplied to Step 8.

## Cycle-only baseline

Step 9 creates the simplest meaningful RUL model: it uses only the flight-cycle cutoff and ignores all telemetry channels. This establishes how much can be predicted from UAV age alone and provides a minimum benchmark that later telemetry-based models should outperform.

The baseline is not required to train the final model, but it is retained because it provides several important advantages at negligible computational cost:

- **Measures the value of telemetry:** It shows how much RUL can be predicted from flight-cycle age alone. A telemetry model adds useful information only if it clearly improves on this result.
- **Provides context for model metrics:** An RMSE or MAE cannot be judged in isolation; it must be compared with a simple reference produced on the same validation data.
- **Detects pipeline problems:** Failure of a complex model to beat the baseline can indicate weak features, overfitting, leakage-prevention problems, or an implementation error.
- **Justifies feature engineering:** Improvement over the baseline demonstrates that the engineered telemetry features contain information beyond UAV age.
- **Creates a reproducible reference:** Every candidate model can be compared against the same fixed benchmark.

The model is a weighted linear regression:

```text
predicted RUL = max(0, intercept + slope * flight_cycle)
```

- **Input:** `feature__flight_cycle`, which is the final observed cycle of the prefix.
- **Target:** The true RUL at that cutoff.
- **Intercept:** The fitted RUL when flight-cycle age is zero.
- **Slope:** The average change in predicted RUL for each additional flight cycle. A negative slope means predicted RUL decreases as the UAV becomes older.
- **Lower bound:** `max(0, ...)` prevents physically impossible negative RUL predictions.
- **Sample weights:** Each training prefix uses the `1/20` weight assigned in Step 4. Because every UAV has 20 prefixes, each UAV contributes a total weight of one and no UAV dominates the fit through its prefixes.

The baseline is evaluated without UAV leakage. For each outer fold, the regression coefficients are fitted using the training prefixes from the other 80 UAVs and applied to the locked validation prefixes from the 20 held-out UAVs. The five sets of held-out predictions are combined and evaluated using the Step 8 metrics. No telemetry, future cycles, terminal lifetime, or held-out RUL values are used as model inputs.

After validation, a separate baseline is fitted using the prefixes from all 100 training UAVs. Its current test-time equation is:

```text
predicted RUL = max(0, 187.331 - 0.54157 * flight_cycle)
```

The current locked-scenario validation results are:

```text
R2                         -0.005
RMSE                       64.554 cycles
MAE                        51.912 cycles
Bias                       17.712 cycles
95% UAV-bootstrap R2 CI   [-0.153, 0.129]
```

The near-zero negative `R2` means that flight-cycle age alone performs approximately like, and slightly worse than, predicting the mean validation RUL. The positive bias means that it overestimates RUL by about 17.7 cycles on average. These values form the benchmark: a useful telemetry model should achieve lower RMSE and MAE, bias closer to zero, and clearly higher `R2`, preferably with an uncertainty interval supporting the improvement.

Step 9 writes:

- `locked_predictions.csv` containing the held-out predictions used for validation.
- `fold_coefficients.csv` containing the separately fitted intercept and slope for every outer fold.
- `metrics/` containing the overall and grouped Step 8 metric reports.
- `full_training_coefficients.json` containing the model fitted on all training UAVs.
- `test_predictions.csv` containing one cycle-only prediction for every test UAV.

## Automated leakage checks

Step 10 is a final automated safety gate for the validation and feature-engineering pipeline. It does not train or evaluate another model. Instead, it reloads the artifacts from Steps 1-7 and 9 and checks that the fold assignments, cutoffs, features, preprocessing parameters, and baseline predictions still obey the intended leakage-prevention rules. If any assertion fails, the script stops with an error instead of producing a passed report.

The following checks are performed:

- **Outer-fold separation:** Confirms that every outer fold contains exactly 20 unique UAVs and that no UAV appears in more than one outer fold. This prevents rows or prefixes from the same UAV entering both outer training and validation.
- **Inner/outer separation:** Confirms that the 20 outer-validation UAVs are absent from every inner fold belonging to that outer round. This keeps final outer evaluation UAVs out of model and hyperparameter selection.
- **Test-like cutoff reproduction:** Confirms that each of the 5 development and 20 locked scenarios contains 100 unique UAVs and exactly the same multiset of history lengths as the 100 test UAVs.
- **Feature-table dimensions:** Confirms that the training, development-validation, locked-validation, and test feature tables contain the expected number of rows and all 606 generated feature columns.
- **Finite feature values:** Reads every feature table in chunks and rejects missing, positive-infinite, or negative-infinite feature values.
- **Safe feature names:** Requires every model input to begin with `feature__` and rejects feature names containing target- or future-related terms such as `RUL`, `target`, `terminal`, `lifetime`, `final`, or `future`.
- **Prefix causality:** Recalculates features for 10 prefixes after replacing all post-cutoff telemetry values with extreme artificial values. The prefix features must remain exactly unchanged, demonstrating that the extractor only uses cycles up to the cutoff.
- **Fold-fitted preprocessing:** Recalculates the median, scale, fallback method, and numerical tolerance separately for every outer-training fold and feature set. These values must match the saved Step 7 parameters, unit fallbacks must represent numerically constant features, and transforming held-out rows must produce only finite values.
- **Held-out baseline predictions:** Confirms that the Step 9 baseline has exactly one prediction for every `(scenario, uav_id)` pair and that every prediction is associated with the UAV's correct held-out outer fold.

The current verification result is:

```text
Status                         passed
Outer folds                    5 x 20 UAVs
Development scenarios          5
Locked scenarios               20
Training feature rows          2,000
Development validation rows      500
Locked validation rows         2,000
Test feature rows                100
Generated features               606
```

The complete results are stored in `verification_report.json`. This script should be rerun whenever folds, cutoff generation, feature engineering, preprocessing, or baseline prediction code changes. A passed report confirms that the encoded structural and leakage assertions hold; it does not by itself prove that the selected features or model will predict RUL accurately.

## Run 5 feature experiment profile

Run 5 does not copy Phase 1 into separate `phase_1_1`, `phase_1_2`, and similar
implementations. One implementation reads `phase_1_settings.toml` and writes
versioned artifacts under `1_dataset_construction/runs/run_5/`. Existing
canonical Phase 1 artifacts remain the immutable legacy profile used by Runs
3 and 4.

Steps 1-3 and 8 remain shared because feature experiments do not alter the
raw-data audit, UAV folds, validation scenarios, or metrics. Step 9 is repeated
inside each variant directory so its verified inputs and output stay with that
variant, although the age-only baseline is expected to be identical. Prefix
policies receive separate variant directories because they change training
rows. Feature recipes using the same prefix rows share one superset feature
table and are declared as membership columns in one feature catalog.

The Run 5 feature sets are:

| Feature set | Count | Purpose |
| --- | ---: | --- |
| `screened_v1` | 310 | Exact control matching the existing `screened` set |
| `screened_robust` | 558 | Control plus robust baseline, distribution, and recent-window features |
| `screened_acceleration` | 400 | Control plus explicit changes between 5-, 20-, and 50-cycle trends |
| `screened_compact` | 256 | Control representation using a reduced set of highly redundant degradation channels |
| `all_generated_v2` | 1,288 | Every legacy and Run 5 generated feature, retained as a maximum-information reference |

Robust features include baseline median, MAD and IQR; history quantiles and
IQR; median absolute changes; and safely normalized deviations from the UAV's
own baseline. Acceleration features compare recent means, slopes, and
variability across the fixed windows. Every value is calculated only from
cycles at or before the prefix cutoff.

Two training-prefix policies are generated:

- `current20`: the existing 20 empirical test-like cutoffs per UAV;
- `prefix40_stratified`: up to 40 distinct eligible cutoffs per UAV, allocated
  across test cutoff-age bands and reweighted so every UAV still contributes a
  total training weight of one.

The profile is executed with:

```powershell
py 1_dataset_construction\run_all.py --profile run5 --run-number 5
```

Every completed variant also writes `phase_2_interface.json`. The interface is
the authoritative handoff record for Phase 2 and contains the observed prefix
count or bounds, generated-feature count, exact feature-set counts, training
row count, and portable paths to all required Phase 1 artifacts. The top-level
`phase_1_run_manifest.json` links both interfaces. Existing verified artifacts
can receive a refreshed interface without rebuilding the feature tables:

```powershell
py 1_dataset_construction\run_all.py --profile run5 --run-number 5 --refresh-interface
```

Step 7 discovers feature-set membership columns from the catalog instead of a
hard-coded tuple. Step 10 verifies each generated variant, including prefix
causality, finite values, catalog membership, equal total UAV weight, and
fold-fitted preprocessing.



