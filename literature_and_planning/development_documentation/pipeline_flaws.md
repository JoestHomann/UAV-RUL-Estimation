# Diagnosed Pipeline Flaws and Improvement Plan

## Status and purpose

This document records the pipeline diagnosis performed after the first three
Phase 3 submissions did not reproduce the performance suggested by development
validation. It separates measured evidence from working hypotheses and defines
the experiments required before changing the main Phase 1 to Phase 3 protocol.

The immediate objective is to determine whether a leaderboard `R2` of at least
`0.9` is attainable without using test targets or adapting decisions to repeated
leaderboard feedback. Predictions should also be conservative: when errors are
unavoidable, underestimating RUL is preferable to overestimating it. The current
evidence makes the score objective plausible, but does not yet demonstrate it on
the hidden Kaggle targets or establish the best safety-performance tradeoff.

## Observed leaderboard and validation gap

The first three Kaggle submissions produced the following public scores:

| Submission | Public `R2` |
| --- | ---: |
| Run 1 | 0.59695 |
| Run 2 | 0.48924 |
| Run 3 | 0.54136 |

Run 3 selected XGBoost candidate 43 with the `screened_drift_pruned` feature
set. Its saved development results reported mean-fold `RMSE = 28.5019`,
mean-fold `R2 = 0.78554`, `MAE = 18.8048`, and bias `= +5.192`. Its test
predictions had mean `87.27`, median `83.34`, and range `19.92` to `216.09`.

The lower public score is therefore not explained by a lack of improvement in
the saved development metric alone. It indicates that the development task and
the hidden test task are not sufficiently aligned.

## Dataset identification and boundary

The local data have the structural fingerprints of NASA C-MAPSS `FD003`:

- 24,720 training rows and 16,596 test rows;
- 100 training units and 100 test units;
- maximum training lifetime of 525 cycles;
- one operating condition and two fault modes.

NASA describes C-MAPSS training trajectories as complete run-to-failure
histories and test trajectories as histories truncated before failure. The task
is to estimate the remaining cycles for each truncated test trajectory. See the
[NASA C-MAPSS dataset description](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).

This identification is useful for understanding the data-generating process,
but it creates an important experimental boundary. The original C-MAPSS release
also distributes the true test RUL values. Those values must not be downloaded,
read, joined, or used for development unless the competition rules explicitly
permit that external ground truth. Using them would disclose the prediction
target and invalidate the experiment.

## Flaw 1: cutoff matching does not make validation test-like

The current scenario construction in
[`create_test_like_scenarios.py`](../../1_dataset_construction/3_test_like_validation_scenarios/create_test_like_scenarios.py)
reproduces the exact marginal distribution of test history lengths. For each
test cutoff, it randomly chooses an unused training UAV whose lifetime is long
enough to support that cutoff.

This procedure preserves the cutoff distribution, but it does not preserve or
model the joint relationship between:

- observed history length;
- degradation state at the endpoint;
- total lifetime; and
- remaining useful life.

Consequently, the development scenarios contain endpoints with true RUL as high
as 435 cycles. At such endpoints the UAV can still be in an approximately
healthy operating regime. The model is nevertheless scored on its ability to
predict the exact distant failure time, even when the observed prefix may not
contain enough information to distinguish 150 remaining cycles from 400.

The existing XGBoost diagnostics support this interpretation:

| Cutoff band | RMSE | `R2` |
| --- | ---: | ---: |
| 1-50 | 49.85 | -1.82 |
| 51-100 | 49.35 | 0.358 |
| 101-200 | 23.95 | 0.828 |
| Above 200 | 15.66 | 0.915 |

The weakest performance is concentrated at short observed histories, where an
exact long-horizon RUL target is least identifiable.

## Flaw 2: the raw target assumes exact early-life predictability

The XGBoost adapter currently obtains the unmodified target through
`target_values(training_data)` in
[`xgboost.py`](../../2_model_architecture_study/4_model_adapters/models/tabular/xgboost.py).
No piecewise target transformation is available in the experiment contract.

A common C-MAPSS formulation treats early healthy life as a constant-RUL
plateau and predicts the subsequent degradation region. A representative
implementation caps training RUL at 125 cycles; see this
[C-MAPSS methodology example](https://pmc.ncbi.nlm.nih.gov/articles/PMC12986558/).
The cap is not universal ground truth and must be tested as a modelling choice.
Its purpose is to avoid requiring the model to infer precise distant failure
times from telemetry that may not yet express degradation.

The required transformation is:

```text
model_target = min(raw_rul, maximum_rul)
```

Candidate values such as `110`, `125`, `140`, and `150` must be compared using
only grouped development data. Whether final predictions should also be clipped
must be a separate, predeclared setting rather than an automatic consequence of
training with a capped target.

## Required safety objective: conservative RUL predictions

For this project, an overprediction occurs when `y_pred > y_true`. It tells the
operator that more useful life remains than is actually available. An
underprediction can cause maintenance to happen earlier than necessary, but an
overprediction can allow operation beyond the estimated safe life. The model
selection procedure should therefore prefer underprediction when two candidates
have similar predictive performance.

Conservatism must not be inferred from mean bias alone. Positive and negative
errors can cancel, so a model with bias near zero can still make a small number
of severe overpredictions. Every experiment should report at least:

- overprediction rate: `mean(y_pred > y_true)`;
- mean overprediction: `mean(max(y_pred - y_true, 0))`;
- root mean squared overprediction;
- 90th, 95th, and maximum positive residual;
- the same quantities within critical true-RUL bands, especially `0-25`,
  `26-50`, `51-100`, and above `100` cycles;
- underprediction rate and magnitude, to expose the operational cost of being
  excessively conservative; and
- ordinary `R2`, RMSE, MAE, and bias, so safety is not improved by making
  predictions uniformly uninformative.

Several training and calibration methods should be compared.

### Asymmetric regression loss

Use a larger loss weight when the model overpredicts RUL:

```text
error = y_pred - y_true

loss = overprediction_weight * error^2  if error > 0
       underprediction_weight * error^2 otherwise
```

With `underprediction_weight = 1`, test overprediction weights such as `1.0`,
`1.5`, `2.0`, and `3.0`. A weight of `1.0` is the symmetric control. This method
directly changes model fitting and can be implemented as a custom objective for
models that support one, or as an asymmetric batch loss for neural models.

### Lower-quantile regression

Train the conditional lower quantile of RUL instead of its conditional mean.
Pinball loss with quantile `tau < 0.5` penalizes overprediction more strongly
than underprediction. Candidate quantiles should include `0.50` as the control
and conservative alternatives such as `0.45`, `0.40`, `0.35`, and `0.30`.

Quantile regression is attractive because its conservatism has a clear
statistical meaning. However, a nominal quantile is not automatically calibrated
on unseen UAVs, so empirical grouped coverage must still be measured.

### Development-calibrated safety adjustment

A model-independent alternative is to subtract a fixed safety offset from raw
predictions. The offset must be estimated exclusively from out-of-fold
development residuals. A one-sided grouped conformal variant can instead return
a calibrated lower prediction bound for a selected coverage level.

Test fixed or calibrated policies that target several empirical non-overprediction
rates, for example 60%, 70%, 80%, and 90%. Calibration must use predictions from
UAVs that were not used to fit the corresponding model. A safety offset must
never be chosen from test predictions or a leaderboard result.

These approaches solve related but different problems. Target capping defines
the early-life learning target; asymmetric or quantile loss changes the fitted
conditional estimate; and a calibrated offset changes the final decision
policy. They must be configurable and evaluated separately before combinations
are tested.

## Flaw 3: architecture search is optimizing the wrong task

Additional architectures and larger candidate budgets cannot repair a target or
validation mismatch. They optimize performance under the task they receive. A
model can therefore become better at the current development objective while
becoming no better, or even worse, on the hidden test objective.

The Run 3 development-to-leaderboard gap is direct evidence of this risk. The
next expensive architecture study should not reuse the current scenario and
target formulation unchanged.

## Flaw 4: sequence training underuses the available endpoints

The current `current20` profile creates 20 training prefixes per UAV, or about
2,000 prefix examples in total. The raw training set contains 24,720 causal
cycle endpoints. Although adjacent endpoints are correlated and must never be
split across folds independently, they can still provide substantially more
training signal when all prefixes from a UAV remain in the same fold.

Sparse prefix sampling is particularly restrictive for neural sequence models.
The existing architecture comparison may therefore underestimate sequence
models because they were trained with too few examples rather than because the
architectures are intrinsically unsuitable.

Dense-prefix experiments must preserve equal total UAV influence. Otherwise,
long-lived UAVs would receive more aggregate training weight solely because
they contribute more endpoints.

## Exploratory capped-target evidence

Exploratory in-memory experiments were run using the existing grouped folds and
causal feature builder. These were diagnostic experiments, not registered
Phase 2 or Phase 3 runs, and no result in this section should be treated as a
locked estimate.

Five alternative scenarios were constructed by reproducing all 100 observed
test cutoff values while using deterministic one-to-one assignment between
cutoffs and training UAVs. Assignment was constrained to endpoints with
`1 <= raw_rul <= 125`. A perfect assignment existed for every scenario.

| Diagnostic experiment | Pooled `R2` | RMSE | Bias |
| --- | ---: | ---: | ---: |
| Uncapped XGBoost on cap-constrained scenarios | 0.50870 | 22.925 | +12.31 |
| RUL cap 125 with the same base configuration | 0.88731 | 10.979 | -0.590 |
| Cap-specific 20-candidate XGBoost search | **0.89473** | **10.612** | **-0.329** |

The best cap-specific candidate produced fold `R2` values of `0.8820`, `0.8916`,
`0.8790`, `0.8751`, and `0.9350`. Its parameters were:

```toml
learning_rate = 0.011947083763386997
max_depth = 8
min_child_weight = 0.7455233132886098
reg_alpha = 0.00012503596823176485
reg_lambda = 0.003193048036905057
subsample = 0.6928139456281494
colsample_bytree = 0.5261715964760796
```

On the existing development scenarios, restricting evaluation post hoc to rows
with `raw_rul <= 150` gave `R2 = 0.90253` and `RMSE = 12.44` for the cap-125
model. This is supporting evidence only. A post-hoc subset is not an acceptable
headline metric because the subset was chosen after observing results.

The diagnostic gain is large enough to justify a controlled experiment, but it
does not prove that the hidden test targets follow the cap-constrained scenario
distribution. Constraining scenario RUL and then measuring a capped model on
those scenarios partially embeds the modelling assumption into validation.
Multiple caps, an unchanged control, and a locked selection protocol are needed
to estimate whether the gain is robust.

Simple seed and parameter averaging did not improve the best capped XGBoost
result. More ensembling should therefore not be the first priority.

## Required controlled experiments

### Phase 1: validation and prefix construction

Add an experimental validation profile without replacing the current baseline:

1. Preserve the exact empirical test cutoff multiset in every scenario.
2. Assign cutoffs to training UAVs through deterministic bipartite matching.
3. Support a declared maximum scenario RUL for sensitivity experiments.
4. Generate separate profiles for several caps and retain an uncapped control.
5. Report target distributions by scenario, including minimum, quartiles,
   maximum, standard deviation, and counts by RUL band.
6. Add dense or fixed-stride causal prefix profiles while keeping UAV-grouped
   folds and equal total weight per UAV.
7. Extend Step 8 with one-sided overprediction and underprediction metrics,
   including grouped reports by fold, scenario, cutoff band, and true-RUL band.

The matching constraint and model-target cap must remain separate settings. The
first controls which validation endpoints exist; the second controls what the
model learns. Keeping them separate makes circular validation assumptions
visible.

### Phase 2: target-aware architecture study

Add a declared target policy to the experiment contract:

```toml
[target]
mode = "piecewise_cap"
maximum_rul = 125
clip_predictions = false

[prediction_policy]
loss = "symmetric_rmse"
overprediction_weight = 1.0
quantile = 0.5
calibration = "none"
non_overprediction_coverage = 0.5
```

The fields above describe the experimental dimensions. Settings that do not
apply to the selected loss or calibration mode should be validated as inactive
rather than silently affecting training.

Run a focused study with at least the following configurations:

- unchanged raw-target control;
- caps at 110, 125, 140, and 150;
- XGBoost as the primary tabular family and ExtraTrees as the independent
  tree-family control;
- CatBoost only in a separate confirmation experiment after the cheaper
  families support a specific pipeline change;
- symmetric RMSE as the conservatism control;
- asymmetric overprediction weights of 1.5, 2.0, and 3.0;
- lower-quantile objectives at selected quantiles between 0.30 and 0.45;
- out-of-fold calibrated offsets or lower bounds at several coverage levels;
- dense-prefix sequence models only after their data adapter is updated;
- raw-RUL metrics for final comparability, plus one-sided error metrics by
  cutoff and RUL band.

The cap and scenario profile must be selected with development folds only.
Locked scenarios must remain closed until one complete policy is selected.
The first experiment should vary one conservatism mechanism at a time using the
best development-only target cap. Only the best individual mechanisms should
advance to a small interaction experiment. This avoids an unnecessarily large
cap-by-loss-by-calibration search.

Candidate selection should use a predeclared Pareto rule. First reject policies
whose `R2` or RMSE degradation exceeds an accepted tolerance relative to the
best symmetric model. Among the remaining candidates, prefer the policy with
the lowest severe-overprediction rate and tail magnitude. Suggested tolerances
to test are an absolute `R2` loss of `0.005` and `0.01`; the final tolerance must
be fixed before locked evaluation.

### Phase 3: frozen target policy

Phase 3 must inherit the selected target policy from Phase 2. The final search,
all-UAV fit, stored model, regenerated predictions, submission verification, and
post-run reporting must use the same policy. Phase 3 must not choose or adjust
the cap after viewing test predictions or a Kaggle score.

The report should include the submission prediction distribution and the number
of predictions at or near any configured clipping boundary. Prediction clipping
must be verified against regenerated frozen-model output, just like every other
inference transformation.

The frozen artifact must also store the loss type, asymmetry weight or quantile,
calibration method, calibration residual summary, selected coverage, and any
safety offset. Submission verification must regenerate both the raw model
prediction and the final policy-adjusted prediction. This prevents a safety
adjustment from being applied only to the CSV and becoming irreproducible.

## Secondary improvement experiments

After the target and validation experiments are complete, the most promising
additional work is:

- dense sliding-window or dense-prefix training for sequence models;
- degradation onset and change-point features;
- health-index features that summarize monotonic degradation;
- unsupervised fault-mode clustering followed by mode-aware models or features;
- residual analysis by inferred fault mode, RUL band, cutoff band, and UAV.

FD003 contains two fault modes, so a single global regression surface may be
averaging distinct degradation paths. This should be investigated only with
training data and grouped validation; test clusters must not be interpreted
using hidden outcomes.

## Non-solutions and non-causes

- Increasing an unchanged architecture candidate budget from 25 to 50 or more
  does not address the validation-target mismatch.
- Adding more model families before correcting the experimental task is unlikely
  to produce a reliable large gain.
- GAN-generated trajectories are not a priority. Synthetic data could amplify
  an incorrect target relationship and would be difficult to validate with only
  100 independent training UAVs.
- The XGBoost CPU-input/CUDA-booster `DMatrix` warning can increase prediction
  time and memory use, but it does not explain the observed score gap.
- The public leaderboard must not be used as an iterative hyperparameter tuning
  set. Repeated adaptation to its score would produce another form of
  overfitting.

## Decision criteria

The capped formulation should replace the raw-target baseline only if a fresh,
predeclared experiment shows all of the following:

- stable improvement across all five grouped outer folds;
- improvement across multiple independently generated validation scenarios;
- materially lower RMSE in short-history and high-uncertainty regions;
- non-positive or otherwise predeclared acceptable bias;
- lower overprediction rate and positive-residual tail magnitude than the
  symmetric control, particularly below 50 true RUL cycles;
- empirical conservative coverage that is stable across folds and close to its
  declared target;
- no collapse in `R2`, RMSE, or any RUL band from excessive underprediction;
- consistent conclusions across reasonable cap values;
- reproducible Phase 2 and Phase 3 artifacts; and
- no use of original or competition test targets.

The current best diagnostic result, pooled `R2 = 0.89473`, is close enough to
the `0.9` objective to justify implementing this experiment before another
ordinary architecture run. It is evidence for a direction, not a promise of a
`0.9` public score.

The conservatism objective introduces a genuine tradeoff with leaderboard `R2`.
The project should therefore seek a model on the measured Pareto frontier, not
the largest possible negative bias. The preferred model is the most conservative
candidate that retains the predeclared level of predictive performance.
