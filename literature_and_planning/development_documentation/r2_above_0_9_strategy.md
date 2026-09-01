# Strategy for Reaching Public R2 Above 0.9

Last updated: 2026-09-01

## Executive Decision

Reaching `R2 > 0.9` is plausible, but the current pipeline is unlikely to get
there through more tree hyperparameter search, another target cap, or stronger
conservative calibration. The best public score is `0.86741`; at unchanged
target variance, reaching `0.9` requires approximately a **13.2% reduction in
RMSE**. That is a representation-level improvement, not a normal tuning gain.

The recommended path is:

1. Train compact temporal models on **dense, causal sliding windows** while
   preserving UAV-grouped validation and equal total weight per UAV.
2. Stack the best genuinely complementary temporal model with the current tree
   blend using strictly out-of-fold predictions.
3. In parallel, run a small domain-robustness ablation and an individualized
   degradation-onset experiment.
4. Freeze safety calibration during model discovery; restore and compare
   `q = 0.50` and `q = 0.55` only after the accuracy model is selected.

No experiment guarantees the leaderboard threshold. The program below is
designed to reject weak ideas cheaply and reserve locked/Kaggle evaluation for
changes capable of closing the measured gap.

## What the Current Evidence Says

Phase 3 Run 6 uses the calibrated 50/50 XGBoost-ExtraTrees blend,
`screened_drift_pruned`, cap 125, and `q = 0.55`. It reached:

| Evaluation | R2 | RMSE | Meaning |
| --- | ---: | ---: | --- |
| Mean development fold | 0.89274 | 10.6332 | Current offline estimate |
| Pooled selected-candidate OOF | 0.89736 | 10.6685 | 500 development endpoints |
| Kaggle public | 0.86741 | Not exposed | Current best submission |

At fixed variance, the target public score needs the equivalent of reducing
RMSE to `86.85%` of its present value. Applying that ratio to the development
RMSE gives a directional target near `9.23`.

The residuals point to a specific missing capability:

| Region | OOF result | Interpretation |
| --- | --- | --- |
| True RUL 0-25 | RMSE 3.68 | Near-failure prediction is already strong |
| True RUL 51-75 | RMSE 11.09 | Degradation state is ambiguous |
| True RUL 76-100 | RMSE 11.16 | Degradation state is ambiguous |
| True RUL 101-125 | RMSE 12.11 | Global cap/onset model is too coarse |
| Cutoff at most 100 | R2 0.184, RMSE 12.49 | Short histories are the hardest |
| Cutoff 201-300 | R2 0.952, RMSE 8.21 | Long histories are modeled well |

This is not evidence that the cap should be removed. PE_5 directly tested that
idea on Kaggle: cap 125 scored `0.85609`, versus `0.83609` for a soft tail,
`0.83482` for weighted raw RUL, and `0.82866` for raw RUL. Keep cap 125 as the
control.

## New Domain-Shift Finding

Every development scenario exactly matches the test cutoff distribution. Yet a
cross-validated ExtraTrees classifier can distinguish development endpoints
from the unlabeled Kaggle test endpoints with mean AUC `0.834` using the 298
current features. Cutoff alone is effectively chance. Stable shifts include:

- telemetry 16 state cardinality;
- telemetry 15 and 16 window-20 standard deviation; and
- telemetry 15 and 16 slope features.

This confirms observable covariate shift. It does **not** justify immediately
reweighting training: the most test-like development quintile is locally the
easiest (`RMSE = 8.37`), and domain propensity correlates negatively with
absolute error (`r = -0.184`). The next domain experiment should therefore
compare robust feature sets and clipped weighting under grouped validation,
not silently adapt the production model.

Covariate-shift correction is statistically valid only under assumptions such
as stable `P(RUL | features)` and reliable density ratios; importance-weighted
cross-validation addresses that setting, but those assumptions remain unproven
here ([Sugiyama, Krauledat, and Mueller, 2007](https://jmlr.org/papers/v8/sugiyama07a.html)).

## Experiment 1: Dense Temporal Learning

This is the highest-priority experiment. The existing LSTM, TCN, multiscale
CNN, and graph-TCN results do not settle whether sequence learning helps: they
were trained from only 20 prefixes per UAV. Successful C-MAPSS sequence methods
typically prepare dense time windows from normalized sensor histories. Li,
Ding, and Sun explicitly use time-window sample preparation for a deep CNN
([RESS, 2018](https://www.sciencedirect.com/science/article/pii/S0951832017307779));
multiscale LSTM work likewise begins with sliding-window samples
([Alexandria Engineering Journal, 2021](https://www.sciencedirect.com/science/article/pii/S187775032100171X)).

Use one new pipeline experiment with these fixed controls:

| Setting | Required value |
| --- | --- |
| Target | Cap 125 |
| Validation scenarios | Current winning early/middle protocol |
| Folds | Existing UAV-grouped folds; never split windows from one UAV across folds |
| Window endpoints | Every cycle or stride 2; no sparse `current20` sampling |
| UAV weighting | Sum of weights equals one per UAV |
| Window lengths | 20, 30, and 50; causal left padding plus a mask when needed |
| Channels | Dynamic telemetry channels only, fold-fitted scaling |
| Calibration | None or fixed `q = 0.50` during architecture comparison |
| Models | Compact multiscale 1D CNN, TCN, and GRU/LSTM control |
| Seeds | At least 3 for the selected configurations |

Do not launch a 50-candidate search for every neural family. First run a small
screen of 8-12 configurations per model. Promote a temporal model only if it
meets all of these development gates:

- mean grouped `R2 >= 0.89`;
- mean grouped `RMSE <= 10.7`;
- no fold worse than the tree control by more than `1.0` RMSE;
- residual correlation with the current tree blend below `0.90`; and
- stable results across three seeds.

The residual-correlation gate matters. A model with the same score and the same
errors cannot improve an ensemble.

## Experiment 2: Leakage-Free OOF Stacking

Run this only after Experiment 1 produces at least one complementary temporal
model. Build UAV-grouped out-of-fold predictions for:

- the current XGBoost-ExtraTrees blend;
- the best dense temporal model; and
- optionally one learned health-index model if it passes its own gate.

Compare a convex blend, nonnegative ridge regression, and a shallow XGBoost
meta-model. Meta-model fitting must be nested: an endpoint's base and meta
prediction may never come from a model trained on that endpoint's UAV. Optimize
ordinary RMSE/R2 first and report safety separately.

A very recent preprint reports FD003 `R2 = 0.906` and `RMSE = 8.613` from an
XGBoost stack of LSTM/CNN-family OOF predictions
([Hossain et al., arXiv:2608.27940](https://arxiv.org/abs/2608.27940)). This is
evidence that the target is technically plausible, not proof that the same
score will transfer here: the paper is unreviewed, four days old at this
writing, and its evaluation protocol may differ from the competition.

Promotion gate: stacking must improve mean grouped RMSE by at least `3%`, win
at least four of five folds, and preserve the gain across three retraining
seeds. Otherwise retain the simpler model.

## Experiment 3: Individualized Degradation Onset

The fixed cap assumes that every UAV enters measurable degradation at the same
RUL threshold. The middle-RUL residuals suggest that assumption is too coarse.
Test a fold-fitted change-point detector that estimates each training UAV's
degradation onset from temporal correlation or a monotonic health index. Use
the estimated onset to define a personalized piecewise target, then compare it
against cap 125 with the model and folds fixed.

Related C-MAPSS work reports 5.6% and 7.5% accuracy improvements from
device-level change points
([Arunan et al., Control Engineering Practice](https://doi.org/10.1016/j.conengprac.2023.105840)). Its
strongest evidence comes from multi-condition subsets, so this is second-line
rather than the first experiment for the single-condition/two-fault-mode data.

Required safeguards:

- fit detector thresholds using training UAVs only inside each fold;
- never use terminal lifetime when detecting onset at inference time;
- test the target transformation independently from the model architecture;
- reject it unless gains occur specifically in RUL 51-125 without harming
  RUL 0-50.

## Experiment 4: Domain Robustness

Run four development-only cells with the current tree blend:

1. current `screened_drift_pruned` control;
2. remove the consistently shifted telemetry 15/16 state, variance, and slope
   features;
3. use robust rank/quantile transforms fitted inside each fold; and
4. clipped density-ratio weights, only after repeated estimation confirms
   adequate source/test overlap.

Evaluate both ordinary grouped metrics and test-propensity-stratified metrics.
Require improvement in at least four folds and no concentration of errors in
the high-propensity rows. If all simple cells fail, a self-supervised domain
adaptation encoder is scientifically justified; such methods explicitly use
unlabeled target telemetry to learn domain-invariant RUL representations
([Le Xuan, Munderloh, and Ostermann, RESS 2024](https://www.sciencedirect.com/science/article/pii/S0951832024003685)).
Confirm that the competition permits transductive use of unlabeled test inputs
before using test-distribution information for feature selection, weighting,
or representation learning. If it does not, retain the domain classifier only
as a diagnostic and construct all robustness choices from training folds.

## Lower-Priority Ideas

| Idea | Decision |
| --- | --- |
| More XGBoost/ExtraTrees candidates | Stop; repeated tree searches have plateaued and cannot plausibly supply 13% RMSE reduction alone. |
| Different cap or uncapped target | Stop for now; PE_5 leaderboard evidence favors cap 125. |
| Stronger `q` calibration | Safety tool, not the route to 0.9; higher `q` deliberately trades R2 for fewer overpredictions. |
| Monotonic tree constraints | Test only on a learned health index or features with stable physical direction. XGBoost supports named constraints, but `hist` may need larger `max_bin` ([official documentation](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html)). |
| Raw multichannel DTW | Do not repeat; the existing trajectory model was poor. A learned monotonic health-index trajectory is a distinct later experiment. |
| GAN augmentation | Defer; there is no evidence that synthetic samples solve the observed short-history state ambiguity or domain gap. |
| Large transformer/foundation model | Defer until a compact dense-window model proves that temporal representation helps. |

## Safety Policy

Accuracy discovery and conservative calibration should be separated. The
current `q = 0.55` policy is defensible for safety, but local prototypes showed
that `q = 0.50` slightly improves R2 while `q = 0.55` reduces overprediction at
a small accuracy cost. During Experiments 1-4, hold calibration at `q = 0.50`
or none. After freezing the predictive model, compare exactly two final
policies:

- `q = 0.50`: accuracy-oriented submission;
- `q = 0.55`: conservative submission.

Choose the operational policy using grouped R2/RMSE plus overprediction rate,
RMS overprediction, 95th-percentile overprediction, and the same metrics in the
true-RUL 0-25 and 26-50 bands. Do not tune the quantile against repeated Kaggle
submissions.

## Recommended Order and Stop Rules

| Order | Work | Approximate decision |
| ---: | --- | --- |
| 1 | Dense temporal screen | Stop neural work if no model reaches the stated local gates. |
| 2 | OOF stack | Run only with a complementary temporal winner. |
| 3 | Change-point target | Promote only with consistent gains in RUL 51-125. |
| 4 | Domain robustness | Prefer simple robust features; use adaptation only if justified. |
| 5 | Locked evaluation | Open once for the frozen winner, not once per idea. |
| 6 | Phase 3 and Kaggle | Produce accuracy and conservative policies from the same frozen model. |

The most credible route is therefore **dense temporal windows -> complementary
OOF stack -> one frozen locked confirmation**. If that sequence does not move
grouped development to at least approximately `R2 >= 0.91` and `RMSE <= 9.5`,
do not spend the locked evaluation. The directional target implied by the
public gap is closer to `RMSE = 9.23`; the slightly looser gate acknowledges
sampling noise but still demands a material improvement. If the sequence does
not reach that region, a public score above 0.9 is unlikely without changing
the validation/data assumptions or accepting heavy leaderboard overfitting.

## Experimental Boundary

The local data structurally match NASA C-MAPSS FD003. NASA provides complete
run-to-failure training histories and truncated test histories
([official dataset page](https://data.nasa.gov/dataset/groups/cmapss-jet-engine-simulated-data)).
The original benchmark also has test RUL labels. Do not download, read, join,
or use those labels unless the competition rules explicitly authorize them;
doing so would invalidate the experiment and the leaderboard result.
