# Pipeline Experiments

Last updated: 2026-08-30

## Purpose

`pipeline_experiments` contains controlled experiments that change assumptions
across Phase 1 and Phase 2. Each named experiment owns its resolved settings,
checkpoints, and results under
`pipeline_experiments/runs/<experiment>/`. The catalog is defined in
[`pipeline_experiments.toml`](../../pipeline_experiments/pipeline_experiments.toml).

The current experiment sets test whether validation/target assumptions explain
the offline-to-Kaggle gap and whether identified telemetry degradation families
add generalizable RUL information beyond UAV age and current sensor values.

## Shared Protocol

The target/scenario matrix keeps the following choices fixed:

- training prefixes: `current20`, with equal total UAV weight;
- tabular features: `screened_drift_pruned`;
- compared families: XGBoost and ExtraTrees;
- search budget: 50 candidates per family and outer fold;
- search seed: 13; retraining seeds: 13, 37, and 73;
- evaluation target: raw RUL;
- prediction policy: symmetric RMSE, without clipping, asymmetry, or offset.

The `current` scenario profile uses eligible random assignment while preserving
the empirical test-cutoff multiset. The `early_and_middle` profile uses
deterministic bipartite assignment with true validation RUL restricted to
1-125 cycles. The `capped_125` fitting target is
`min(raw_rul, 125)`; reported metrics remain calculated against raw RUL.

New cells initially use `phase_2_scope = "selection_only"`. They stop after
development Step 5 and cannot open locked Step 6. Only the development-selected
experiment is later changed to `"complete"` and evaluated in Steps 6-7.

## Experiment Register

| Experiment | Scenario | Fitting target | What is investigated | Status |
| --- | --- | --- | --- | --- |
| `PE_run_1` | Current | Raw | Baseline task and architecture performance | Complete through Step 7 |
| `PE_2x2_current_cap125` | Current | Cap 125 | Target-cap effect without changing validation scenarios | Step 5 complete; rejected |
| `PE_2x2_early_raw` | Early/middle | Raw | Scenario-construction effect without changing the target | Step 5 complete; rejected |
| `PE_2x2_early_cap125` | Early/middle | Cap 125 | Interaction between bounded scenarios and capped fitting | Selected; locked Steps 6-7 complete; Phase 3 Run 7 started |

### PE_run_1: current scenarios, raw target

This is the unchanged control. It uses current validation scenarios and raw RUL
for fitting and evaluation. It originally compared the mean baseline, Random
Forest, ExtraTrees, and XGBoost; XGBoost and ExtraTrees provide the matrix
control values. Its predictions reproduce the corresponding Run 5 results.

The experiment establishes that stronger architecture tuning alone does not
close the validation-to-leaderboard gap and quantifies systematic
overprediction under the original task.

### PE_2x2_current_cap125: current scenarios, capped target

This cell reuses the `PE_run_1` Phase 1 artifacts and changes only the fitting
target from raw RUL to cap 125. It tests whether reducing the requirement for
exact early-life RUL prediction helps under the existing validation
distribution. Prediction clipping is not applied.

### PE_2x2_early_raw: bounded scenarios, raw target

This cell builds the shared `PE_2x2_early_middle` Phase 1 artifacts and retains
the raw fitting target. It isolates the effect of constructing validation
endpoints in the 1-125 RUL region. A gain here would indicate that scenario
alignment, rather than target transformation alone, drives the result.

### PE_2x2_early_cap125: bounded scenarios, capped target

This cell reuses the early/middle Phase 1 artifacts and changes only the fitting
target to cap 125. It tests the cap-by-scenario interaction suggested by the
exploratory capped-target diagnostics. A large gain confined to this cell would
show that the apparent benefit depends on both assumptions and must therefore
be interpreted cautiously.

## Development Results

Step 5 results are the only values used to choose which cell may enter locked
evaluation. Update this table from
`pipeline_experiments/selection_experiment_comparison.csv` after all selections
finish.

| Experiment | Family | Inner RMSE mean | Fold SD | Inner R2 mean | Bias | Overprediction rate | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `PE_run_1` | XGBoost | 29.8059 | 1.7513 | 0.7708 | +5.2165 | 67.35% | Current/raw control |
| `PE_run_1` | ExtraTrees | 32.7578 | 2.6545 | 0.7264 | +4.4906 | 66.00% | Lower safety bias, weaker RMSE |
| `PE_2x2_current_cap125` | XGBoost | 36.2141 | 2.5645 | 0.6690 | -10.8488 | 46.70% | Cap harms the current-scenario task |
| `PE_2x2_current_cap125` | ExtraTrees | 37.1100 | 2.5884 | 0.6525 | -10.7370 | 50.95% | Same failure in the independent family |
| `PE_2x2_early_raw` | XGBoost | 23.3667 | 1.5326 | 0.4867 | +11.4973 | 71.35% | Lower RMSE from narrower targets, but poor R2 and bias |
| `PE_2x2_early_raw` | ExtraTrees | 26.6406 | 2.9202 | 0.2703 | +10.6021 | 67.60% | Rejected for instability and positive bias |
| `PE_2x2_early_cap125` | XGBoost | **11.9746** | 0.9418 | **0.8661** | -0.9007 | 47.30% | Selected development cell |
| `PE_2x2_early_cap125` | ExtraTrees | 12.2693 | 0.8427 | 0.8598 | -1.0864 | 47.15% | Independent confirmation |

The development decision selected `PE_2x2_early_cap125`: it was the only new
cell with high R2, low RMSE, near-zero conservative bias, stable folds, and
agreement across both model families. The matrix also showed that cap 125 is
not a universal improvement: it fails when the validation distribution
contains substantial true RUL above 125. The decision therefore freezes the
specific early/middle-scenario and capped-target policy, not target capping in
isolation.

This rationale was stated and the catalog was frozen before Step 6 was started,
but it was not written into this file until after the locked run completed. That
documentation-order deviation is recorded here explicitly; no locked metric
was used to change the selected cell.

The completed `PE_run_1` locked results were XGBoost `R2 = 0.8041`,
`RMSE = 28.4989`, and overprediction rate `70.95%`; ExtraTrees produced
`R2 = 0.7640`, `RMSE = 31.2779`, and overprediction rate `66.60%`. Pairwise
intervals did not establish a decisive R2 or RMSE difference. XGBoost was the
nominal accuracy winner but overpredicted 97.66% of endpoints with true RUL
0-25, so the original task does not satisfy the conservative-prediction goal.

## Locked Confirmation and Phase 3 Decision

Only `PE_2x2_early_cap125` entered locked evaluation. Across 20 locked
scenarios and three retraining seeds, the results were:

| Family | R2 | 95% UAV-bootstrap interval | RMSE | Bias | Overprediction rate |
| --- | ---: | --- | ---: | ---: | ---: |
| XGBoost | **0.8810** | 0.8577 to 0.9026 | **11.3693** | -0.5590 | 48.92% |
| ExtraTrees | 0.8786 | 0.8511 to 0.9019 | 11.4868 | -0.8031 | 47.78% |

No paired XGBoost-versus-ExtraTrees interval excluded zero. XGBoost remains the
declared Phase 3 family because it has the nominally better R2/RMSE and is much
faster and smaller, not because the locked comparison established statistical
superiority. Phase 3 Run 7 inherits cap 125 and the symmetric prediction policy;
its final report and Kaggle score are pending.

The overall negative bias does not establish safety. For XGBoost at true RUL
0-25, bias was `+2.2731`, overprediction rate was `67.65%`, and RMS
overprediction was `4.9345`. Conservative loss or calibration therefore
remains a separate development-only experiment. The locked R2 mean also remains
below the 0.9 objective, and the upper confidence limit merely reaches 0.9026.

## Decision Rule

1. Complete Step 5 for the three new cells.
2. Compare only development-selection results across all four cells.
3. Reject candidates with unstable fold behavior or materially worse RMSE/R2.
4. Among similarly accurate candidates, prefer lower positive bias,
   overprediction rate, and RMS overprediction.
5. Record the selected experiment and rationale before changing its
   `phase_2_scope` to `"complete"`.
6. Run locked Steps 6-7 once for that experiment; keep all other cells
   selection-only.

This rule was executed with `PE_2x2_early_cap125` as the sole promoted cell.

Locked results test the frozen decision and must not trigger another round of
target or scenario selection. Conservative losses and calibrated offsets are a
subsequent experiment family and must not be mixed into this matrix.

## Signal-Family Ablation

### Question

Which degradation-signal families add repeatable RUL information beyond cycle
age and the current values of all nonconstant telemetry channels?

### Fixed protocol

- group: `PE_signal_family_ablation`;
- Phase 1 run: `PE_signal_family_ablation`, `current20` prefixes;
- models: XGBoost and ExtraTrees;
- search: 25 candidates per family and outer fold, seed 13;
- target/loss: raw RUL and symmetric RMSE;
- evaluation: development Step 5 only; locked Steps 6-7 remain closed.

The control set contains `flight_cycle`, `log1p_flight_cycle`, and the latest
value of every nonconstant channel. Each treatment adds one signal family's
baseline deviation, direction-normalized degradation score, recent
slopes/deltas, slope acceleration, sustained threshold onset, unsupervised
change point, and state-transition features where applicable. Every feature is
calculated from rows at or before the prefix cutoff.

| Experiment | Added family | Why it is tested | Result |
| --- | --- | --- | --- |
| `PE_signal_control` | None | Establish age/current-value performance | XGB R2 0.6081; ET R2 0.5779 |
| `PE_signal_family_13_16_22_25_28` | Shared five-channel axis | Most consistent fleet-wide degradation group; `25/28` have no drift warning | Supported: R2 improved in 5/5 folds for both models |
| `PE_signal_family_19_21` | Redundant rising pair | Strongest within-UAV RUL association and possible persistent change points | Supported: R2 improved in 5/5 folds for both models |
| `PE_signal_family_15_23` | Inverse pair | Large early-to-late movement but possible fault-mode dependence | Strongest single family: XGB delta R2 +0.1198; ET +0.1237 |
| `PE_signal_family_07` | Discrete state signal | Test transition timing rather than treating channel 07 as continuous | Unsupported: XGB lost all folds; ET result was negligible |
| `PE_signal_all_families` | All four groups | Test complementarity and whether combined signals outweigh added complexity | Retain: strongest result, XGB delta R2 +0.1731 and ET +0.1764, both 5/5 folds |

Run with:

```powershell
.venv\Scripts\python.exe pipeline_experiments\run_experiments.py `
  --group PE_signal_family_ablation
```

Results are written to
`pipeline_experiments/runs/PE_signal_family_ablation/reporting/`. The summary
reports fold-paired R2, RMSE, absolute-bias, overprediction-rate, and RMS
overprediction improvements relative to the same model/fold control. A family
is supported only when accuracy gains are consistent across folds and both
models; leaderboard scores are not used to select it.

## Degradation-Learning Experiment Register

The following groups address distinct pipeline hypotheses. They use XGBoost
and ExtraTrees, `candidate_budget = 25`, the same fold/retraining seeds, raw
RUL evaluation, and `phase_2_scope = "selection_only"`. Each group writes
paired fold and summary tables under its own `reporting/` directory.

| Group | Cells | Question | Result |
| --- | --- | --- | --- |
| `PE_failure_cycle_target` | Raw RUL; failure cycle | Is total lifetime easier and more trajectory-consistent to learn than remaining lifetime directly? | Reject: delta R2 -0.4002 XGB and -0.5073 ET; 0/5 fold wins |
| `PE_baseline_normalization` | Raw; robust normalized; combined | Do within-UAV deviations expose degradation while raw values preserve operating context? | Retain raw: combined was mixed and tiny; robust-only worsened both models |
| `PE_fault_mode` | Global; mode indicator; mode experts | Does one global regressor average two incompatible failure mechanisms? | Retain global: neither treatment improved both models consistently |
| `PE_signal_compression` | Individual; median; PCA; individual+median; individual+PCA | Can redundant signal families be represented more stably with compact health indices? | Retain individual features: compression-only was harmful; added indices were neutral |
| `PE_dense_prefix_training` | `current20`; `dense_stride_5` | Do more real degradation stages improve generalization when each UAV retains equal total weight? | Retain current20: stride 5 lost R2/RMSE in every fold for both models |

### Failure-cycle target

The treatment fits `failure_cycle = cutoff + RUL`. At inference, the adapter
returns `max(0, predicted_failure_cycle - cutoff)` through the existing
nonnegative prediction policy. The raw-RUL cell is the unchanged control. This
changes the learning target without changing features, scenarios, or loss.

Result: reject. XGBoost R2 fell from 0.7577 to 0.3575 and RMSE increased by
18.41 cycles; ExtraTrees R2 fell from 0.7263 to 0.2190 and RMSE increased by
21.56 cycles. The treatment lost every fold for both models.

### Per-UAV baseline normalization

The raw and robust sets contain matched temporal concepts. Robust features
divide slopes, deltas, and last-minus-mean values by an early-life robust scale
computed only from the observed prefix. The combined cell retains both forms.
This tests whether absolute sensor level and within-UAV degradation carry
complementary information.

Result: retain raw features. The combined set gave XGBoost only
`delta R2 = +0.0051` while ExtraTrees lost `0.0065`; robust-only features
worsened both families. The mixed, small effect does not justify added
complexity.

### Fold-fitted fault modes

The treatment derives two modes with training-fold-only scaling and K-means,
using the latest available training prefix for each UAV. `indicator` appends
the assigned mode. `experts` fits one global model plus one model per mode;
assignments beyond the training-fold distance threshold use the global model.
Because these unsupervised modes may reflect degradation state rather than a
physical failure label, they are hypotheses to test, not asserted fault types.

Result: retain the global model. The indicator reduced R2 for both families.
Experts were effectively neutral for ExtraTrees and reduced XGBoost R2 by
0.0114, with no repeatable accuracy or safety gain.

### Signal-family compression

The four degradation families are compressed either by a
direction-oriented median or by a scaler and first principal component fitted
on the training fold. `median_only` and `pca_only` keep age, latest telemetry,
and the family indices; the combined treatments retain all individual signal
features as well. Validation and locked data only reuse the fitted transform.
The group was run after the family ablation established that the underlying
families contain useful signal.

Result: retain the individual signal features. Median-only and PCA-only
representations reduced R2 by roughly 0.09-0.11 in both families. Adding a
median or PCA index alongside the individual features changed R2 by less than
0.004 and was inconsistent across folds.

### Dense prefix training

The treatment changes training-prefix spacing from the `current20` control to
five cycles. Phase 1 continues to allocate equal total sample weight to every
UAV, so longer histories do not dominate merely by producing more rows. This
cell has substantially more training rows and therefore higher runtime and
memory use.

Result: retain `current20`. Dense stride 5 reduced R2 by 0.0338 for XGBoost
and 0.0362 for ExtraTrees, increased RMSE by about two cycles, and lost every
fold for both models. A lower overprediction frequency did not compensate for
worse RMS overprediction and overall accuracy.

Run one group from the repository root with:

```powershell
.venv\Scripts\python.exe pipeline_experiments\run_experiments.py `
  --group PE_failure_cycle_target
```

Replace the group name with any entry in the table. A treatment should advance
only when its paired R2/RMSE gains repeat across folds and model families and
its positive bias, overprediction rate, and RMS overprediction do not violate
the conservative-prediction objective.
