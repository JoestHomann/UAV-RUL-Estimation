# Pipeline Experiments

Last updated: 2026-08-29

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

New cells use `phase_2_scope = "selection_only"`. They stop after development
Step 5 and cannot open locked Step 6. Only the development-selected experiment
is later changed to `"complete"` and evaluated in Steps 6-7.

## Experiment Register

| Experiment | Scenario | Fitting target | What is investigated | Status |
| --- | --- | --- | --- | --- |
| `PE_run_1` | Current | Raw | Baseline task and architecture performance | Complete through Step 7 |
| `PE_2x2_current_cap125` | Current | Cap 125 | Target-cap effect without changing validation scenarios | Step 5 pending |
| `PE_2x2_early_raw` | Early/middle | Raw | Scenario-construction effect without changing the target | Step 5 pending |
| `PE_2x2_early_cap125` | Early/middle | Cap 125 | Interaction between bounded scenarios and capped fitting | Step 5 pending |

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
| `PE_2x2_current_cap125` | XGBoost | Pending | Pending | Pending | Pending | Pending | Pending |
| `PE_2x2_current_cap125` | ExtraTrees | Pending | Pending | Pending | Pending | Pending | Pending |
| `PE_2x2_early_raw` | XGBoost | Pending | Pending | Pending | Pending | Pending | Pending |
| `PE_2x2_early_raw` | ExtraTrees | Pending | Pending | Pending | Pending | Pending | Pending |
| `PE_2x2_early_cap125` | XGBoost | Pending | Pending | Pending | Pending | Pending | Pending |
| `PE_2x2_early_cap125` | ExtraTrees | Pending | Pending | Pending | Pending | Pending | Pending |

The completed `PE_run_1` locked results were XGBoost `R2 = 0.8041`,
`RMSE = 28.4989`, and overprediction rate `70.95%`; ExtraTrees produced
`R2 = 0.7640`, `RMSE = 31.2779`, and overprediction rate `66.60%`. Pairwise
intervals did not establish a decisive R2 or RMSE difference. XGBoost was the
nominal accuracy winner but overpredicted 97.66% of endpoints with true RUL
0-25, so the original task does not satisfy the conservative-prediction goal.

## Decision Rule

1. Complete Step 5 for the three pending cells.
2. Compare only development-selection results across all four cells.
3. Reject candidates with unstable fold behavior or materially worse RMSE/R2.
4. Among similarly accurate candidates, prefer lower positive bias,
   overprediction rate, and RMS overprediction.
5. Record the selected experiment and rationale before changing its
   `phase_2_scope` to `"complete"`.
6. Run locked Steps 6-7 once for that experiment; keep all other cells
   selection-only.

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
| `PE_signal_control` | None | Establish age/current-value performance | Pending |
| `PE_signal_family_13_16_22_25_28` | Shared five-channel axis | Most consistent fleet-wide degradation group; `25/28` have no drift warning | Pending |
| `PE_signal_family_19_21` | Redundant rising pair | Strongest within-UAV RUL association and possible persistent change points | Pending |
| `PE_signal_family_15_23` | Inverse pair | Large early-to-late movement but possible fault-mode dependence | Pending |
| `PE_signal_family_07` | Discrete state signal | Test transition timing rather than treating channel 07 as continuous | Pending |
| `PE_signal_all_families` | All four groups | Test complementarity and whether combined signals outweigh added complexity | Pending |

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
RUL evaluation, and `phase_2_scope = "selection_only"`. Results are pending;
each group writes paired fold and summary tables under its own `reporting/`
directory.

| Group | Cells | Question | Result |
| --- | --- | --- | --- |
| `PE_failure_cycle_target` | Raw RUL; failure cycle | Is total lifetime easier and more trajectory-consistent to learn than remaining lifetime directly? | Pending |
| `PE_baseline_normalization` | Raw; robust normalized; combined | Do within-UAV deviations expose degradation while raw values preserve operating context? | Pending |
| `PE_fault_mode` | Global; mode indicator; mode experts | Does one global regressor average two incompatible failure mechanisms? | Pending |
| `PE_signal_compression` | Individual; median; PCA; individual+median; individual+PCA | Can redundant signal families be represented more stably with compact health indices? | Pending |
| `PE_dense_prefix_training` | `current20`; `dense_stride_5` | Do more real degradation stages improve generalization when each UAV retains equal total weight? | Pending |

### Failure-cycle target

The treatment fits `failure_cycle = cutoff + RUL`. At inference, the adapter
returns `max(0, predicted_failure_cycle - cutoff)` through the existing
nonnegative prediction policy. The raw-RUL cell is the unchanged control. This
changes the learning target without changing features, scenarios, or loss.

### Per-UAV baseline normalization

The raw and robust sets contain matched temporal concepts. Robust features
divide slopes, deltas, and last-minus-mean values by an early-life robust scale
computed only from the observed prefix. The combined cell retains both forms.
This tests whether absolute sensor level and within-UAV degradation carry
complementary information.

### Fold-fitted fault modes

The treatment derives two modes with training-fold-only scaling and K-means,
using the latest available training prefix for each UAV. `indicator` appends
the assigned mode. `experts` fits one global model plus one model per mode;
assignments beyond the training-fold distance threshold use the global model.
Because these unsupervised modes may reflect degradation state rather than a
physical failure label, they are hypotheses to test, not asserted fault types.

### Signal-family compression

The four degradation families are compressed either by a
direction-oriented median or by a scaler and first principal component fitted
on the training fold. `median_only` and `pca_only` keep age, latest telemetry,
and the family indices; the combined treatments retain all individual signal
features as well. Validation and locked data only reuse the fitted transform.
Run this group after the family ablation has established that the underlying
families contain useful signal.

### Dense prefix training

The treatment changes training-prefix spacing from the `current20` control to
five cycles. Phase 1 continues to allocate equal total sample weight to every
UAV, so longer histories do not dominate merely by producing more rows. This
cell has substantially more training rows and therefore higher runtime and
memory use.

Run one group from the repository root with:

```powershell
.venv\Scripts\python.exe pipeline_experiments\run_experiments.py `
  --group PE_failure_cycle_target
```

Replace the group name with any entry in the table. A treatment should advance
only when its paired R2/RMSE gains repeat across folds and model families and
its positive bias, overprediction rate, and RMS overprediction do not violate
the conservative-prediction objective.
