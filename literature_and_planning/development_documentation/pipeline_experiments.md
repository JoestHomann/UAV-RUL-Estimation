# Pipeline Experiments

Last updated: 2026-08-31

## Purpose

`pipeline_experiments` contains controlled experiments that change assumptions
across Phase 1 and Phase 2. Each pipeline run owns one directory. Its named
sub-experiments own resolved settings, checkpoints, and results below
`pipeline_experiments/runs/<pipeline_run>/<sub_experiment>/`. Each numbered run
has one source TOML and one execution entry point under
[`definitions/PE_run_X`](../../pipeline_experiments/definitions/). The top-level
[`pipeline_experiments.toml`](../../pipeline_experiments/pipeline_experiments.toml)
only composes those definitions for compatibility tools.

| Run | Definition | Single entry point |
| --- | --- | --- |
| `PE_run_1` | `definitions/PE_run_1/PE_run_1.toml` | `definitions/PE_run_1/run_PE_run_1.py` |
| `PE_run_2` | `definitions/PE_run_2/PE_run_2.toml` | `definitions/PE_run_2/run_PE_run_2.py` |
| `PE_run_3` | `definitions/PE_run_3/PE_run_3.toml` | `definitions/PE_run_3/run_PE_run_3.py` |
| `PE_run_4` | `definitions/PE_run_4/PE_run_4.toml` | `definitions/PE_run_4/run_PE_run_4.py` |

Every launcher prints its exact script plan before execution. Use
`--list-steps` to review without running and `--step <name>` to rerun one
declared sub-experiment group.

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
| `PE_2x2_early_cap125` | Early/middle | Cap 125 | Interaction between bounded scenarios and capped fitting | Selected; locked Steps 6-7 and Phase 3 Run 4 complete |

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
superiority. Phase 3 Run 4 inherited cap 125 and the symmetric prediction
policy, completed all seven steps, and scored `0.84513` on the public Kaggle
leaderboard. This is a major improvement over Run 3 (`0.54136`) but remains
below the `0.9` objective and below the locked estimate.

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
.venv\Scripts\python.exe pipeline_experiments\definitions\PE_run_2\run_PE_run_2.py `
  --step signal_family_ablation
```

Results are written to
`pipeline_experiments/runs/PE_run_2/PE_signal_family_ablation/reporting/`. The summary
reports fold-paired R2, RMSE, absolute-bias, overprediction-rate, and RMS
overprediction improvements relative to the same model/fold control. A family
is supported only when accuracy gains are consistent across folds and both
models; leaderboard scores are not used to select it.

For every experiment and group, all generated PNG plots are additionally
gathered in `pipeline_experiments/runs/<pipeline_run>/figures/`. The original plots and
numeric artifacts remain in their stage folders; `figure_manifest.json` maps
the flat gallery copies back to those canonical sources.

## PE_run_3: Focused Performance and Safety Study

PE_run_3 implements the ordered sequence
`feature union -> cap sensitivity -> ensemble/calibration -> severity loss ->
frozen locked confirmation`. Every model-training cell uses development Step 5
only. Locked scenarios stay closed until the complete policy is selected and
frozen; Phase 3 remains closed until the confirmation is reviewed.

| Group | Sub-experiments | Why it is investigated | Result |
| --- | --- | --- | --- |
| `PE3_feature_union` | Drift-pruned; all signal families; union | Test whether the previously supported degradation features add information to the Run 4 feature set | Drift-pruned selected; added signal features did not improve the equal-weight two-model result |
| `PE3_cap_sensitivity` | Caps 110, 125, 140, 150 | Determine whether 125 is robust or an arbitrary match to the development endpoint range | Cap 125 retained |
| `PE3_ensemble_calibration` | XGBoost; ExtraTrees; fixed blends; cross-fitted residual functions | Test complementary tree errors and replace a constant safety offset with a learned, leakage-controlled correction | Calibrated 50/50 blend selected: development R2 0.87955, RMSE 11.5080 |
| `PE3_severity_loss` | Symmetric; severity weights 1.5, 2.0, 3.0 | Penalize large overpredictions increasingly strongly while retaining ordinary squared loss for underprediction | Weight 2.0 was safest among loss variants, but R2 0.86802 fell outside the final 0.005 tolerance |
| `PE3_final_ensemble` | Frozen calibrated 50/50 blend | Confirm the complete selected policy once on locked scenarios before Phase 3 | Locked R2 0.89972 and RMSE 10.4383; promoted to Phase 3 Run 5 |

The union is the logical OR of `screened_drift_pruned` and
`signal_all_families`; duplicate features occur only once. The cap cells keep
the early/middle scenarios, models, folds, seeds, and candidate budget fixed.

The ensemble report uses the selected candidate's inner validation predictions
from each outer fold. Blends are predeclared at 25%, 50%, and 75% XGBoost. The
calibrator predicts residual from raw prediction and cutoff, holding out each
inner fold when fitting the correction. It therefore evaluates a function of
predicted severity rather than choosing one scalar offset on the same rows.

For positive residual `e = prediction - target`, severity-loss XGBoost uses
`e^2 + ((w - 1) / 10) * e^3`; for `e <= 0`, it remains `e^2`. This leaves
underprediction at the symmetric baseline penalty while making severe
overprediction progressively more expensive. Selection must still enforce the
predeclared R2/RMSE tolerance before considering safety improvements.

Run the complete sequence with:

```powershell
.venv\Scripts\python.exe pipeline_experiments\definitions\PE_run_3\run_PE_run_3.py
```

The launcher selects feature and cap winners by highest equal-weight mean
development R2 across model families, with RMSE and lexical name as deterministic
tie-breakers. It propagates the selected feature set, target cap, and ensemble
source through a resolved in-memory catalog. For the final safety decision,
only candidates within `0.005` absolute R2 of the best observed method are
eligible; RMS overprediction, overprediction rate, RMSE, and R2 then decide in
that order. The chosen cap-125 feature experiment is reused as the cap control,
and the selected cap experiment's XGBoost predictions are reused as the
symmetric safety control. The exact choices and resolved catalog are stored in
`pipeline_experiments/runs/PE_run_3/workflow/selection_manifest.json` and
`resolved_catalog.json`.

The selected policy is frozen in `PE3_final_ensemble/promotion_contract.json`.
Its calibrator is fitted only from selected development OOF predictions before
the existing Step 6 evaluator opens locked data. Locked XGBoost and ExtraTrees
family/fold checkpoints are resumable and are combined without further tuning.
The confirmation writes seed metrics, RUL-band safety metrics, predictions, and
figures under `PE3_final_ensemble/`; all plots are also copied into
`pipeline_experiments/runs/PE_run_3/figures/`. Phase 3 Run 5 selected candidate
10 of 25, reached development mean-fold R2 0.89422 and RMSE 10.5306, and
produced a verified 100-row submission. Its public Kaggle R2 was 0.86525,
improving on Run 4's 0.84513 while retaining a validation-to-leaderboard gap.

## PE_run_4: Conditional Conservative Calibration

PE_run_4 tests whether the remaining overprediction can be reduced without a
global scalar offset. It starts from Phase 3 Run 5's selected development OOF
predictions and estimates a correction as a function of predicted RUL. For each
outer validation fold, the correction curve is fitted only on the other four
folds. Each bin's correction is the nonnegative residual quantile, and inference
uses `max(0, prediction - interpolated_correction)`, so calibration can never
raise a prediction.

The control and quantiles 0.50, 0.55, 0.60, 0.65, and 0.70 are predeclared in
`definitions/PE_run_4/PE_run_4.toml`. Policies must finish within 0.005 mean-fold R2 of
the best policy before RMS overprediction, overprediction rate, RMSE, and R2
select the winner. No locked or test targets enter fitting or selection.

| Policy | Mean-fold R2 | Mean-fold RMSE | Bias | Overprediction rate | RMS overprediction | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Control | 0.89422 | 10.5306 | +0.083 | 51.2% | 7.255 | Reference |
| `q=0.50` | **0.89456** | 10.5354 | -0.802 | 46.2% | 6.817 | Best accuracy |
| `q=0.55` | 0.89253 | 10.6430 | -1.329 | 43.0% | **6.567** | Selected within tolerance |
| `q=0.60` | 0.88931 | 10.8144 | -1.997 | 40.8% | 6.236 | Outside tolerance |
| `q=0.65` | 0.88106 | 11.2152 | -3.255 | 35.0% | 5.629 | Rejected accuracy loss |
| `q=0.70` | 0.87070 | 11.6989 | -4.512 | 31.6% | 5.053 | Rejected accuracy loss |

The selected `q=0.55` policy reduces the largest systematic excess in the
26-75 RUL region. It is fitted once on all 500 development OOF rows and applied
to the verified, unlabeled Run 5 submission predictions. The candidate
submission, calibrator, cross-fitted predictions, numeric reports, and four
figures are stored under `pipeline_experiments/runs/PE_run_4/`. Leaderboard
confirmation remains pending.

Run or reproduce the complete experiment with:

```powershell
.venv\Scripts\python.exe pipeline_experiments\definitions\PE_run_4\run_PE_run_4.py
```

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
