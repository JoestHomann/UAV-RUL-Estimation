# Pipeline Experiments

Last updated: 2026-09-01

## Purpose

`pipeline_experiments` contains controlled experiments that change assumptions
across Phase 1 and Phase 2. Each `PE_X` owns one directory and may contain
multiple execution folders named `run_N`. Its named sub-experiments own resolved
settings, checkpoints, and results below
`2_architecture_experiments/1_pipeline_experiments/experiments/PE_X/runs/run_N/<sub_experiment>/`.
Each experiment has one user-editable `settings.toml` and one `run.py` entry
point under
[`experiments/PE_X`](../../2_architecture_experiments/1_pipeline_experiments/experiments/). The top-level
[`pipeline_experiments.toml`](../../2_architecture_experiments/1_pipeline_experiments/pipeline_experiments.toml)
only composes those settings for compatibility tools. Shared implementation
defaults live under `_internal/` and are not an additional per-run edit point.

| Run | Definition | Single entry point |
| --- | --- | --- |
| `PE_1` | `experiments/PE_1/settings.toml` | `experiments/PE_1/run.py` |
| `PE_2` | `experiments/PE_2/settings.toml` | `experiments/PE_2/run.py` |
| `PE_3` | `experiments/PE_3/settings.toml` | `experiments/PE_3/run.py` |
| `PE_4` | `experiments/PE_4/settings.toml` | `experiments/PE_4/run.py` |
| `PE_5` | `experiments/PE_5/settings.toml` | `experiments/PE_5/run.py` |

Every launcher prints its exact script plan before execution. Use `--list` to
review without running, `--status` to inspect saved progress, and
`--only <name>` to rerun one declared step or sub-experiment group. Running
without options executes or resumes the complete large experiment.

The active execution is declared once in each settings file:

```toml
[pipeline]
experiment = "PE_X"
run = "run_1"
```

To start another execution of the same experiment, change only `pipeline.run`
to an unused name such as `run_2`, edit the intended scientific settings, and
invoke the same `run.py`. Existing run folders are not overwritten. A run that
rebuilds Phase 1 or launches Phase 3 must also use unused external Phase 1/3
identities unless reuse is an intentional part of the experiment.

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
| `PE_1` | Current | Raw | Baseline task and architecture performance | Complete through Step 7 |
| `PE_2x2_current_cap125` | Current | Cap 125 | Target-cap effect without changing validation scenarios | Step 5 complete; rejected |
| `PE_2x2_early_raw` | Early/middle | Raw | Scenario-construction effect without changing the target | Step 5 complete; rejected |
| `PE_2x2_early_cap125` | Early/middle | Cap 125 | Interaction between bounded scenarios and capped fitting | Selected; locked Steps 6-7 and Phase 3 Run 4 complete |

### PE_1: current scenarios, raw target

This is the unchanged control. It uses current validation scenarios and raw RUL
for fitting and evaluation. It originally compared the mean baseline, Random
Forest, ExtraTrees, and XGBoost; XGBoost and ExtraTrees provide the matrix
control values. Its predictions reproduce the corresponding Run 5 results.

The experiment establishes that stronger architecture tuning alone does not
close the validation-to-leaderboard gap and quantifies systematic
overprediction under the original task.

### PE_2x2_current_cap125: current scenarios, capped target

This cell reuses the `PE_1` Phase 1 artifacts and changes only the fitting
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
`2_architecture_experiments/1_pipeline_experiments/selection_experiment_comparison.csv` after all selections
finish.

| Experiment | Family | Inner RMSE mean | Fold SD | Inner R2 mean | Bias | Overprediction rate | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `PE_1` | XGBoost | 29.8059 | 1.7513 | 0.7708 | +5.2165 | 67.35% | Current/raw control |
| `PE_1` | ExtraTrees | 32.7578 | 2.6545 | 0.7264 | +4.4906 | 66.00% | Lower safety bias, weaker RMSE |
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

The completed `PE_1` locked results were XGBoost `R2 = 0.8041`,
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
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_2\run.py `
  --only signal_family_ablation
```

Results are written to
`2_architecture_experiments/1_pipeline_experiments/experiments/PE_2/runs/run_1/PE_signal_family_ablation/reporting/`. The summary
reports fold-paired R2, RMSE, absolute-bias, overprediction-rate, and RMS
overprediction improvements relative to the same model/fold control. A family
is supported only when accuracy gains are consistent across folds and both
models; leaderboard scores are not used to select it.

For every experiment and group, all generated PNG plots are additionally
gathered in `2_architecture_experiments/1_pipeline_experiments/experiments/PE_X/runs/run_N/figures/`. The original plots and
numeric artifacts remain in their stage folders; `figure_manifest.json` maps
the flat gallery copies back to those canonical sources.

## PE_3: Focused Performance and Safety Study

PE_3 implements the ordered sequence
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
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_3\run.py
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
`2_architecture_experiments/1_pipeline_experiments/experiments/PE_3/runs/run_1/workflow/selection_manifest.json` and
`resolved_catalog.json`.

The selected policy is frozen in `PE3_final_ensemble/promotion_contract.json`.
Its calibrator is fitted only from selected development OOF predictions before
the existing Step 6 evaluator opens locked data. Locked XGBoost and ExtraTrees
family/fold checkpoints are resumable and are combined without further tuning.
The confirmation writes seed metrics, RUL-band safety metrics, predictions, and
figures under `PE3_final_ensemble/`; all plots are also copied into
`2_architecture_experiments/1_pipeline_experiments/experiments/PE_3/runs/run_1/figures/`. Phase 3 Run 5 selected candidate
10 of 25, reached development mean-fold R2 0.89422 and RMSE 10.5306, and
produced a verified 100-row submission. Its public Kaggle R2 was 0.86525,
improving on Run 4's 0.84513 while retaining a validation-to-leaderboard gap.

## PE_4: Conditional Conservative Calibration

PE_4 tests whether the remaining overprediction can be reduced without a
global scalar offset. It starts from Phase 3 Run 5's selected development OOF
predictions and estimates a correction as a function of predicted RUL. For each
outer validation fold, the correction curve is fitted only on the other four
folds. Each bin's correction is the nonnegative residual quantile, and inference
uses `max(0, prediction - interpolated_correction)`, so calibration can never
raise a prediction.

The control and quantiles 0.50, 0.55, 0.60, 0.65, and 0.70 are predeclared in
`experiments/PE_4/settings.toml`. Policies must finish within 0.005 mean-fold R2 of
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
figures are stored under `2_architecture_experiments/1_pipeline_experiments/experiments/PE_4/runs/run_1/`. Leaderboard
R2 was 0.86525, unchanged from the source Run 5 submission at displayed
leaderboard precision. The safety calibration improved development safety
metrics but did not produce a measurable public-score gain by itself.

Run or reproduce the complete experiment with:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_4\run.py
```

## PE_5: Target-Tail Submission Experiment

PE_5 tests whether the hard cap at RUL 125 removes useful information needed
for the hidden Kaggle distribution. It fixes the selected Run 5
`screened_drift_pruned` features, selected ExtraTrees and XGBoost component
configurations, 50/50 blend, UAV folds, model seed 13, and q=0.55 conditional
calibration. Only the training-target treatment changes:

| Variant | Treatment above RUL 125 | Purpose |
| --- | --- | --- |
| `hard_cap_125` | Replace with 125 | Current bounded-target control |
| `raw` | Preserve the label | Test the complete observed target range |
| `weighted_raw` | Preserve the label; multiply its row weight by 0.25 | Retain tail information with reduced influence |
| `soft_tail` | Fit `125 + 0.50 * (RUL - 125)` and invert after prediction | Retain ordering while compressing the tail |

Weighted rows are renormalized within each UAV so every UAV retains equal total
training weight. The old polynomial residual calibrator is not reused because
it was fitted to capped predictions. Each variant instead receives its own
q=0.55 calibrator fitted from development OOF predictions only. The workflow
then trains on all available training prefixes, saves and reloads the complete
model bundle, checks prediction equivalence, and verifies submission IDs and
RUL values. Locked labels and test targets are never loaded.

### Run 1 development evidence

| Variant | Mean R2 | Mean RMSE | Bias | Test prediction range | Kaggle R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hard_cap_125` | **0.8802** | **11.19** | -1.75 | 6.04-122.14 | **0.85609** |
| `soft_tail` | 0.5422 | 18.22 | +0.45 | 6.30-187.03 | 0.83609 |
| `weighted_raw` | 0.5268 | 18.78 | +0.45 | 6.09-186.03 | 0.83482 |
| `raw` | 0.5157 | 18.95 | +0.61 | 5.48-232.84 | 0.82866 |

Hard cap 125 is decisively best on development data and the public leaderboard
confirms the direction. Relative to the cap control, soft tail lost 0.02000,
weighted raw lost 0.02127, and raw RUL lost 0.02743. Increasing the amount and
range of high-RUL prediction therefore made performance progressively worse;
the cap should remain in the current pipeline. The large uncapped errors are
not fixed by restoring the raw target tail.

The PE_5 cap control is valid for comparisons among the four PE_5 target
policies but is not an exact reproduction of Phase 3 Run 6. PE_5 deliberately
omits the older capped-prediction residual calibrator because it was not valid
for uncapped variants. Phase 3 Run 6 retains the selected calibrated blend,
uses cap 125 plus q=0.55 conditional calibration, and scored 0.86741. That is
0.01132 above the PE_5 cap control and 0.00216 above the previous 0.86525 best.

Upload-ready files are stored in
`experiments/PE_5/runs/run_1/submissions/`. All four leaderboard scores have
been recorded; the target-policy decision is complete.

Run or resume the complete experiment with:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_5\run.py
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
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_2\run.py `
  --only failure_cycle_target
```

Replace the group name with any entry in the table. A treatment should advance
only when its paired R2/RMSE gains repeat across folds and model families and
its positive bias, overprediction rate, and RMS overprediction do not violate
the conservative-prediction objective.

## R2 Above 0.9 Development Program

These experiments test larger changes to the learning formulation after the
feature, cap, and calibration studies reached a public leaderboard plateau.
Unless explicitly stated otherwise, every result is development-only and does
not reopen locked evaluation.

| Experiment | What changes | Why | Result |
| --- | --- | --- | --- |
| `PE_6` | Sparse versus stride-2 versus every-cycle sequence endpoints, followed by lookbacks 20/30/50 | Test whether temporal models need denser causal supervision than the failed stride-5 tabular experiment provided | Dense stride 1 selected; lookback 20 selected |
| Temporal architecture `run_7` | TCN, multiscale CNN, GRU, and LSTM on the frozen PE_6 sampling policy | Find a compact temporal model whose errors complement the tree blend | Complete; no promotion. LSTM was best at mean OOF R2 0.6531 and RMSE 19.57 versus required values of 0.89 and 10.7 |
| `PE_7` | Fixed blends, nonnegative ridge, and shallow-XGBoost OOF stacks | Test whether a complementary temporal model corrects tree residuals without meta-model leakage | Blocked; Run 7 produced no eligible temporal component, so stacking was not run |
| `PE_8` | Global cap 125 versus temporal-correlation and monotonic-health-index onset targets | Test whether a universal degradation-onset assumption causes the 51-125 RUL errors | Complete; no promotion. Both onset targets lost all five folds and raised tree RMSE from about 12.2 to about 19.6-20.0 |
| `PE_9` | Control, top-5/top-10 shift pruning, and fold-local target-aware pruning | Test whether development/test-separating features reduce hidden-domain robustness | Complete; no promotion. Domain AUC was 0.8565, but no pruning treatment passed the accuracy and high-propensity gates |
| `PE_10` | Fixed hybrid CNN with recent-only versus pooled long-history plus recent raw telemetry | Determine whether multi-resolution history improves a joint sequence/engineered representation | Complete; recent-only retained. Multi-resolution RMSE 20.332 versus 18.336, 0/5 wins, 10.89% worse |
| Hybrid architecture `run_8` | XGBoost, hybrid CNN, and hybrid GRU using the frozen PE_10 representation, promoted against frozen PE_3 tree OOF predictions | Test whether temporal information adds value when the model also receives all 298 engineered full-history features | Complete; no promotion. Hybrid GRU RMSE 14.618 and hybrid CNN 17.671 versus tree-blend 11.508; neither won a fold |
| `PE_11` | Three-seed XGBoost/ExtraTrees members, mean/median/trimmed aggregation, nested blend fitting, and shallow residual correction | Test variance reduction and systematic residual learnability without changing data or opening locked scenarios | Implemented; pending execution |
| Architecture `run_9` | Raw scalar XGBoost, right-censored XGBoost AFT, and discrete failure-horizon XGBoost | Test whether the cap should represent censoring or a survival curve rather than an exact scalar target | Implemented; pending execution |
| `PE_12` | Propensity-weighted ranking of PE_3, PE_11, and Run 9 OOF predictions | Test whether candidate choice changes in the part of development data most similar to the unlabeled test endpoints | Implemented; pending PE_11 and Run 9 |
| `PE_13` | Prediction-band and ensemble-uncertainty-dependent conservative subtraction | Reduce near-failure overprediction more selectively than a global scalar or prediction-only quantile curve | Implemented; pending PE_11 |

Each experiment has one entry point and one editable TOML under
`2_architecture_experiments/1_pipeline_experiments/experiments/PE_X/`. PE_7 is
blocked because temporal Run 7 failed its development gate and did not enter
three-seed confirmation. Exact earlier gates and artifact contracts are
documented in `r2_above_0_9_implementation_plan.md`.

### PE_8 to PE_10 conclusions

PE_8 rejected both personalized-onset formulations. XGBoost RMSE increased
from `12.154` for cap 125 to `19.686` for temporal correlation and `19.593` for
the monotonic health index. ExtraTrees showed the same failure. This indicates
that these fold-fitted change points discard useful absolute target information;
it does not prove that every personalized degradation model is impossible.

PE_9 confirmed measurable covariate shift (`OOF domain AUC = 0.8565`) but
showed that the most shifted features cannot simply be removed. The nominally
best target-aware XGBoost treatment changed relative RMSE by only `+0.013%`,
won two folds, and worsened high-propensity RMSE by `0.282`. The shift signal is
therefore retained as a validation diagnostic in PE_12, not treated as feature
importance or justification for test-distribution weighting during training.

PE_10 and Run 8 jointly reject the tested temporal representations. Adding
pooled long history made the fixed hybrid CNN 10.89% worse. With recent-only
history, hybrid GRU and hybrid CNN still trailed the current tree blend by
27.0% and 53.5% RMSE, respectively. The temporal branch therefore should not
be added to Phase 3 from these results.

### PE_11: cross-fitted bagging and residual correction

PE_11 regenerates development predictions for the five outer studies, four
inner folds, XGBoost and ExtraTrees, and seeds 13, 37, and 73. Every member is
trained without its validation UAVs. It compares the member mean, median,
trimmed mean, an inner-fold-selected nonnegative family blend, and a shallow
histogram-gradient residual model using base prediction, cutoff, seed spread,
range, and family disagreement. Blend tuning and residual fitting use three
inner folds and predict the fourth inside the same outer study. The provenance
tables must report zero UAV overlap.

Promotion requires at least four outer-fold wins over the frozen calibrated
PE_3 blend and at least 1% mean RMSE improvement. Artifacts are checkpointed
after every member fit, so an interrupted run resumes rather than restarting
completed fits.

```powershell
& .\.venv\Scripts\python.exe `
  .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_11\run.py
```

### Architecture Run 9: censored and horizon targets

Run 9 is an architecture study because it changes the estimator's output and
loss contract. `xgboost_aft` treats observed RUL up to 125 as exact and values
above 125 as right-censored at 125. `horizon_xgboost` learns ordered failure
probabilities at 10, 20, 35, 50, 70, 90, 110, and 125 cycles and integrates the
monotone survival curve into a capped RUL estimate. Raw scalar XGBoost is the
within-run control; promotion is nevertheless judged against the stronger
frozen PE_3 calibrated blend.

The adapters are registered in the shared Phase 2 model interface, but Run 9
stops after Step 5 and its dedicated report. A treatment needs four fold wins
and 2% lower RMSE than the current tree blend before locked confirmation could
be considered.

```powershell
& .\.venv\Scripts\python.exe `
  .\2_architecture_experiments\2_model_architecture_study\run_censored_architecture_study.py
```

### PE_12: test-like validation selection

PE_12 does not fit a model on test rows and never reads test labels. It reuses
PE_9's repeated OOF development-versus-test propensity estimates to compute
clipped density-ratio weights, then ranks frozen PE_3, PE_11, and Run 9 OOF
predictions by weighted RMSE. It also reports ordinary RMSE, highest-propensity
quintile RMSE, and effective sample size. A weighted winner is ineligible if
ordinary RMSE regresses by more than 0.5% or effective sample size falls below
50% of the fold rows.

Run PE_11 and Run 9 first:

```powershell
& .\.venv\Scripts\python.exe `
  .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_12\run.py
```

### PE_13: uncertainty-dependent safety

PE_13 starts from PE_11's automatically selected prediction method and member
standard deviation. It subtracts `strength * uncertainty * severity_multiplier`, where
the multiplier is largest for low predicted RUL. Prediction bands, not true
RUL bands, define the deployed correction. Strength is selected on three inner
folds and applied to the fourth. This directly tests whether disagreement marks
dangerous overprediction rather than assuming that every endpoint needs the
same offset.

The safety policy advances only if near-failure RMS overprediction improves by
at least 5% while mean RMSE regresses by no more than 0.5%. It is a safety variant, not an
accuracy-discovery shortcut, and remains separate from the canonical Phase 3
submission unless it passes that gate.

```powershell
& .\.venv\Scripts\python.exe `
  .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_13\run.py
```

### Execution order

1. Run PE_11 and architecture Run 9; they are independent and may run in
   separate terminals if GPU/CPU contention is acceptable.
2. Run PE_12 after both upstream OOF tables exist.
3. Run PE_13 after PE_11.
4. Compare the development manifests. Do not open locked evaluation unless one
   candidate passes its declared promotion gate.
