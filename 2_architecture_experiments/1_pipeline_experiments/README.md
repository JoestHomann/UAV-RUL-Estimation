# Pipeline experiments

This folder is the experiment catalog for questions that span Phase 1 feature
engineering, Phase 2 architecture studies, and optional Phase 3 final training.
It is deliberately separate from `0_data_analysis`: the files here describe
reproducible end-to-end experiments and their outcomes, while that phase owns
feature discovery and diagnostics.
The compact scientific register, experiment rationale, and result record live
in [`pipeline_experiments.md`](../../literature_and_planning/development_documentation/pipeline_experiments.md).

## One folder per experiment

Every numbered pipeline experiment has one reviewable TOML and one execution
entry point under `experiments/PE_X/`. Its executions are separate `run_N`
artifact folders below the same experiment. The TOML owns the scientific cells,
groups, workflow settings, ordered script plan, and active run name.

```text
experiments/
  PE_1/
    run.py
    settings.toml
    README.md
    runs/run_1/
  PE_2/
    run.py
    settings.toml
    README.md
    runs/run_1/
  PE_3/...
  PE_4/...
_internal/
  shared_settings.toml
  phase_2_base_settings.toml
```

For normal use, edit only the `settings.toml` beside the selected `run.py`.
`_internal/` contains implementation defaults shared by the launchers and is
not a second per-run settings surface. PE2 imports the completed PE1 baseline
as a frozen control; new PE2 choices still belong only in PE2's settings file.
`pipeline_experiments.toml` is an include-only compatibility index for tools
that need to inspect all runs; it is not an editable second catalog.

## Run an experiment

From the repository root:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_1\run.py --list
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_1\run.py
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_2\run.py --status
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_2\run.py --only signal_family_ablation
```

Run the launcher without options to execute or resume every declared step in
order. Use `--list` to inspect exact commands, `--status` to inspect artifacts,
and repeat `--only NAME` to rerun selected steps or sub-experiments. Add
`--force` only when an existing completed target must be rebuilt. The shared
manager remains an internal diagnostic interface; ordinary runs should start
through the colocated `run.py`.

## Start another run

Each settings file declares its identity once:

```toml
[pipeline]
experiment = "PE_2"
run = "run_1"
```

To start a second execution of the same large experiment, change only
`run = "run_2"`, make the intended scientific setting changes, and execute the
same `run.py`. The manager writes to `PE_2/runs/run_2/`; `run_1` remains
untouched. Run names must use `run_N` and be unique within that PE folder.
When a run rebuilds Phase 1 or launches Phase 3, also assign unused
`phase_1_run_name` or `phase_3_run_number` values in the same settings file;
reuse them only when reuse is deliberate.

## Signal-family ablation

Run the complete development-only degradation-signal study with one command:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_2\run.py `
  --only signal_family_ablation
```

The group builds one shared `current20` Phase 1 run, then compares XGBoost and
ExtraTrees using the same folds, seeds, raw target, symmetric loss, and
25-candidate budget. The control contains cycle age and the latest value of all
22 nonconstant telemetry channels. Treatments add temporal features for
`13/16/22/25/28`, `19/21`, `15/23`, `07`, or all four families.

Every component has `phase_2_scope = "selection_only"`. After all six complete,
the runner writes paired fold results, a summary, a figure, and a manifest to
`2_architecture_experiments/1_pipeline_experiments/experiments/PE_2/runs/run_1/PE_signal_family_ablation/reporting/`. Positive
`r2_improvement` and `rmse_improvement` favor the treatment. Retain a family
only when gains repeat across folds and both model families.

## Degradation-learning experiments

The following groups are predeclared development studies. Every cell uses
XGBoost and ExtraTrees, a 25-candidate Step 5 search, and keeps locked Steps
6-7 closed:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_2\run.py --only failure_cycle_target
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_2\run.py --only baseline_normalization
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_2\run.py --only fault_mode
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_2\run.py --only signal_compression
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_2\run.py --only dense_prefix_training
```

- `PE_failure_cycle_target` compares direct raw-RUL regression with prediction
  of total failure cycle followed by `RUL = predicted_failure_cycle - cutoff`.
- `PE_baseline_normalization` compares matched raw temporal features,
  early-life robust-normalized features, and their union.
- `PE_fault_mode` compares one global model, a fold-fitted two-mode indicator,
  and two fold-fitted mode experts with a global fallback for uncertain UAVs.
- `PE_signal_compression` compares individual degradation features with
  direction-oriented family medians and fold-fitted family PCA scores, alone
  and alongside the individual features.
- `PE_dense_prefix_training` compares `current20` with five-cycle prefix
  spacing while preserving equal total training weight per UAV.

After a group finishes, its `reporting/` folder contains
`paired_fold_results.csv`, `paired_summary.csv`,
`paired_comparison.png`, and `report_manifest.json`. Every experiment and group
also contributes to the flat `experiments/PE_X/runs/run_N/figures/` gallery. Its
`figure_manifest.json` records each plot's canonical source; detailed data and
canonical outputs remain in their sub-experiment and stage folders. The
fault-mode expert and dense-prefix cells perform more fitting work than their
controls and should be expected to run longer.

## PE_3 performance experiments

`PE_3` contains four ordered development-only sub-experiment groups and one
frozen locked-confirmation gate. Run or resume the complete workflow with one
command:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_3\run.py
```

1. `PE3_feature_union` compares the Run 4 drift-pruned features, all supported
   signal-family features, and their explicit union.
2. `PE3_cap_sensitivity` compares fitting-target caps 110, 125, 140, and 150
   with the same early/middle scenarios.
3. `PE3_ensemble_calibration` reuses selected inner-fold predictions to compare
   XGBoost, ExtraTrees, fixed 25/50/75% XGBoost blends, and cross-fitted
   polynomial residual corrections based on raw prediction and cutoff.
4. `PE3_severity_loss` compares symmetric XGBoost with three superlinear
   overprediction penalties. For residual `e = prediction - target`, the loss
   is `e^2 + ((weight - 1) / 10) * max(e, 0)^3`.

The workflow automatically selects feature and cap winners by highest
equal-weight mean development R2 across model families, then lowest RMSE. It
propagates those choices into downstream cells without editing the source TOML.
The selected cap-125 feature cell is reused as the cap control, and the selected
cap cell's XGBoost predictions are reused as the symmetric safety control, so
those searches are not duplicated.
The final safety selection admits candidates within `0.005` R2 of the best
observed candidate, then minimizes RMS overprediction, overprediction rate, and
RMSE in that order. The resolved catalog and development-only decision are written under
`2_architecture_experiments/1_pipeline_experiments/experiments/PE_3/runs/run_1/workflow/`.

All training cells remain `selection_only`. After the final policy is selected,
`PE3_final_ensemble` freezes its component weights and residual calibrator from
development OOF rows, then opens locked scenarios once. It evaluates XGBoost
and ExtraTrees over five outer folds and three retraining seeds, combines their
aligned predictions, and writes locked accuracy and RUL-band safety diagnostics
below `2_architecture_experiments/1_pipeline_experiments/experiments/PE_3/runs/run_1/PE3_final_ensemble/`. It does not
create Phase 3; that remains a manual gate after reviewing the locked result.

New Step 5 runs export
`selected_inner_predictions.csv.gz`, which makes ensemble and calibration
comparisons reproducible without refitting candidates.

The individual `--group PE3_*` commands remain available for targeted reruns,
but they use the static TOML defaults and do not propagate winners. Rerunning
`--run PE_3` resumes completed cells, rebuilds the automatic reports and
selection manifest, and resumes incomplete locked component folds. To resume
only the confirmation step, run:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\run_experiments.py --run PE3_final_ensemble
```

All PE_3 plots are collected directly in
`2_architecture_experiments/1_pipeline_experiments/experiments/PE_3/runs/run_1/figures/`. Sub-experiments retain only their
canonical reports and numeric artifacts, not separate gallery copies.

## PE_4 conditional safety calibration

`PE_4` reuses only the selected development OOF predictions from Phase 3
Run 5. It cross-fits subtraction-only residual curves at quantiles 0.50, 0.55,
0.60, 0.65, and 0.70 as functions of predicted RUL. No model is retrained and
no locked or test targets are loaded. Run or reproduce it with:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_4\run.py
```

The automatic rule admits policies within 0.005 mean-fold R2 of the best, then
minimizes RMS overprediction, overprediction rate, and RMSE. It selected
`q=0.55`: mean-fold R2 changed from 0.89422 to 0.89253, overprediction rate
from 51.2% to 43.0%, and RMS overprediction from 7.255 to 6.567. The selected
calibrator and candidate submission are under
`2_architecture_experiments/1_pipeline_experiments/experiments/PE_4/runs/run_1/artifacts/`; the four plots are gathered in
`2_architecture_experiments/1_pipeline_experiments/experiments/PE_4/runs/run_1/figures/`. Kaggle confirmation is pending.

## Target/scenario 2x2 experiment

PE2 contains a predeclared four-cell comparison of current versus early/middle
validation scenarios and raw versus cap-125 fitting targets. `PE_1`
supplies the completed current/raw control. Run or resume the ordered group
through PE2's single entry point:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_2\run.py `
  --only target_scenario_2x2
```

The early/raw run builds the shared early/middle Phase 1 artifacts; the
early/capped run reuses them. All three new cells initially used
`phase_2_scope = "selection_only"`, which stopped after Step 5 and prevented
locked evaluation during development selection.

Compare the selected development candidates without reading Step 6:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\compare_experiments.py `
  --scope selection `
  --run PE_1 `
  --run PE_2x2_current_cap125 `
  --run PE_2x2_early_raw `
  --run PE_2x2_early_cap125
```

This writes `2_architecture_experiments/1_pipeline_experiments/selection_experiment_comparison.csv`. After
recording the development-only decision, change only the winning experiment to
`phase_2_scope = "complete"` and rerun it. The manager preserves Step 5,
performs Steps 6-7 once, and writes the locked comparison in the same experiment
folder. The non-winning cells remain selection-only.

Current status: `PE_2x2_early_cap125` was the sole selected cell. Its locked
Steps 6-7 are complete, it now has `phase_2_scope = "complete"`, and Phase 3
Run 4 completed with XGBoost and scored `0.84513` publicly. The two losing cells
remain selection-only.
The default Phase 3 TOML now contains the selected experiment's artifact paths,
so the primary direct command is:

```powershell
.venv\Scripts\python.exe 3_final_model_training_and_inference\run_phase_3.py
```

The experiment manager remains available when catalog-level status tracking is
desired:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\run_experiments.py `
  --run PE_2x2_early_cap125 `
  --from-stage phase3 `
  --through-stage phase3
```

Phase 1 creates a named run under
`1_dataset_construction/runs/<phase_1_run_name>/`. Phase 2 reads the dedicated
`2_architecture_experiments/1_pipeline_experiments/_internal/phase_2_base_settings.toml`, binds the selected experiment and
Phase 1 interface, and writes its resolved settings under
`2_architecture_experiments/1_pipeline_experiments/experiments/PE_X/runs/run_N/<sub_experiment>/phase2/` for grouped
sub-experiments, while a top-level experiment uses
`2_architecture_experiments/1_pipeline_experiments/experiments/PE_X/runs/run_N/phase2/`. Steps 5-7 use the same
experiment-owned Phase 2 folder. The Phase 2 run number remains recorded
as scientific metadata but no longer selects a standalone Phase 2 directory.
Phase 3 uses the configured numbered Phase 3 run folder. All generated
manifests retain the exact paths used by that experiment. Phase 2 Steps 5 and
6 use the standard parallel orchestrator;
`max_workers` controls the number of independent family/outer-fold subprocesses
that can run concurrently.

Step 7 automatically generates asymmetric safety diagnostics in the
experiment-owned `phase2/7_architecture_comparison/figures/` directory and
gathers them in the run's top-level `figures/` directory. These
include residual ECDFs with fixed offset thresholds, overprediction rate and
RMS magnitude by true-RUL band, fixed-offset accuracy/safety tradeoffs,
positive-residual P90/P95/maximum tails, and per-family prediction alignment
before and after the six-cycle diagnostic offset. Offset plots are descriptive:
they do not select or apply an offset to a trained model.

Refresh one run's figure gallery without rerunning its experiment:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_3\run.py --collect-figures
```

The compatibility manager can still refresh all galleries for maintenance.

The launcher resumes Phase 2 and Phase 3 checkpoints. A partial Phase 1 or
Phase 2 run can be continued by running the same command again; an interrupted
stage is recorded as `interrupted` until it is resumed. Use new
`phase_1_run_name`, `phase_2_run_number`, and `phase_3_run_number` values for a
new scientific experiment. The experiment name owns its complete artifact
directory; the Phase 2 run number is retained for manifest identity and Phase
3 traceability. `--force` is intended for rerunning the selected
Phase 3 range; it does not change the TOML catalog.

## Configuration ownership

Standalone Phase 2 runs use
`2_architecture_experiments/2_model_architecture_study/1_architecture_study_settings/architecture_study_settings.toml`.
Pipeline experiments never read or modify that file. They use the independent
`2_architecture_experiments/1_pipeline_experiments/_internal/phase_2_base_settings.toml` for Phase 2 defaults and model
search spaces, while each `experiments/PE_X/settings.toml` owns that experiment's
named scientific choices.
The two workflows may therefore evolve without silently changing each other's
runs.

## Edit the catalog

Create one new `experiments/PE_X/` folder containing `run.py`,
`settings.toml`, and a compact `README.md`. Give its `[experiments.*]` blocks
new names, set `[pipeline].experiment`, and begin with `[pipeline].run =
"run_1"`. Add the settings file to the compatibility index only after it loads
independently.

- `phase_1_profile`, `scenario_profile`, and `prefix_variant` choose the data
  construction policy;
- `architectures`, `feature_set`, `candidate_budget`, and seeds choose the
  Phase 2 study;
- `fault_mode_strategy` and `signal_compression_strategy` select fold-fitted
  tabular transformations declared by an experiment;
- `phase_2_scope = "selection_only"` stops at development selection, while
  `"complete"` permits the locked evaluation and comparison;
- `target_profile` and `prediction_profile` choose raw/capped targets and
  symmetric/conservative fitting;
- the Phase 3 fields enable optional final selection, training, inference,
  submission verification, and model-agnostic reporting.

Each run's `[execution].max_workers` setting controls parallel Step 5 and Step
6 study workers for that large experiment. Set it to a positive integer or to
`"auto"`; it is an execution setting, not a scientific model setting.

The manager does not create an implicit Cartesian product. Every experiment
block remains one named scientific question. An `[experiment_groups.*]` table
only orders explicitly listed experiments and invokes its declared reporter.

## Compare and record scores

After one or more Phase 2 studies finish:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\compare_experiments.py --scope locked
```

This writes `experiment_comparison.csv` from the locked, offline comparison
tables. Public competition scores are entered manually only for selected
submissions:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\record_leaderboard_result.py `
  --run PE_1 --public-score 0.61234 `
  --submission-description PE_1
```

The record is stored in the experiment's artifact folder below
`2_architecture_experiments/1_pipeline_experiments/experiments/PE_X/runs/run_N/` and is kept
separate from the offline validation results. No test labels are loaded by
these tools, and no automatic leaderboard submission is attempted.
