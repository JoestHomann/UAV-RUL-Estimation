# Pipeline experiments

This folder is the experiment catalog for questions that span Phase 1 feature
engineering, Phase 2 architecture studies, and optional Phase 3 final training.
It is deliberately separate from `0_data_analysis`: the files here describe
reproducible end-to-end experiments and their outcomes, while that phase owns
feature discovery and diagnostics.

## Run an experiment

From the repository root:

```powershell
.venv\Scripts\python.exe pipeline_experiments\run_experiments.py --list
.venv\Scripts\python.exe pipeline_experiments\run_experiments.py --run PE_run_1
.venv\Scripts\python.exe pipeline_experiments\run_experiments.py --run PE_run_1 --from-stage phase1 --through-stage phase1
.venv\Scripts\python.exe pipeline_experiments\run_experiments.py --run PE_run_1 --from-stage phase3 --through-stage phase3
.venv\Scripts\python.exe pipeline_experiments\run_experiments.py --status
```

Each experiment can set `from_stage` and `through_stage` in the TOML. These
values are used when the command does not provide the corresponding CLI
option; explicit CLI options override the catalog for one invocation.

## Target/scenario 2x2 experiment

The catalog contains a predeclared four-cell comparison of current versus
early/middle validation scenarios and raw versus cap-125 fitting targets.
`PE_run_1` supplies the completed current/raw control. Run the other three
development selections in this order:

```powershell
.venv\Scripts\python.exe pipeline_experiments\run_experiments.py --run PE_2x2_current_cap125
.venv\Scripts\python.exe pipeline_experiments\run_experiments.py --run PE_2x2_early_raw
.venv\Scripts\python.exe pipeline_experiments\run_experiments.py --run PE_2x2_early_cap125
```

The early/raw run builds the shared early/middle Phase 1 artifacts; the
early/capped run reuses them. All three new cells declare
`phase_2_scope = "selection_only"`, which stops after Step 5 and cannot launch
locked evaluation or architecture comparison.

Compare the selected development candidates without reading Step 6:

```powershell
.venv\Scripts\python.exe pipeline_experiments\compare_experiments.py `
  --scope selection `
  --run PE_run_1 `
  --run PE_2x2_current_cap125 `
  --run PE_2x2_early_raw `
  --run PE_2x2_early_cap125
```

This writes `pipeline_experiments/selection_experiment_comparison.csv`. After
recording the development-only decision, change only the winning experiment to
`phase_2_scope = "complete"` and rerun it. The manager preserves Step 5,
performs Steps 6-7 once, and writes the locked comparison in the same experiment
folder. The non-winning cells remain selection-only.

Phase 1 creates a named run under
`1_dataset_construction/runs/<phase_1_run_name>/`. Phase 2 reads the dedicated
`pipeline_experiments/phase_2_settings.toml`, binds the selected experiment and
Phase 1 interface, and writes its resolved settings under
`pipeline_experiments/runs/<experiment>/phase2/`, while Steps 5-7 use the
same experiment-owned Phase 2 folder. The Phase 2 run number remains recorded
as scientific metadata but no longer selects a standalone Phase 2 directory.
Phase 3 uses the configured numbered Phase 3 run folder. All generated
manifests retain the exact paths used by that experiment. Phase 2 Steps 5 and
6 use the standard parallel orchestrator;
`max_workers` controls the number of independent family/outer-fold subprocesses
that can run concurrently.

Step 7 automatically generates asymmetric safety diagnostics in the
experiment-owned `phase2/7_architecture_comparison/figures/` directory. These
include residual ECDFs with fixed offset thresholds, overprediction rate and
RMS magnitude by true-RUL band, fixed-offset accuracy/safety tradeoffs,
positive-residual P90/P95/maximum tails, and per-family prediction alignment
before and after the six-cycle diagnostic offset. Offset plots are descriptive:
they do not select or apply an offset to a trained model.

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
`2_model_architecture_study/1_architecture_study_settings/architecture_study_settings.toml`.
Pipeline experiments never read or modify that file. They use the independent
`pipeline_experiments/phase_2_settings.toml` for Phase 2 defaults and model
search spaces, while `pipeline_experiments.toml` owns named experiment choices.
The two workflows may therefore evolve without silently changing each other's
runs.

## Edit the catalog

Copy an `[experiments.*]` block, give it a new name and new run identities,
then edit the question-specific fields:

- `phase_1_profile`, `scenario_profile`, and `prefix_variant` choose the data
  construction policy;
- `architectures`, `feature_set`, `candidate_budget`, and seeds choose the
  Phase 2 study;
- `phase_2_scope = "selection_only"` stops at development selection, while
  `"complete"` permits the locked evaluation and comparison;
- `target_profile` and `prediction_profile` choose raw/capped targets and
  symmetric/conservative fitting;
- the Phase 3 fields enable optional final selection, training, inference,
  submission verification, and model-agnostic reporting.

The global `[execution].max_workers` setting controls parallel Step 5 and Step
6 study workers for every experiment. Set it to a positive integer or to
`"auto"`; it is an execution setting, not a scientific model setting.

The manager does not create an implicit Cartesian product. Every block is one
named scientific question, which keeps leaderboard submissions and result
interpretation straightforward.

## Compare and record scores

After one or more Phase 2 studies finish:

```powershell
.venv\Scripts\python.exe pipeline_experiments\compare_experiments.py --scope locked
```

This writes `experiment_comparison.csv` from the locked, offline comparison
tables. Public competition scores are entered manually only for selected
submissions:

```powershell
.venv\Scripts\python.exe pipeline_experiments\record_leaderboard_result.py `
  --run PE_run_1 --public-score 0.61234 `
  --submission-description PE_run_1
```

The record is stored at
`pipeline_experiments/runs/<experiment>/leaderboard_result.json` and is kept
separate from the offline validation results. No test labels are loaded by
these tools, and no automatic leaderboard submission is attempted.
