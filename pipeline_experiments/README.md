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

Phase 1 creates a named run under
`1_dataset_construction/runs/<phase_1_run_name>/`. Phase 2 copies the selected
inputs and writes its resolved settings under
`pipeline_experiments/runs/<experiment>/phase2/`, while Steps 5-7 use the
configured numbered Phase 2 run folder. Phase 3 uses the configured numbered
Phase 3 run folder. All generated manifests retain the exact paths used by
that experiment.

The launcher resumes Phase 2 and Phase 3 checkpoints. A partial Phase 1 or
Phase 2 run can be continued by running the same command again. Use new
`phase_1_run_name`, `phase_2_run_number`, and `phase_3_run_number` values for a
new scientific experiment. `--force` is intended for rerunning the selected
Phase 3 range; it does not change the TOML catalog.

## Edit the catalog

Copy an `[experiments.*]` block, give it a new name and new run identities,
then edit the question-specific fields:

- `phase_1_profile`, `scenario_profile`, and `prefix_variant` choose the data
  construction policy;
- `architectures`, `feature_set`, `candidate_budget`, and seeds choose the
  Phase 2 study;
- `target_profile` and `prediction_profile` choose raw/capped targets and
  symmetric/conservative fitting;
- the Phase 3 fields enable optional final selection, training, inference,
  submission verification, and model-agnostic reporting.

The manager does not create an implicit Cartesian product. Every block is one
named scientific question, which keeps leaderboard submissions and result
interpretation straightforward.

## Compare and record scores

After one or more Phase 2 studies finish:

```powershell
.venv\Scripts\python.exe pipeline_experiments\compare_experiments.py
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
