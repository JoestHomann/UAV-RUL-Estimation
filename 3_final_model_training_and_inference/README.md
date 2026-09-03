# Final model training and inference

Phase 3 validates the manually selected Phase 2 winner, performs one final
development-only configuration search, freezes the training contract, fits the
model on all 100 training UAVs, predicts the test UAVs, and verifies the Kaggle
submission.

The human-edited source of truth is
`1_winning_architecture_selection/phase_3_settings.toml`. The experiment
manager may instead pass a generated, resolved JSON settings file. In both
cases, the generated JSON, model, prediction, and submission artifacts must
not be edited manually.

Run all seven steps from the repository root. Step 7 automatically creates the
model-agnostic plots after submission verification:

```powershell
.\.venv\Scripts\python.exe 3_final_model_training_and_inference\run_phase_3.py
```

Optionally monitor the final search and all-UAV fit in a second terminal:

```powershell
.\.venv\Scripts\python.exe 3_final_model_training_and_inference\launch_tensorboard.py
```

Inspect progress without modifying files:

```powershell
.\.venv\Scripts\python.exe 3_final_model_training_and_inference\run_phase_3.py --status
```

Run the training/development-only implementation checks before opening the real
test features:

```powershell
.\.venv\Scripts\python.exe 3_final_model_training_and_inference\verify_phase_3_implementation.py
```

Resume from the final search after an interruption:

```powershell
.\.venv\Scripts\python.exe 3_final_model_training_and_inference\run_phase_3.py --from-step 2
```

To stop at the final test-access gate, run through Step 4 first and start Step 5
separately after reviewing the frozen contract and training manifest:

```powershell
.\.venv\Scripts\python.exe 3_final_model_training_and_inference\run_phase_3.py --through-step 4
.\.venv\Scripts\python.exe 3_final_model_training_and_inference\run_phase_3.py --from-step 5
```

The Optuna study preserves completed candidates in a run-local SQLite
checkpoint. Candidate fits are sequential so several folds do not compete for
one GPU. `--force` deliberately replaces completed work in requested Steps 2-6.
If Step 7 is included in the requested range, its report is regenerated from
the resulting artifacts. Forcing an upstream step invalidates all downstream
completion manifests until those steps are rerun.
The current TOML must exactly match the settings resolved by Step 1. Once Step
2 has written any output, a settings change requires a new settings version and
Phase 3 run number.

Run numbers are phase-local. The current default TOML writes Phase 3 Run 7
under `3_final_model_training_and_inference/runs/run_7/` and deploys the
development-selected `PE_11::residual_corrected` method confirmed by PE_12's
test-like ranking. The base component pair is frozen to the configuration
selected by the preceding Run 6 search. There is consequently one Phase 3
candidate rather than another search over the same component grid.

For every development fold, the Run 7 adapter partitions only that fold's
training UAVs into four internal groups. Each internal fit uses all available
training prefixes from three groups and predicts the established five
development endpoints for the fourth. Those leakage-safe OOF predictions fit
the XGBoost/ExtraTrees blend weight and shallow residual model. The six base
members are then refitted on the complete fold training side. Final training
repeats the same procedure across all 100 training UAVs before fitting the six
deployable members. Run 7 applies no additional quantile calibration because
PE_13 rejected the tested uncertainty-dependent safety policy for excessive
RMSE regression.

New Phase 3 runs can freeze two named policies from the same selected OOF
predictions and base model. Add the following to the new run's TOML; do not
edit a completed run's settings:

```toml
canonical_submission_policy = "accuracy_q50"

[[submission_policies]]
name = "accuracy_q50"
calibration = "conditional_quantile"
safety_offset = 0.0
non_overprediction_coverage = 0.50
calibration_prediction_bin_edges = [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0]
calibration_minimum_bin_rows = 10

[[submission_policies]]
name = "conservative_q55"
calibration = "conditional_quantile"
safety_offset = 0.0
non_overprediction_coverage = 0.55
calibration_prediction_bin_edges = [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0]
calibration_minimum_bin_rows = 10
```

Steps 5-6 then produce and independently regenerate
`submission_accuracy_q50.csv` and `submission_conservative_q55.csv` in
addition to canonical `submission.csv`. The canonical choice is fixed in the
TOML, never selected from repeated leaderboard feedback.

Test features remain inaccessible until Steps 1-4 are complete. Step 5 never
loads a test target or calculates a test metric. Step 6 emits exactly the
columns `id,RUL`, sorted by `id`; each Kaggle `id` is copied from the internal
`uav_id` prediction identifier.

Regenerate only the model-agnostic report and figures when needed:

```powershell
.\.venv\Scripts\python.exe 3_final_model_training_and_inference\run_phase_3.py --from-step 7
```

Step 7 uses the same shared artifacts for every selectable family. It
creates search, fold-stability, input-alternative, efficiency, final-training,
and target-free test-prediction figures without changing the selected model or
submission. A reporting failure leaves the verified Step 6 model and submission
valid and can be resumed independently. After success, the top-level run
manifest includes the report path. See the
[reporting README](7_post_run_reporting/README.md) for its verification command.
