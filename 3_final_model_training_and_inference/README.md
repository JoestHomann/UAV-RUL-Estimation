# Final model training and inference

Phase 3 validates the manually selected Phase 2 winner, performs one final
development-only configuration search, freezes the training contract, fits the
model on all 100 training UAVs, predicts the test UAVs, and verifies the Kaggle
submission.

The human-edited source of truth is
`1_winning_architecture_selection/phase_3_settings.toml`. The generated JSON,
model, prediction, and submission artifacts must not be edited manually.

Run all six steps from the repository root:

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
Forcing an upstream step invalidates all downstream completion manifests until
those steps are rerun.
The current TOML must exactly match the settings resolved by Step 1. Once Step
2 has written any output, a settings change requires a new settings version and
Phase 3 run number.

Run numbers are phase-local. The default settings read Phase 2 Run 3 from
`2_model_architecture_study/runs/run_3/` and write Phase 3 Run 1 under
`3_final_model_training_and_inference/runs/run_1/`.

Test features remain inaccessible until Steps 1-4 are complete. Step 5 never
loads a test target or calculates a test metric. Step 6 emits exactly the
columns `uav_id,RUL`, sorted by `uav_id`.
