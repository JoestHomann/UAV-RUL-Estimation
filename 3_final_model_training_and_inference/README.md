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

Run numbers are phase-local. The current default TOML reads the locked-confirmed
`PE3_final_ensemble` policy and its development-only `PE3_features_drift`
component source, then writes Phase 3 Run 6 under
`3_final_model_training_and_inference/runs/run_6/`. The promoted policy keeps
the 50/50 blend and residual calibrator frozen. Step 2 searches the complete
`5 ExtraTrees x 5 XGBoost` grid of development-selected component
configurations, for 25 candidates total. Its repository-relative artifact
paths make the no-argument entry point sufficient. A future standalone Phase 3
TOML may omit those optional paths and use the numbered Phase 2 default.

Run 6 additionally enables the PE4 `q=0.55` conditional conservative
calibration. Step 2 evaluates it cross-fold and fits the final correction curve
only from the selected candidate's complete development OOF residuals. Step 3
freezes that curve in the training contract, and Steps 5-6 apply and regenerate
the same subtraction-only transformation. Test labels are never loaded, and
both the policy parameters and learned curve remain inspectable JSON data.

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
