# Architecture Run 10: inputs and execution

This study compares six standalone models using the feature definitions in
`other_pipelines/uav_rul_pipeline_v4 (1).py`. It does not use the script's blend,
its validation shortcut, or a q=0.55 correction.

## The 266 features

The script retains 22 nonconstant telemetry channels. For each sensor it creates
12 features: current value, difference from its first value, expanding mean,
expanding standard deviation, current value minus expanding mean, expanding
linear slope, and rolling means and standard deviations for windows 5, 10 and
20. Flight cycle and `log1p(flight_cycle)` add two more: **22 × 12 + 2 = 266**.
Missing initial standard deviations and slopes are filled with zero, exactly
as in the script. Each row uses only observations up to that row's cycle.
UAV identifiers, RUL labels, and terminal lifetime are never input features.

| Model | Features and input shape | Saved tuning source |
|---|---|---|
| XGBoost | All 266 features at the prediction cycle; no feature scaling | Architecture Run 9, standard XGBoost selections |
| Standard MLP | The same 266-feature endpoint row; training-only robust scaling; fully connected layers with batch normalization, ReLU and dropout | Architecture Run 3 |
| LSTM | 266 channels at each of the last 20 cycles | Architecture Run 7 |
| Multi-scale CNN | 266 channels at each of the last 20 cycles; convolutions run over time | Architecture Run 7 |
| Transformer | 266 channels per cycle; last 50 cycles in folds 0–3, last 100 in fold 4 | Architecture Run 3 |
| trajectory_DTW_KNN | The full observed trajectory of 266-feature rows; the existing retrieval algorithm downsamples each trajectory to at most 48 points | Architecture Run 5 |

Temporal networks retain their existing two age side inputs. These duplicate
flight cycle and log cycle already present among the 266 channels; they add no
new feature definitions. Histories shorter than the lookback are left-padded
with zero and accompanied by an explicit padding mask. Temporal channels are
scaled using medians and IQRs from the active training UAVs only, with standard
deviation and unit fallbacks for near-constant channels. DTW uses the same
scaling for queries and training references. Full training reference histories
are available to DTW; evaluation queries stop strictly at the observed cutoff.

## Comparison protocol

- Five shared, existing UAV-grouped outer folds. No evaluation UAV enters model
  fitting, preprocessing or the DTW reference library.
- Dense training: every observed training cycle, unit row weights, and fitting
  labels capped at 125, matching the simpler script. Longer training UAVs
  therefore contribute more rows. DTW instead keeps one reference per UAV.
- Reuse each family's saved **inner-selected configuration for the corresponding
  outer fold**. The complete values, source paths, and hashes are recorded in
  `runs/run_10/reporting/hyperparameter_sources.json`. No new architecture search.
- Keep the previous batch size, patience, epoch limit, lookback, optimizer
  settings and model parameters. Re-estimate training duration on a deterministic
  20% stopping split drawn only from the outer training UAVs; then refit on all
  outer training UAVs for that duration. Held outer labels never control stopping.
- Three model seeds (13, 37, 73); deterministic DTW runs once per fold. This gives
  80 evaluation cells and at most 155 fits including stopping fits and refits.
- Score raw RUL on historical endpoints and the nominal and unrestricted
  empirical-cutoff suites used by the recent campaign. Report R², RMSE, MAE,
  bias, fold results, and whole-UAV bootstrap comparisons against XGBoost.
- One worker, four CPU threads, automatic CUDA for neural models and XGBoost.
  Cell checkpoints support resuming; source/settings changes reject reuse.

The script's endpoint features are identical in definition across models, but
temporal models receive multiple historical rows. This compares architectures
with their appropriate history representations, not identical input information
budgets. Past hyperparameters were selected with other representations and are
not established optima for these new inputs. DTW previously tested only one
configuration. These UAVs and folds have been studied before; results are
exploratory and do not constitute a fresh test or a guaranteed Kaggle score.

## Run and resume

From the repository root:

```powershell
& .\.venv\Scripts\python.exe .\2_architecture_experiments\2_model_architecture_study\run_engineered_feature_study.py --check
& .\.venv\Scripts\python.exe .\2_architecture_experiments\2_model_architecture_study\run_engineered_feature_study.py
```

The second command also resumes interrupted training. Do not start a duplicate
while a queued or active instance holds `runs/run_10/run.lock`. Progress and
partial results are in `runs/run_10/reporting/`; model artifacts, fit audits and
neural learning curves are under `runs/run_10/cells/`. This study does not
automatically choose a winner, fit a final submission model, or submit to Kaggle.
