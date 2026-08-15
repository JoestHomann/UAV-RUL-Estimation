# Phase 2 TensorBoard monitoring

## Purpose

This folder contains the complete TensorBoard integration for Phase 2. Event
logging is mandatory whenever the Phase 2 pipeline trains or compares models.
The numbered pipeline steps contain only the small calls needed to describe a
run, report progress, and close it.

TensorBoard is a monitoring and visualization layer. The Step 5, Step 6, and
Step 7 CSV and JSON artifacts remain the authoritative experiment results.

## Start the dashboard

Open a second PowerShell terminal in the repository root and run:

    .\.venv\Scripts\python.exe 2_model_architecture_study\tensorboard_monitoring\launch_tensorboard.py

Open the printed address, normally "http://localhost:6006". To use another
port:

    .\.venv\Scripts\python.exe 2_model_architecture_study\tensorboard_monitoring\launch_tensorboard.py --port 6007

The dashboard process is independent from the training pipeline. It can stay
open while Phase 2 is started, interrupted, and resumed.

## Logged information

| Scope | Live TensorBoard data |
| --- | --- |
| Every Step 5 fit | Architecture, folds, candidate, seed, representation, feature set or lookback, hyperparameters, row counts, feature count, training time, inference time, parameter count when available, and overall and age-band development RMSE, MAE, R2, and bias |
| Neural models | Weighted training MSE, development RMSE when permitted, learning rate, seconds per epoch, best RMSE, and early-stopping patience every epoch |
| XGBoost | Weighted training RMSE, development RMSE when permitted, and seconds per boosting iteration every ten rounds plus the final round |
| Atomic-fit models | Start and completion state, dimensions, hyperparameters, total training time, inference time, and final permitted metrics |
| Step 5 study | Mean candidate RMSE, fold variation, timing, retraining duration, and the automatically selected configuration within that family |
| Step 6 | Training progress, architecture, fold, seed, fixed duration, dimensions, timing, and completion state only |
| Step 7 | Final overall and age-band locked RMSE, MAE, R2, bias, uncertainty intervals, seed variation, and efficiency for every architecture |

Step 6 intentionally does not publish locked predictive metrics while runs are
still in progress. Those values appear only after the complete Step 6 gate has
passed and Step 7 has calculated the fixed comparison.

Weights, gradients, raw prediction arrays, feature histograms, and system
resource sampling are not logged. Excluding them keeps event files compact and
avoids adding model-specific diagnostic code or another runtime dependency.

## Run organization

Step 5 fit paths follow:

    logs\step_5\architecture\outer_fold_N\candidate_N\inner_fold_N

Step 6 paths follow:

    logs\step_6\architecture\outer_fold_N\seed_N

Final Step 7 paths follow:

    logs\step_7\final_comparison\architecture

These paths contain no custom timestamp or generated run identifier. Restarting
the same logical fit replaces its visible events. Completed fits that the Phase
2 resume logic skips are left unchanged.

## Files

- "monitoring.py" owns dependency checks, safe log paths, SummaryWriter
  lifecycle, scalar tags, logging intervals, and final comparison publishing.
- "xgboost_callback.py" translates official boosting-round callback data into
  the same neutral scalar interface used by the other training code.
- "launch_tensorboard.py" starts the dashboard on the shared log directory.
- "verify_tensorboard_monitoring.py" performs a small isolated writer and event
  readability check without training the architecture study.
- "logs/" contains generated events and is ignored by Git.

Any TensorBoard write or flush failure stops the current fit. Phase 2 never
silently falls back to unmonitored training.
