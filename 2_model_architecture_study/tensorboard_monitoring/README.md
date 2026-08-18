# Phase 2 TensorBoard monitoring

## Purpose

This folder contains the complete TensorBoard integration for Phase 2. The
numbered pipeline steps contain only the small calls needed to report training
progress; no step imports TensorBoard itself.

TensorBoard is a live monitoring layer and nothing more. The Step 5, Step 6,
and Step 7 CSV and JSON artifacts are the authoritative experiment results, so
this layer logs only what is worth watching while a run is still going: curves
over an optimization axis, and the tuning search curve. Every finished value --
final metrics, age-band breakdowns, timings, row and parameter counts,
completion flags, the whole Step 7 comparison -- is read from the artifacts
that already record it.

## Start the dashboard

Open a second PowerShell terminal in the repository root and run:

    .\.venv\Scripts\python.exe 2_model_architecture_study\tensorboard_monitoring\launch_tensorboard.py

Open the printed address, normally "http://localhost:6006". To use another
port:

    .\.venv\Scripts\python.exe 2_model_architecture_study\tensorboard_monitoring\launch_tensorboard.py --port 6007

The dashboard process is independent from the training pipeline. It can stay
open while Phase 2 is started, interrupted, and resumed.

## Logged information

| Scope | Tag | Written |
| --- | --- | --- |
| One fit | "train/loss" | Weighted training MSE every epoch (neural) or training RMSE every tenth boosting round and the final round (XGBoost) |
| One fit | "val/rmse" | Development RMSE on the same axis, whenever a development fold exists |
| One Step 5 study | "search/candidate_rmse" | Mean inner RMSE at step = candidate number, one point per completed candidate |
| One Step 5 study | "search/candidate_NNN" | That candidate's hyperparameters as text, so a point on the curve is readable without opening the CSV |

Nothing else is logged. In particular there is no tag for the best score so
far (it is the running minimum of "val/rmse"), early-stopping patience (it is
the flat stretch after that minimum), the learning rate (constant while no
schedule is attached), per-epoch timing, or any value produced after a fit
finished.

Atomic-fit families -- Ridge, Elastic Net, Random Forest, RBF SVR -- expose no
optimization iterations and therefore publish no curve. Their results are in
the Step 5 and Step 6 artifacts like every other family's.

## Step 5 fit curves are opt-in

A Step 5 study fits (candidate budget x inner fold count) models, so writing a
curve for each one fills a single scalar panel with hundreds of tag prefixes.
During a normal search the candidate curve is the useful view, so per-fit
curves are switched off. Set the environment variable to turn them on while
debugging one architecture:

    $env:PHASE2_TENSORBOARD_FIT_CURVES = "1"

Subprocesses dispatched by "run_phase_2.py" inherit the variable, and the
pipeline prints a line at startup when it is set. Step 6 retrains are few and
each one is a deliverable, so they always publish "train/loss".

## The locked-metric boundary

Step 6 is never given the locked validation dataset, so it cannot publish a
locked predictive metric even by accident. Its runs show a training-loss curve
and nothing else. Locked results appear only in the Step 7 artifacts, after the
complete Step 6 gate has passed; Step 7 writes no TensorBoard events at all.

## Run organization

Step 5 fit curves:

    logs\step_5\architecture\outer_fold_N\fit_progress

Step 5 search curve:

    logs\step_5\architecture\outer_fold_N\study_progress

Step 6 fit curves:

    logs\step_6\architecture\outer_fold_N\fit_progress

One writer is shared by every fit inside a study, so the generated directory
count is fixed per study instead of growing with the candidate budget. Each
fit's curves stay separately selectable through its tag prefix, for example
"^candidate_007/" or "^seed_013/" in TensorBoard's filter box.

These paths contain no timestamp or generated run identifier. Restarting the
same logical study replaces its visible events; completed studies that the
Phase 2 resume logic skips are left unchanged.

## Failure behavior

A TensorBoard write, flush, or writer-creation failure prints one warning to
standard error and disables monitoring for the remainder of that study.
Training continues. Losing an event file costs a live curve; aborting a
multi-hour retraining over a flush error costs the run, and the authoritative
artifacts are written by the pipeline steps themselves.

A missing TensorBoard installation is still a hard failure, checked once before
any expensive work starts.

## Files

- "monitoring.py" owns dependency checks, safe log paths, SummaryWriter
  lifecycle, the four tags above, and the logging intervals.
- "xgboost_callback.py" translates official boosting-round callback data into
  the same two tags the neural loop reports.
- "launch_tensorboard.py" starts the dashboard on the shared log directory.
- "verify_tensorboard_monitoring.py" fits tiny models and asserts that these
  tags, and no others, are readable afterwards.
- "logs/" contains generated events and is ignored by Git.

"monitoring.py" also holds "calculate_regression_metrics". That function is not
a monitoring helper: Step 5 selects candidates on the RMSE it returns. It stays
here because that is where it has always lived and moving it would touch the
selection path for no benefit.
