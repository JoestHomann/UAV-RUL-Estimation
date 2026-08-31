# Model architecture study

Phase 2 compares several RUL architectures through the same UAV-grouped validation procedure. Hyperparameters are tuned within each architecture, but the pipeline does not choose an architecture winner. It saves comparable results and plots for the manual winning-architecture decision in Phase 3 Step 1.

## Single entry point

[run_phase_2.py](run_phase_2.py) is the user-facing entry point for all seven
steps. It calls each existing step runner with the same Python interpreter,
stops on the first failure, and preserves the individual step artifacts.
Scientific calculations remain in their corresponding step folders.

Run the complete pipeline from the repository root:

    py 2_architecture_experiments\2_model_architecture_study\run_phase_2.py

When using the project virtual environment, use:

    .\.venv\Scripts\python.exe 2_architecture_experiments\2_model_architecture_study\run_phase_2.py

Inspect progress without modifying files:

    py 2_architecture_experiments\2_model_architecture_study\run_phase_2.py --status

TensorBoard is available for every Phase 2 run. Start the dashboard in a
second terminal before or after starting the pipeline:

    .\.venv\Scripts\python.exe 2_architecture_experiments\2_model_architecture_study\tensorboard_monitoring\launch_tensorboard.py

It opens on the run that "run_number" currently selects, whose events live in
"runs/run_<n>/tensorboard_logs/"; pass "--all-runs" to compare several runs in
one view. The dashboard process remains separate from training so it can be
restarted without interrupting a model fit.

Resume from the expensive tuning stage while keeping completed work:

    py 2_architecture_experiments\2_model_architecture_study\run_phase_2.py --from-step 5

The default resume behavior skips completed Step 5 family/fold studies and
complete Step 6 family/fold evaluations. If a Step 6 family/fold was interrupted
between stochastic seeds, that family/fold is rerun as one complete unit.

### Numbered runs

Steps 5, 6 and 7 write every artifact under one numbered run folder:

    2_architecture_experiments\2_model_architecture_study\runs\run_<n>\5_inner_model_selection\
    2_architecture_experiments\2_model_architecture_study\runs\run_<n>\6_locked_outer_evaluation\
    2_architecture_experiments\2_model_architecture_study\runs\run_<n>\7_architecture_comparison\

"n" is the "run_number" in the architecture study settings. Nothing advances
it automatically, which is the whole point: stopping Phase 2 and resuming it
later resolves to the same folder, so the resumed work joins the work that
already finished and a partially complete run can never be split across two
run numbers. Increment "run_number" by hand only when a genuinely new run
should begin -- that starts the new run from nothing rather than resuming.

Steps 1 to 4 keep their fixed "artifacts" directories. Their outputs are the
validated settings, the copied data adapters, and the model registry: shared by
every run and reproduced identically from the same inputs, so a per-run copy
would duplicate large datasets without adding traceability.

"--status" and the pipeline banner both print the active run folder. Each
step's "--output-dir" and its upstream input flags still accept explicit paths,
which is how one run can be pointed at another run's Step 5 or Step 6 results.

Confirm the layout without running any training:

    py 2_architecture_experiments\2_model_architecture_study\verify_run_layout.py

Step 7 additionally saves "architecture_study_settings.csv" beside its result
tables, so a finished run records the configuration that produced it even after
the settings file has moved on.

### Conditional conservative calibration

Phase 2 Run 6 enables `prediction_policy.calibration =
"conditional_quantile"` with residual quantile `0.55`. For every candidate,
Step 5 fits the prediction-dependent correction for each validation fold using
only the other inner folds. Candidate metrics therefore describe calibrated
predictions without letting a row influence its own correction. Step 6 fits
one curve from the selected candidate's complete Step 5 OOF residuals and
applies it to locked predictions. Prediction tables retain the uncalibrated
value and subtracted adjustment alongside the final prediction.

The configurable fields are `non_overprediction_coverage`,
`calibration_prediction_bin_edges`, and `calibration_minimum_bin_rows`.
Setting `calibration = "none"` restores uncalibrated behavior.

Use "--through-step" to stop after a chosen step. For example, this prepares
the settings, both data adapters, and model registry without starting training:

    py 2_architecture_experiments\2_model_architecture_study\run_phase_2.py --through-step 4

"--force" deliberately replaces completed Step 5 and Step 6 work and should be
used only when a full rerun is intended. Do not start this entry point while an
older Phase 2 training command is still running, because both processes would
write to the same checkpoint directory.

### Parallel Step 5/6 studies

Every family/outer-fold combination in Step 5 and Step 6 is an independent
study with its own checkpoints, TensorBoard log directory, and inner UAV
folds. "run_phase_2.py" runs up to "--max-workers" of these studies at the
same time, each as its own subprocess, and defaults to the number of
available CPU cores:

    py 2_architecture_experiments\2_model_architecture_study\run_phase_2.py --from-step 5 --max-workers 4

Use "--max-workers 1" to fall back to strictly sequential studies (the
previous behavior), which also makes console output easier to read since
concurrent studies' printed progress interleaves in the same terminal. Each
subprocess caps its own CPU/BLAS thread pools so several studies running at
once do not oversubscribe the machine; this happens automatically and needs
no configuration.

### Staged runs

The full grid is large: the classical families (mean baseline, cycle-only
baseline, regularized linear, random forest, Extra Trees, and XGBoost) are
inexpensive, while
the neural families (MLP, TCN, multi-scale CNN, sensor-graph TCN, LSTM, and
Transformer) plus CatBoost are the slowest part of both Step 5 and Step 6.
Rather than waiting for everything at once, it is reasonable to run the cheap
Step 5 families first and review results within hours:

    py 2_architecture_experiments\2_model_architecture_study\5_inner_model_selection\run_inner_model_selection.py --family mean_baseline --family cycle_only_baseline --family regularized_linear --family random_forest --family extra_trees --family xgboost

then start CatBoost and the neural families separately, for example overnight
or in the background:

    py 2_architecture_experiments\2_model_architecture_study\5_inner_model_selection\run_inner_model_selection.py --family catboost
    py 2_architecture_experiments\2_model_architecture_study\5_inner_model_selection\run_inner_model_selection.py --family multiscale_cnn
    py 2_architecture_experiments\2_model_architecture_study\5_inner_model_selection\run_inner_model_selection.py --family sensor_graph_tcn

`--family` is accepted by the Step 5 and Step 6 runners directly (repeat the
flag for each family); "run_phase_2.py" itself does not filter by family. Run
all Step 5 families before Step 6, because the locked-evaluation gate requires
the complete Step 5 manifest. Since both stages checkpoint per study,
"--from-step 5" safely resumes and only fills in whatever is still missing.

### GPU acceleration

Neural training and XGBoost automatically use an available
NVIDIA GPU; no flag is required. Each relevant Step 5/6 run prints its neural
and XGBoost device choices at startup so the active devices are visible in the
log. XGBoost still uses one host worker ("n_jobs=1"); Random Forest, Extra
Trees, and CatBoost remain CPU-only. Neural CUDA determinism is enforced the same way CPU determinism
already was ("torch.use_deterministic_algorithms(True)"); on a CUDA build
that lacks a deterministic kernel for some operation, PyTorch raises a
"RuntimeError" naming that operation rather than silently producing
non-reproducible results. This has not been exercised on real GPU hardware
as part of this change; before relying on a long GPU run, first do a small
"--family multiscale_cnn --outer-fold 0" dry run and confirm it
completes without that error.

The workflow begins in [`1_architecture_study_settings/`](1_architecture_study_settings/), which turns the human-readable TOML settings into a validated, deterministic JSON specification.

The next implemented step is the [tabular data adapter](2_tabular_data_adapter/README.md). It creates verified local copies of the Phase 1 tabular inputs and exposes shared feature-loading and UAV-fold selection methods for every tabular architecture.

The [sequence data adapter](3_sequence_data_adapter/README.md) creates causal padded telemetry windows and applies robust channel scaling fitted only on the active training UAVs.

The [trajectory data adapter](3_trajectory_data_adapter/README.md) reuses the
verified Step 3 inputs to expose variable-length causal queries and complete
run-to-failure reference trajectories from active training UAVs only. The
optional Run 4 trajectory DTW-kNN family consumes this interface; the other
Run 4 sequence models continue to use fixed-window inputs.

The [model adapters](4_model_adapters/README.md) implement the baselines,
classical tabular estimators, trajectory retrieval, and neural sequence estimators behind one common
fit, prediction, and persistence interface. Their generated registry records
which settings families are implemented and enabled.

The [inner model-selection runner](5_inner_model_selection/README.md) performs
automatic tuning separately inside each enabled family and outer fold. It uses
only inner UAV folds and development scenarios, writes per-study checkpoints,
and never ranks architectures.

The [locked outer-evaluation runner](6_locked_outer_evaluation/README.md)
re-trains completed Step 5 selections and saves held-out predictions, models,
and efficiency facts. A mandatory completion gate prevents it from loading any
locked split while Step 5 remains partial.

The [architecture comparison](7_architecture_comparison/README.md) calculates
the fixed metrics, paired whole-UAV bootstrap intervals, reliability views,
seed stability, efficiency summaries, and comparison figures. It preserves the
settings order and never ranks architectures or writes a winner. The completed
comparison is the final Phase 2 output and the input to Phase 3 Step 1.

The [TensorBoard monitoring layer](tensorboard_monitoring/README.md) records
only what is worth watching while a run is still going: a "train/loss" and
"val/rmse" curve per fit, and a "search/candidate_rmse" curve per Step 5 study.
Everything a run produced once it finished is read from the authoritative CSV
and JSON artifacts instead. Step 6 is never given locked targets, and Step 7
writes no events at all.

Install the Phase 2 dependencies from the repository root:

```powershell
py -m pip install -r 2_architecture_experiments\2_model_architecture_study\requirements.txt
```

Only dependencies needed by implemented steps are listed. Model adapters use
scikit-learn, XGBoost, PyTorch, and joblib; TensorBoard provides live training
monitoring; Step 7 uses Matplotlib for the fixed comparison figures.
