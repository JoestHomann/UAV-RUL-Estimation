# Model architecture study

Phase 2 compares several RUL architectures through the same UAV-grouped validation procedure. Hyperparameters are tuned within each architecture, but the pipeline does not choose an architecture winner. It saves comparable results and plots so the final architecture can be selected manually.

## Single entry point

[run_phase_2.py](run_phase_2.py) is the user-facing entry point for all seven
steps. It calls each existing step runner with the same Python interpreter,
stops on the first failure, and preserves the individual step artifacts.
Scientific calculations remain in their corresponding step folders.

Run the complete pipeline from the repository root:

    py 2_model_architecture_study\run_phase_2.py

When using the project virtual environment, use:

    .\.venv\Scripts\python.exe 2_model_architecture_study\run_phase_2.py

Inspect progress without modifying files:

    py 2_model_architecture_study\run_phase_2.py --status

TensorBoard monitoring is mandatory for every Phase 2 run. Start the dashboard
in a second terminal before or after starting the pipeline:

    .\.venv\Scripts\python.exe 2_model_architecture_study\tensorboard_monitoring\launch_tensorboard.py

Open the printed address, normally "http://localhost:6006". Phase 2 always
writes to the stable "tensorboard_monitoring/logs/" hierarchy; no monitoring
flag is required. The dashboard process remains separate from training so it
can be restarted without interrupting a model fit.

Resume from the expensive tuning stage while keeping completed work:

    py 2_model_architecture_study\run_phase_2.py --from-step 5

The default resume behavior skips completed Step 5 family/fold studies and
complete Step 6 family/fold evaluations. If a Step 6 family/fold was interrupted
between stochastic seeds, that family/fold is rerun as one complete unit.

Use "--through-step" to stop after a chosen step. For example, this prepares
the contract, both data adapters, and model registry without starting training:

    py 2_model_architecture_study\run_phase_2.py --through-step 4

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

    py 2_model_architecture_study\run_phase_2.py --from-step 5 --max-workers 4

Use "--max-workers 1" to fall back to strictly sequential studies (the
previous behavior), which also makes console output easier to read since
concurrent studies' printed progress interleaves in the same terminal. Each
subprocess caps its own CPU/BLAS thread pools so several studies running at
once do not oversubscribe the machine; this happens automatically and needs
no configuration.

### Staged runs

The full grid is large: the classical families (mean baseline, cycle-only
baseline, regularized linear, random forest, XGBoost) are inexpensive, while
the three neural families (MLP, TCN, LSTM) are the slowest part of both Step
5 and Step 6. Rather than waiting for everything at once, it is reasonable to
run the cheap families first and review results within hours:

    py 2_model_architecture_study\run_phase_2.py --from-step 5 --through-step 6 --family mean_baseline --family cycle_only_baseline --family regularized_linear --family random_forest --family xgboost

then start the neural families separately, for example overnight or in the
background:

    py 2_model_architecture_study\run_phase_2.py --from-step 5 --through-step 6 --family mlp --family tcn --family lstm

`--family` is accepted by the Step 5 and Step 6 runners directly (repeat the
flag for each family); "run_phase_2.py" itself does not filter by family, so
pass "--family" through "--from-step"/"--through-step" only when calling the
Step 5/6 scripts directly, or invoke them one stage at a time as shown above.
Since Step 5 and Step 6 already checkpoint per study, "--from-step 5" safely
resumes and only fills in whatever is still missing.

### GPU acceleration

Neural training (MLP, TCN, LSTM) automatically uses an available NVIDIA GPU;
no flag is required. Each Step 5/6 run prints "Neural training device: cuda"
or "Neural training device: cpu" once at startup so the active device is
visible in the log. XGBoost and Random Forest remain CPU-only
("n_jobs=1"), matching their existing single-threaded, deterministic
configuration. CUDA determinism is enforced the same way CPU determinism
already was ("torch.use_deterministic_algorithms(True)"); on a CUDA build
that lacks a deterministic kernel for some operation, PyTorch raises a
"RuntimeError" naming that operation rather than silently producing
non-reproducible results. This has not been exercised on real GPU hardware
as part of this change; before relying on a long GPU run, first do a small
"--family mlp --outer-fold 0" (or "tcn"/"lstm") dry run and confirm it
completes without that error.

The workflow begins in [`1_experiment_contract/`](1_experiment_contract/), which turns the human-readable TOML contract into a validated, deterministic JSON specification.

The next implemented step is the [tabular data adapter](2_tabular_data_adapter/README.md). It creates verified local copies of the Phase 1 tabular inputs and exposes shared feature-loading and UAV-fold selection methods for every tabular architecture.

The [sequence data adapter](3_sequence_data_adapter/README.md) creates causal padded telemetry windows and applies robust channel scaling fitted only on the active training UAVs.

The [model adapters](4_model_adapters/README.md) implement the baselines,
classical tabular estimators, and neural sequence estimators behind one common
fit, prediction, and persistence interface. Their generated registry records
which contract families are implemented and enabled.

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
contract order and never ranks architectures or writes a winner.

The [TensorBoard monitoring layer](tensorboard_monitoring/README.md) records
live optimization curves, candidate summaries, training and inference timing,
and completion state. Step 6 deliberately withholds locked predictive metrics;
Step 7 publishes them only after the complete locked-evaluation gate passes.

Install the Phase 2 dependencies from the repository root:

```powershell
py -m pip install -r 2_model_architecture_study\requirements.txt
```

Only dependencies needed by implemented steps are listed. Model adapters use
scikit-learn, XGBoost, PyTorch, and joblib; TensorBoard provides mandatory live
monitoring; Step 7 uses Matplotlib for the fixed comparison figures.
