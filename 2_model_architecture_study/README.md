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

Install the Phase 2 dependencies from the repository root:

```powershell
py -m pip install -r 2_model_architecture_study\requirements.txt
```

Only dependencies needed by implemented steps are listed. Model adapters use
scikit-learn, XGBoost, PyTorch, and joblib; Step 7 uses Matplotlib for the fixed
comparison figures.
