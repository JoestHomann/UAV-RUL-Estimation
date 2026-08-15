# Model architecture study

Phase 2 compares several RUL architectures through the same UAV-grouped validation procedure. Hyperparameters are tuned within each architecture, but the pipeline does not choose an architecture winner. It saves comparable results and plots so the final architecture can be selected manually.

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

Install the Phase 2 dependencies from the repository root:

```powershell
py -m pip install -r 2_model_architecture_study\requirements.txt
```

Only dependencies needed by implemented steps are listed. Step 4 currently
uses scikit-learn, XGBoost, PyTorch, and joblib.
