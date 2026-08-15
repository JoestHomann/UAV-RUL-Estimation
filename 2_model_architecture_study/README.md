# Model architecture study

Phase 2 compares several RUL architectures through the same UAV-grouped validation procedure. Hyperparameters are tuned within each architecture, but the pipeline does not choose an architecture winner. It saves comparable results and plots so the final architecture can be selected manually.

The workflow begins in [`1_experiment_contract/`](1_experiment_contract/), which turns the human-readable TOML contract into a validated, deterministic JSON specification.

The next implemented step is the [tabular data adapter](2_tabular_data_adapter/README.md). It creates verified local copies of the Phase 1 tabular inputs and exposes shared feature-loading and UAV-fold selection methods for every tabular architecture.

Install the Phase 2 dependencies from the repository root:

```powershell
py -m pip install -r 2_model_architecture_study\requirements.txt
```

Only the dependencies needed by implemented steps are listed. Model libraries will be added when the corresponding adapters are implemented.
