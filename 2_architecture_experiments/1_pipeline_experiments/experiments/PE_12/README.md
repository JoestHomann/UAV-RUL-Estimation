# PE_12: Test-Like Validation Selection

PE_12 uses the PE_9 development-versus-test propensity model to reweight only
already cross-fitted development predictions. It compares PE_3, PE_11, and Run
9 candidates. Test labels and locked targets are never read. A candidate may
win the weighted ranking only when ordinary development RMSE remains within the
configured guard and propensity weighting retains adequate effective sample
size.

Run PE_11 and architecture Run 9 first, then execute:

```powershell
& .\.venv\Scripts\python.exe `
  .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_12\run.py
```

