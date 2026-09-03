# PE_11: Cross-Fitted Bagging and Residual Correction

PE_11 tests whether variance reduction and a small error model improve the
current calibrated tree blend. It regenerates XGBoost and ExtraTrees predictions
for seeds 13, 37, and 73 on development folds only. Blend weights and residual
corrections are learned on the other inner folds inside the same outer study;
the held-out UAVs are never used to fit their own correction.

```powershell
& .\.venv\Scripts\python.exe `
  .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_11\run.py
```

