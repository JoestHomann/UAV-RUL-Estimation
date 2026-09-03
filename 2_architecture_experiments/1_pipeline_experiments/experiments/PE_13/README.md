# PE_13: Uncertainty-Dependent Conservative Correction

PE_13 replaces a uniform safety subtraction with a correction proportional to
the disagreement among PE_11's independently seeded XGBoost and ExtraTrees
members. The multiplier is strongest for low predicted RUL. Its strength is
selected on three inner folds and applied to the fourth, so every reported
prediction remains cross-fitted.

Run PE_11 first, then execute:

```powershell
& .\.venv\Scripts\python.exe `
  .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_13\run.py
```

