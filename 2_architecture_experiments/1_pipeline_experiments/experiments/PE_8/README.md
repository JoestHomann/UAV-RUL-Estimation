# PE_8: Personalized Degradation Onset

This development-only experiment holds the XGBoost and ExtraTrees
configurations fixed and changes only their fold-training target. Each onset
detector is fitted after the active inner-fold UAV split, and its provenance
records zero overlap with validation UAVs.

```powershell
& .\.venv\Scripts\python.exe `
  .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_8\run.py
```
