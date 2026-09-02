# PE_10: Hybrid Temporal Representation

PE_10 compares a fixed hybrid CNN using the latest 20 raw cycles against the
same model using 20 pooled older-history bins plus the latest 20 raw cycles.
Both cells receive the same 298 engineered features, folds, target, calibration,
optimizer, and dense stride-1 endpoints. Locked data is never opened.

Run from the repository root:

```powershell
& .\.venv\Scripts\python.exe `
  .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_10\run.py
```

The automatic winner is written to
`runs/run_1/reporting/winner_manifest.json`. Architecture Run 8 reads that
manifest directly.
