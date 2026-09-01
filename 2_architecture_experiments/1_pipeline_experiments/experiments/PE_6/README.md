# PE_6: Dense Temporal Sample Construction

Run from the repository root:

```powershell
& .\.venv\Scripts\python.exe `
  .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_6\run.py
```

Use `--list` to review the exact script chain and `--status` to inspect saved
progress. The workflow compares sparse and dense sequence supervision at a
fixed lookback, opens the lookback comparison only when dense sampling passes
its development gate, and never runs locked evaluation.
