# PE_9: Domain Robustness

The diagnostic uses no test labels. It compares all-feature domain AUC against
a cutoff-only negative control, freezes deterministic shift rankings, and
recomputes target relevance inside each active training fold.

```powershell
& .\.venv\Scripts\python.exe `
  .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_9\run.py
```

Set `competition_rules_allow_unlabelled_test_adaptation = false` if the
competition disallows unsupervised test-distribution diagnostics; in that case
do not promote a PE_9 treatment.
