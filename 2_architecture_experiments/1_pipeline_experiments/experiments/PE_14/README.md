# PE_14: complete-pipeline validation

This refits the frozen Run 6 and Run 7 systems across three independently generated five-fold UAV partitions and evaluates newly generated test-cutoff assignments. Run 6's residual calibrator and Run 7's residual head are fitted from inner OOF predictions using only the active 80 training UAVs. The nominal suite keeps RUL 1–125; the stress suite removes that label-dependent upper bound. It never reads test labels or creates submissions.

```powershell
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_14\run.py
```

The run checkpoints each system/profile/fold in `runs/run_1/reporting/fold_predictions.csv`.

Run 7 achieved pooled nominal R² 0.9011 and improved mean RMSE by 3.36%, but
won 11/15 folds against the required 12/15 and its bootstrap interval crossed
zero. Unrestricted-support pooled R² was 0.6188. No candidate was promoted.
