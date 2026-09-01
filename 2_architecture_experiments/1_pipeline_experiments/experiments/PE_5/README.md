# PE_5

PE_5 tests whether hard target capping is suppressing useful high-RUL
predictions. It holds the Run 5 `screened_drift_pruned` features, selected
ExtraTrees/XGBoost component configurations, 50/50 blend, UAV folds, model
seed, and q=0.55 conditional calibration fixed.

The four variants are:

- `hard_cap_125`: clip fitting labels at 125.
- `raw`: fit the original RUL labels.
- `weighted_raw`: retain raw labels but multiply weights above 125 by 0.25,
  then restore equal total weight per UAV.
- `soft_tail`: retain an invertible compressed tail above 125 with scale 0.50.

The old polynomial residual calibrator is deliberately not reused because it
was fitted specifically to capped-125 predictions. Each variant instead fits
its own q=0.55 correction from development OOF predictions only.

## Run

Edit only `settings.toml`, then execute or resume the complete experiment:

```powershell
& .\.venv\Scripts\python.exe `
  .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_5\run.py
```

Use `--list` to inspect the exact script chain, `--status` to inspect progress,
or `--force` to replace all four generated variants. Change `[pipeline].run`
to `run_2` before testing another set of target-policy values.

Upload-ready files are gathered under `runs/run_1/submissions/`. Development
metrics are diagnostic because the early/middle development scenarios are
bounded at RUL 125; Kaggle scores are required to compare behavior above 125.

## Run 1 results

All four submissions completed model-bundle reload verification and contain
100 unique IDs with finite, nonnegative RUL predictions.

| Variant | Development R2 | RMSE | Submission maximum | Kaggle R2 |
| --- | ---: | ---: | ---: | ---: |
| `hard_cap_125` | **0.8802** | **11.19** | 122.14 | **0.85609** |
| `soft_tail` | 0.5422 | 18.22 | 187.03 | 0.83609 |
| `weighted_raw` | 0.5268 | 18.78 | 186.03 | 0.83482 |
| `raw` | 0.5157 | 18.95 | 232.84 | 0.82866 |

Both development and Kaggle evidence favor the hard cap. Restoring progressively
more of the upper tail reduced the public score: soft tail lost 0.02000,
weighted raw lost 0.02127, and raw lost 0.02743 relative to the PE_5 cap
control. Preserve the cap for the current pipeline.

The PE_5 cap control is a controlled comparison within this experiment, not an
exact reproduction of Phase 3 Run 6. PE_5 deliberately excludes the older
capped-prediction residual calibrator; Run 6 retains that selected component
and reached public R2 0.86741.
