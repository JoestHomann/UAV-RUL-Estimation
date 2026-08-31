# PE_run_4

Conditional conservative calibration of Phase 3 Run 5 predictions. This run
cross-fits subtraction-only correction curves and does not retrain a model.

## Use

Edit only `settings.toml`, then run or resume:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_run_4\run.py
```

Use `--list` to inspect the command, `--status` to inspect completion, or
`--only conditional_safety_calibration --force` to rebuild it. Outputs are
under `2_architecture_experiments/1_pipeline_experiments/runs/PE_run_4/`.

## Settings

`[conditional_calibration_workflows.PE_run_4]` controls the source Phase 3 run,
candidate quantiles, prediction bins, minimum rows, and R2 tolerance.
