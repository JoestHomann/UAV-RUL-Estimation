# PE_run_3

Automatic feature-union, target-cap, ensemble/calibration, and conservative-loss
workflow with one frozen locked confirmation.

## Use

Edit only `settings.toml`, then run or resume the complete winner-propagating
workflow:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_run_3\run.py
```

Use `--list` to inspect every downstream script or `--status` to inspect saved
artifacts. Focused diagnostic targets include `PE3_feature_union`,
`PE3_cap_sensitivity`, `PE3_ensemble_calibration`, `PE3_severity_loss`, and
`PE3_final_ensemble`. A focused group run does not propagate winners to later
groups; the bare command owns that ordered workflow.

Outputs are grouped under
`2_architecture_experiments/1_pipeline_experiments/runs/PE_run_3/`, including
the flat `figures/` gallery.

## Settings

The group tables define candidate cells. `[experiment_workflows.PE_run_3]`
defines winner-selection and safety gates; `[promotions.PE3_final_ensemble]`
defines locked confirmation. `[execution].max_workers` controls concurrency.
