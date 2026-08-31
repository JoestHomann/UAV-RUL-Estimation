# PE_run_2

Pipeline-structure and degradation-learning studies. PE1 is imported only as
the frozen current/raw control for the target/scenario comparison.

## Use

Edit only `settings.toml`, then run or resume all seven studies:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_run_2\run.py
```

Inspect first with `--list` or `--status`. Run selected studies by repeating
`--only NAME`; available top-level names are:

```text
target_scenario_2x2
signal_family_ablation
failure_cycle_target
baseline_normalization
fault_mode
signal_compression
dense_prefix_training
```

For example:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_run_2\run.py --only signal_family_ablation
```

Outputs are grouped under
`2_architecture_experiments/1_pipeline_experiments/runs/PE_run_2/`.

## Settings

The group tables define controls, treatments, and reporting. Each
`[experiments.*]` table defines one scientific cell. Edit
`[execution].max_workers` once to control concurrency for the whole run.
