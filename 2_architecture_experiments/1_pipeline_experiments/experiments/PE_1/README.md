# PE_1

Original drift-pruned architecture-study baseline.

## Use

Edit only `settings.toml`, then run from the repository root:

```powershell
.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\experiments\PE_1\run.py
```

The command resumes existing artifacts by default. Use `--list` to inspect the
resolved command, `--status` to inspect completion, or
`--only baseline_architecture_study --force` to rebuild the completed target.
Outputs remain under
`2_architecture_experiments/1_pipeline_experiments/experiments/PE_1/runs/run_1/`.

## Settings

`[pipeline].run` selects the artifact folder. Change it from `run_1` to
`run_2` before starting another execution; the same `run.py` is used.
The `[experiments.PE_1]` table controls Phase 1 identity, Phase 2 models,
features, scenario/target profiles, candidate budget, seeds, and scope.
`[execution].max_workers` controls concurrent Phase 2 studies.
