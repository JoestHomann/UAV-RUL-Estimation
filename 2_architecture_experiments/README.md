# Phase 2 architecture experiments

Phase 2 is split into two sibling workflows with separate configuration and
artifact ownership:

- [`1_pipeline_experiments`](1_pipeline_experiments/) contains reusable,
  question-driven experiments that may coordinate Phase 1, the architecture
  study machinery, and optional Phase 3 work. Each `PE_run_X` has one TOML and
  one execution entry point.
- [`2_model_architecture_study`](2_model_architecture_study/) contains the
  standalone seven-step architecture comparison pipeline and the shared data
  and model adapters used by pipeline experiments.

From the repository root, inspect their current state with:

```powershell
.\.venv\Scripts\python.exe 2_architecture_experiments\1_pipeline_experiments\run_experiments.py --status
.\.venv\Scripts\python.exe 2_architecture_experiments\2_model_architecture_study\run_phase_2.py --status
```

Both workflows retain their own `runs/` directory. Repository-relative paths
written before this subdivision are resolved through compatibility aliases so
existing checkpoints remain resumable.
