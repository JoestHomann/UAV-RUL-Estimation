# Phase 3 post-run reporting

Phase 3 Step 7 reads the shared artifacts of a completed run and creates the
same core figures regardless of which model family won. The default Phase 3
pipeline invokes this step automatically after submission verification.
It does not load locked data, test targets, or change the selected
configuration, fitted model, predictions, or submission.

Regenerate only the report from the repository root:

```powershell
.\.venv\Scripts\python.exe 3_final_model_training_and_inference\run_phase_3.py --from-step 7
```

The underlying `build_phase_3_report.py` command remains available for direct
development and verification use.

Outputs are written below
`runs/run_<n>/7_post_run_reporting/`. The report includes search progression,
candidate/fold stability, selected fold metrics, input alternatives, training
cost, the final training curve when a shared TensorBoard loss exists, and
target-free test prediction diagnostics.

The plots use only fields common to all registered model adapters. A family
without an iterative training curve still receives every applicable plot, and
the missing curve is recorded in `report_manifest.json` rather than treated as
a failure.

Verify compatibility with fixed-input baselines and sequence lookbacks:

```powershell
.\.venv\Scripts\python.exe 3_final_model_training_and_inference\7_post_run_reporting\verify_phase_3_reporting.py
```
