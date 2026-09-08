# PE_26: short-history specialist

PE_26 compares Run 7 with a specialist fitted on existing `current20` prefixes
whose observed history is at most 100 cycles. Tree hyperparameters, member
seeds, feature set, cap-125 fitting targets, raw evaluation targets, and the
residual-head recipe remain fixed. Prefixes are filtered by observed cutoff,
not their true RUL, and the remaining weights sum to one per training UAV.
No new or future-dependent features are computed.

The specialist's residual calibration uses only the active training UAVs'
existing development-like endpoints with cutoff at most 100. Not every UAV has
an eligible calibration endpoint; preflight reports coverage for every job and
requires enough eligible UAVs for the unchanged four-fold internal OOF fit.
Each calibration UAV is excluded from the base fits that predict its endpoints.
Calibration retains equal total weight per represented UAV.

Three new split seeds (20261107, 20261117, 20261127) each have five outer folds
and four inner folds inside each outer training set. Every evaluation job
refits the complete control and specialist. The 75 jobs contain **150 complete
ensemble fits**, each including its own internal calibration fits.

Inside each outer fold, inner OOF predictions select specialist weight from
`{0, 0.25, 0.50, 1}` using equal-UAV squared error over all endpoints. The
combined prediction is `control + weight * (specialist - control)` only for
histories at most 100 cycles. Longer histories return the exact control; the
specialist is never queried there. The zero-weight option retains Run 7.

Only the combined model can pass the promotion gate: at least 2% lower
mean-fold RMSE, 12/15 fold wins, pooled R2 at least 0.90, and a UAV-bootstrap
95% RMSE-change interval wholly below zero. The hard specialist route is a
diagnostic anchor. Report short-history and true-RUL-band accuracy and
overprediction separately. New splits of these previously examined UAVs are
robustness checks, not a fresh independent test set. A passing result authorizes
consideration for final fitting; this runner never replaces Run 7 or submits.

From the repository root:

```powershell
# Validate all inputs and fold coverage without training.
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_26\run.py --only validate_inputs

# Run or resume the full experiment.
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_26\run.py
```

`--list` prints commands; `--status` shows completion. Each completed control
or specialist fit is checkpointed atomically to
`runs/run_1/reporting/fold_predictions.csv`. Resuming verifies exact endpoints,
labels and finite predictions before skipping a fit. Partially written temporary
files never replace a completed checkpoint. `--force` restarts fitting with
the same registered recipe; changed settings or source/code hashes require a
new `[pipeline].run` directory.

Reports include `input_coverage.csv`, split assignments, selection provenance,
pooled and mean-fold metrics, band/short-history metrics, bootstrap decisions,
and a winner manifest. No locked data, test inputs or test labels are loaded.
PE_25 need not pass or finish before this experiment can run.
