# PE_25: restricted TabPFN blend

PE_25 tests one rule: use a small TabPFN contribution only when Run 7 predicts
RUL at or below a selected threshold. Above the threshold it returns Run 7.
It reuses PE_24's completed inner and outer predictions and requires no model
training, TabPFN import, or checkpoint loading.

Within each of the 15 outer folds, the four corresponding inner OOF partitions
select a weight from `{0, 0.10, 0.25, 0.50}` and threshold from `{50, 75, 100}`.
Selection minimizes squared error with equal total weight per UAV. Exact ties
prefer the smaller weight, then smaller threshold. Only the control prediction
determines eligibility; true RUL is never an inference input. Zero weight keeps
the unchanged control. Exact endpoint, raw-label and nested UAV membership
checks reject incomplete or mismatched source predictions.

The practical screening requirements are 12/15 fold wins, at least 2% lower
mean-fold RMSE, pooled R2 at least 0.90, and a UAV-bootstrap 95% RMSE-change
interval wholly below zero. **Even passing is screening only:** the hypothesis
was selected after examining PE_24 and the same endpoints are reused. It needs
separate confirmation before final fitting. Run 7 remains the production model.

From the repository root:

```powershell
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_25\run.py
```

Use `--only validate_inputs` for source checks, `--list` for the commands, or
`--status` for completion. The comparison writes `runs/run_1/reporting/`:
`summary.csv`, `fold_metrics.csv`, `region_metrics.csv`, `promotion_decisions.csv`,
`method_predictions.csv.gz`, `selection_provenance.csv`, and `winner_manifest.json`.
`pre_registration.json` records settings and hashes of source artifacts/code.
Changing them requires a new `[pipeline].run`, even with `--force`. This
registration freezes the follow-up, not its already observed hypothesis source.

PE_25 does not alter PE_24, open locked evaluation, access test labels, or create
a submission. Its result does not gate PE_26, which tests a separate hypothesis.

Run 1 completed during implementation verification. Restricted blending reduced
mean RMSE from 10.3106 to 10.1143 (1.904%) and reached pooled R2 0.90674, but
won 10/15 folds and its UAV-bootstrap RMSE-change interval was
[-0.4695, +0.0307]. It missed the screening gate; Run 7 remains retained.

## User-requested exploratory Kaggle submission — 11 September 2026

A separate final-fit utility now prepares the explicitly requested leaderboard
trial without changing the screening verdict:

```powershell
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\build_pe25_exploratory_submission.py
```

The verified file is
`runs/run_1/exploratory_submission_v2/submission_PE25_restricted_tabpfn.csv`.
It contains 100 rows with `id,RUL` columns. The final rule selects a 25% TabPFN
weight when Run 7 predicts at most 100 cycles; above 100 it returns Run 7. It
routes 63 of the test UAVs to the blend. The rule uses the original grid and
equal-UAV weighting over all saved PE_24 outer development OOF predictions.
Its fitting score is not an additional held-out validation result.

The utility reuses the saved Run 7 model, checks its predictions against its
original submission, and fits the pinned TabPFN once on all 2,000 training
prefixes. The old Run 7 artifact lacks a subsequently introduced optional
`correction_strength` attribute; the utility restores its contract default of
1.0 in memory. The original model file remains unchanged. The first preparation
directory records the attempt that discovered this compatibility issue; v2
contains the completed preparation with its own immutable input registration.

Input hashes, the frozen rule, component predictions, and the submission SHA-256
are recorded alongside the CSV. No test labels are used and no Kaggle upload is
performed automatically. The 1.904% figure refers to historical validation RMSE
improvement, not an expected percentage-point increase in Kaggle R².
