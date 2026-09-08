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
