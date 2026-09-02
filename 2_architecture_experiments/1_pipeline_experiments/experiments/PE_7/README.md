# PE_7: Leakage-Free OOF Stacking

## Status: blocked

Do not run PE_7. Temporal architecture Run 7 failed its development promotion
gate: after 19/20 studies, even an impossible zero-RMSE result for the remaining
LSTM fold could not bring its mean below the required 10.7. The run was stopped
for mathematical futility, not because of a software failure, and no eligible
temporal winner exists to stack. See
`2_architecture_experiments/2_model_architecture_study/7_architecture_comparison/run_7_conclusion.md`.

The workflow is retained for review and possible reuse by a future qualifying
temporal or hybrid model. Its original execution contract follows.

Prerequisites are a completed PE_6 and temporal architecture Run 7. Review the
exact chain with `run.py --list`, then run from the repository root:

```powershell
& .\.venv\Scripts\python.exe `
  .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_7\run.py
```

The workflow rejects duplicate or unmatched endpoints, cross-fits every fitted
meta-model by inner fold with disjoint UAVs, and does not open locked data.

If and only if `reporting/promotion_contract.json` exists after the development
gate, review it and run the separate `confirm_stacked_model.py` command. That
script requires explicit paths, writes one immutable locked confirmation, and
returns the existing completed manifest on repeated invocation. It is
intentionally absent from the default `run.py` chain.

```powershell
& .\.venv\Scripts\python.exe `
  .\2_architecture_experiments\1_pipeline_experiments\confirm_stacked_model.py `
  --contract .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_7\runs\run_1\reporting\promotion_contract.json `
  --sequence-manifest .\2_architecture_experiments\2_model_architecture_study\runs\run_7\3_sequence_data_adapter\artifacts\sequence_dataset_manifest.json `
  --specification .\2_architecture_experiments\2_model_architecture_study\runs\run_7\1_architecture_study_settings\artifacts\experiment_specification.json `
  --tree-locked-predictions .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_3\runs\run_1\PE3_final_ensemble\reporting\locked_predictions.csv.gz `
  --output-dir .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_7\runs\run_1\locked_confirmation
```
