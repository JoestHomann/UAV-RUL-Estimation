# PE_7: Leakage-Free OOF Stacking

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
