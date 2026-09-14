# PE_36: v13 XGBoost operating-condition ablation

Test the four context/operating-condition channels identified in
`1_dataset_construction/common.py`: telemetry 01, 06, 18 and 26.

| Configuration | Removed channels | Features |
| --- | --- | ---: |
| baseline | None | 153 |
| drop_01 | 01 | 150 |
| drop_06 | 06 | 150 |
| drop_18 | 18 | 150 |
| drop_26 | 26 | 150 |
| drop_all_four | 01, 06, 18, 26 | 141 |

Each removal deletes the channel's raw value, historical mean and historical
sample standard deviation. All eight unclear channels remain included. Every
other v13 feature, including Telemetry 07, remains unchanged.

## Fixed protocol

This is the same main-XGBoost protocol as PE_35: v13's 153-feature contract,
frozen historical tiers, cap-125 fitting targets, 3,000 maximum trees, depth 5,
learning rate 0.015, row/column subsampling 0.8, lambda 1, model seed 0,
one CPU thread, and 100-round early stopping. No hyperparameter search,
specialist, blend or additional features.

All six arms share five deterministic UAV-grouped folds, the same inner 10%
UAV holdout for early stopping, and ten test-cutoff assignments per UAV
(1,000 paired endpoints). Each fold has 72 fitting, eight stopping and 20
scoring UAVs. Feature weights use fitting UAVs only; test-cutoff sample
weighting uses v13's formula. The test file supplies only UAV IDs and cycles.
Models are not refit after early stopping, as in PE_35.

Primary scores use capped RUL; uncapped and error-region metrics are reported
separately. Paired bootstrap intervals resample UAVs with 10,000 draws and the
same bootstrap seed as PE_35. Multiplicity adjustment covers the **five**
removal comparisons. This is a development screen with fixed historical
tiers, not independent confirmation or proof of channel necessity.

## Run manually

From the repository root, start the **30 model fits**:

```powershell
.venv/Scripts/python.exe -u 2_architecture_experiments/1_pipeline_experiments/experiments/PE_36/run.py
```

The command resumes completed, checksummed fits if interrupted. PE_35's
results are preserved; PE_36 fits its own matched baseline. No production
model or submission is changed.

After training finishes, create the chart and fit-diagnostics table:

```powershell
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_36/plot_results.py
```

Validation only, with **no training**:

```powershell
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_36/run.py --check
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_36/test_ablation.py
```

The check validates feature counts/order, causal prefix summaries, disjoint
UAV partitions and matched endpoints. It saves setup artifacts under
`runs/run_1/`; it does not create fitted cells or claim study completion.
Source, data, settings and runtime fingerprints guard resuming; change
`study.run` before executing a modified scientific configuration.

Results will be written to `runs/run_1/reporting/report.md`, with per-channel
effects in `paired_ablation.csv`, all predictions and per-fold/seed/region
metrics. The optional plot command writes `ablation_effects.png`, its SVG
version, and `fit_diagnostics.csv`.
