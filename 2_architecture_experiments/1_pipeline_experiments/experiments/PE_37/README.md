# PE_37: unclear-channel add-one-in study

Use the ten degradation channels and four operating/context channels as a
fixed baseline, then add one unclear channel per configuration. Reuse v13's
original feature representation; no new transforms, tuning or model families.

| Arm | Unclear channels included | Features |
| --- | --- | ---: |
| baseline | None | 129 |
| add_02 | 02 | 132 |
| add_04 | 04 | 132 |
| add_05 | 05 | 132 |
| add_09 | 09 | 132 |
| add_10 | 10 | 132 |
| add_11 | 11 | 132 |
| add_12 | 12 | 132 |
| add_24 | 24 | 132 |

Each addition restores exactly its raw value, historical mean and historical
sample standard deviation, in the original v13 feature order. The four
context channels and Telemetry 07 stay unchanged in every arm.

## Protocol and verified reuse

The baseline is PE_35's `drop_all_eight` arm. Before importing its five saved
fold predictions, the runner verifies source/data hashes, the historical
feature contract, PE_35 runner hash, library versions, tree settings, target
cap, feature order, group memberships, all evaluation endpoints and labels,
prediction checksums, per-fold feature weights and sample-weight extrema.
Reuse fails on a mismatch; it does not silently retrain or substitute a
different baseline. PE_35 files are read only and remain unchanged.

Execution requires **40 new fits** (eight additions times five folds), with
five verified baseline fits imported. All models use v13's main XGBoost:
3,000 maximum trees, depth 5, learning rate 0.015, subsample and column
subsample 0.8, lambda 1, seed 0, one CPU thread, 100-round early stopping,
cap-125 fitting targets, all training cycles, and the existing sample/feature
weighting formulas. No early specialist, gate or hyperparameter search.

The original 153-column matrix supplies the features, then each arm selects
its 129 or 132 columns. Feature weights are computed on fitting UAVs only
and subset without recomputing tiers or changing the remaining weights.

The same five grouped folds have 72 fitting, eight early-stopping and 20
scoring UAVs each. Scoring UAVs never supply fitting or stopping labels.
The ten fixed cutoff seeds give 1,000 paired endpoints across 100 UAVs;
they are repeated evaluations, not independent model-training seeds.

Primary metrics use capped RUL. Uncapped and regional errors are separate.
Negative RMSE change means the added channel improves the baseline. Paired
bootstrap intervals resample UAVs (10,000 draws), with adjustment for **eight**
addition comparisons. These are development-informed fixed-feature tests;
results do not establish the performance of combinations of additions.

## Run manually

From the repository root:

```powershell
.venv/Scripts/python.exe -u 2_architecture_experiments/1_pipeline_experiments/experiments/PE_37/run.py
```

The command imports verified baseline checkpoints and starts the 40 new
fits. Repeat it to resume after interruption. No production model, locked
evaluation or submission changes.

After completion, create the presentation-style blue plot with no text:

```powershell
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_37/plot_results.py
```

Validation without importing baseline cells or training:

```powershell
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_37/run.py --check
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_37/test_additions.py
```

`--check` creates setup/registration artifacts only. All fingerprints,
including baseline provenance, are registered before execution. Changed
inputs or settings require a new `study.run`; incompatible baseline reuse
requires a separately matched baseline study, not bypassing the checks.

Results: `runs/run_1/reporting/report.md`, `paired_additions.csv`, summaries,
per-fold/seed/region metrics and predictions. The optional plot command writes
`addition_effects.png`, `addition_effects.svg` and `fit_diagnostics.csv`.
Baseline fit times in the diagnostic table describe the original PE_35 fits,
not new work performed by PE_37.
