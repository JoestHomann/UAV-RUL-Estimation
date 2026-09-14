# PE_35: individual unclear-channel ablation with v13 XGBoost

Compare the unchanged 153-feature v13 main-model representation with eight
individual channel removals (150 features each) and removal of all eight
(129 features). Channels: 02, 04, 05, 09, 10, 11, 12, 24. Each removal deletes
the raw value, historical mean and historical sample standard deviation.
Telemetry 07 and every other channel retain their existing representations.

Use v13's XGBoost parameters verbatim: 3,000 maximum trees, depth 5, learning
rate 0.015, subsample/column subsample 0.8, lambda 1, seed 0, one CPU thread,
and 100-round early stopping. Use cap-125 targets, all training cycles,
v13 test-cutoff similarity sample weights and correlation-based feature weights.
No tuning, additional model family, specialist/gate, or feature redesign.

## Evaluation

- Five deterministic GroupKFold partitions, matching v13's grouping algorithm.
- Within each outer training partition, reserve 10% of UAVs with RandomState(0)
  for unweighted early stopping, following v13's final-fit holdout rule.
  Fit on the remaining UAVs without a subsequent refit. Scoring UAVs never
  supply early-stopping labels or fitted feature weights.
- Freeze the historical v13 tiers/order from Phase 3 Run 9, rather than
  recomputing tiers after each removal. This is a test of an existing,
  development-informed representation, not independent feature discovery.
- Recompute v13 correlation feature weights using actual fitting UAVs only,
  then subset those same weights for every ablation. Compute similarity sample
  weights on those fitting rows with v13's original formula.
- Reuse v13's eligible cutoff-assignment function for seeds 0 through 9.
  All 100 UAVs receive one cutoff per seed. These are 1,000 paired endpoints,
  not 1,000 independent UAVs or ten independent model-training seeds.
- Read only IDs/cycles from test.csv to reproduce v13's cutoff distribution.
  Score capped RUL primarily; also report uncapped metrics separately.
- Report pooled metrics, per-fold and per-cutoff-seed metrics, and short-history
  and late-life diagnostics. Bootstrap paired errors by UAV (10,000 draws).
  Include ordinary 95% and Bonferroni-adjusted intervals for nine comparisons.
  Intervals condition on these fits and do not cover training-split variability.

The 50 fits are a focused development screen. No production model, locked
evaluation or submission is changed. Promising removals require confirmation;
individual effects cannot establish that their combination is safe to remove.

## Run

From the repository root:

```powershell
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_35/run.py --check
.venv/Scripts/python.exe -u 2_architecture_experiments/1_pipeline_experiments/experiments/PE_35/run.py
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_35/plot_results.py
```

The runner validates exact feature order/counts, causal prefix summaries,
disjoint fitting/stopping/scoring UAVs, and matched endpoints. Each completed
fit saves held predictions and metadata atomically. Repeating the command
resumes verified checkpoints. Input/code/settings/package fingerprints are
registered before fitting; changes require a new `study.run` directory.

Results appear in `runs/run_1/reporting/report.md` and `paired_ablation.csv`.
`fold_membership.csv`, `evaluation_endpoints.csv`, `feature_catalog.json`,
the v13 source snapshot and `registration.json` make the run auditable.
The plot command also writes `reporting/fit_diagnostics.csv`, with selected
tree counts and elapsed fitting time per configuration/fold.

Run the channel-removal, UAV-separation and paired-reporting contract tests:

```powershell
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_35/test_ablation.py
```
