# Step 7: Architecture comparison

## Purpose

This step converts the complete locked predictions from Step 6 into comparable
tables and figures. It reports predictive performance, uncertainty, stability,
reliability across predefined groups, and computational cost for every enabled
architecture.

Step 7 does not tune models, rank architectures, select a best seed, calculate
an overall score, or choose a winner. The researcher makes the architecture
decision from the saved evidence.

## Mandatory Step 6 gate

"comparison_gate.py" reads only the Step 1 experiment specification and the
Step 6 manifest before deciding whether locked result tables may be opened. It
requires:

- Step 6 status "complete";
- all expected family, outer-fold, and seed runs to be complete;
- the exact expected number of locked predictions;
- matching Step 1 and Step 6 contract versions;
- all enabled families in their predeclared contract order;
- fixed Step 5 retraining durations for applicable models;
- no use of locked results for tuning;
- no automatic architecture selection;
- no test-data access.

There is no command-line option that bypasses the gate. A partial Step 6
manifest is rejected before "locked_predictions.csv.gz" or "model_runs.csv" is
read.

## Input validation

After the gate passes, "ArchitectureComparisonAnalyzer" checks that:

- each expected family, fold, and seed run occurs exactly once;
- each run contains 20 held-out UAVs and 20 scenarios, giving 400 predictions;
- all architectures predict the same 2,000 UAV/scenario endpoints per seed;
- endpoint targets, cutoffs, folds, and metadata agree across architectures;
- configuration identifiers agree between predictions and model-run records;
- stored residuals equal prediction minus target;
- predictions are finite and obey the nonnegative RUL boundary;
- training, validation, timing, parameter, and model-size facts are valid.

These checks ensure that differences in the comparison come from model outputs
rather than mismatched validation samples.

## Metrics and seed handling

The four contract metrics are R2, RMSE, MAE, and signed bias. Metrics are saved
separately for every retained seed. The architecture-level value is the mean of
the individual-seed metric values, and the population standard deviation shows
seed sensitivity.

Predictions are not averaged across seeds. Such averaging would evaluate a new
ensemble architecture, which is outside the declared study. Deterministic
families have one seed and therefore a seed standard deviation of zero.

## Reliability groups

Each seed is evaluated:

- overall;
- by held-out outer fold;
- by locked validation scenario;
- by flight-cycle age band;
- by terminal-lifetime quantile.

The grouped architecture table reports the mean and standard deviation across
the retained seeds. RMSE and bias are plotted for every reliability group so a
strong overall result cannot hide a severe subgroup failure.

## Paired UAV bootstrap

The uncertainty procedure uses 1,000 repetitions and seed 20260814 from the
experiment contract. Each repetition samples the 100 complete UAV groups with
replacement. All 20 scenario rows belonging to a sampled UAV receive the same
resampling multiplicity.

The same sampled UAVs are used for every architecture and seed in a repetition.
This pairing removes artificial uncertainty caused by comparing models on
different validation samples. The output includes:

- a bootstrap metric value for every family and repetition;
- a 95% interval for each architecture metric;
- a 95% interval for every pairwise family A minus family B difference.

The pairwise table reports signed differences only. It is evidence for manual
comparison, not a ranking or significance-driven winner rule.

## Efficiency

Training time, inference time, trainable parameter count when the adapter can
provide it, and serialized model size are summarized separately. Peak memory is
not reported because Step 6 does not currently measure it reliably.

No weighted score combines performance and efficiency. The cost plots expose
the trade-off for manual interpretation.

## Files

- "comparison_gate.py" validates the complete Step 6 prerequisite without
  loading locked result tables.
- "architecture_comparison.py" validates inputs, calculates metrics and
  uncertainty, and writes traceable artifacts.
- "plot_architecture_comparison.py" creates figures from the calculated tables.
- "run_architecture_comparison.py" provides the command-line interface.

## Generated artifacts

- "architecture_comparison.csv" contains overall seed means, seed standard
  deviations, and paired-UAV bootstrap intervals.
- "seed_metrics.csv" retains overall results for every individual seed.
- "grouped_metrics.csv" retains every group result for every seed.
- "grouped_architecture_metrics.csv" summarizes grouped results across seeds.
- "bootstrap_architecture_metrics.csv.gz" retains every bootstrap replicate.
- "paired_metric_differences.csv" contains all pairwise metric differences and
  95% paired intervals.
- "efficiency_summary.csv" contains timing, parameter, and model-size facts.
- "figures/" contains overall metrics, uncertainty, fold, scenario, age-band,
  lifetime-group, seed-stability, paired-difference, and efficiency plots.
- "comparison_manifest.json" records the completed procedure and explicitly
  states that no rank or winner was written.

Generated artifacts remain visible locally and are ignored by Git.

## Running Step 7

After Step 6 is complete, run from the repository root:

    py 2_model_architecture_study\7_architecture_comparison\run_architecture_comparison.py

## Current state

Step 6 cannot yet be complete because Step 5 is still partial. The Step 7 gate
therefore remains closed, and no real locked comparison artifact has been
generated.
