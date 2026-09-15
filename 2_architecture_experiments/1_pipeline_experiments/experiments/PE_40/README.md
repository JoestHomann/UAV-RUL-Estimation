# PE_40: how training rows should be weighted in v13

v13 trains on **every** cycle of every UAV. Because all 153 features are
expanding or rolling, each row is already the feature vector of a prefix cut at
that cycle, so there is no resampling step — the complete set of prefixes is
used. The consequence is that a UAV contributes rows in proportion to how long
it lived, and lifetimes differ by a factor of **3.62** (145 to 525 cycles).

v13's only correction is `compute_test_similarity_weights`, a Gaussian kernel on
each row's `flight_cycle` against the 100 real test cut-off lengths. That is a
correction for *cycle position*, not for *UAV*. The Phase-2 pipeline takes the
other route: `current20` resamples roughly twenty prefixes per UAV and gives
every UAV equal total weight. Nothing in the register has ever compared them.

This study crosses the two factors.

| Arm | Similarity weights | Per-UAV normalisation | Features |
| --- | --- | --- | ---: |
| baseline | yes | no | 153 |
| similarity_uav_equal | yes | yes | 153 |
| uav_equal_only | no | yes | 153 |
| unweighted | no | no | 153 |

`baseline` is v13 exactly as shipped. Every arm uses the identical 153-feature
contract, the identical folds, endpoints, feature weights and tree settings.
**The per-row training weight is the only thing that moves.**

## Why a 2x2 rather than one comparison

PE_2 crossed scenario profile against fitting target and found that each change
*hurt* on its own while the pair was the largest gain in the project — capping
alone dropped R² from 0.7708 to 0.6690, bounded scenarios alone to 0.4867, and
together they reached 0.8661. The register's own conclusion was that what gets
frozen is the pair, not either factor. Two weighting factors that plausibly
interact deserve the same treatment, so the report carries an explicit
interaction term: the effect of per-UAV normalisation with similarity weights
present, minus its effect without them.

## What the normalisation does

`equalise_by_uav` divides each row's weight by its own UAV's total, so every UAV
sums to the same amount whatever its lifetime, and then renormalises the whole
vector to mean one. Dividing by the UAV's *total weight* rather than its row
count keeps each UAV's internal cycle profile intact — a long UAV's late rows
stay relatively down-weighted, the UAV simply stops out-voting short ones.
Renormalising to mean one matters because XGBoost's `reg_lambda` and
`min_child_weight` are defined against the weight scale; without it the arms
would differ in effective regularisation as well as in weighting.

Measured on fold 0, the ratio of the heaviest to the lightest UAV's total weight:

| Arm | UAV total ratio |
| --- | ---: |
| unweighted | 3.41 |
| baseline (v13) | 2.09 |
| uav_equal_only | 1.00 |
| similarity_uav_equal | 1.00 |

Worth noting before reading any result: v13's similarity weights already halve
the imbalance as a side effect, from 3.41 to 2.09, because long-lived UAVs carry
many high-cycle rows that the kernel down-weights. They are a partial per-UAV
correction that nobody designed as one. The runner asserts that the two
normalised arms reach exactly 1.00.

## Fixed protocol

The same main-XGBoost protocol as PE_35, PE_36 and PE_39: v13's 153-feature
contract, frozen historical tiers, cap-125 fitting targets, 3,000 maximum trees,
depth 5, learning rate 0.015, row/column subsampling 0.8, lambda 1, model seed 0,
one CPU thread, and 100-round early stopping.

All four arms share five deterministic UAV-grouped folds, the same inner 10% UAV
holdout, and ten test-cutoff assignments per UAV (1,000 paired endpoints). Each
fold has 72 fitting, eight stopping and 20 scoring UAVs. Weights are computed
from the fitting UAVs of each fold only, exactly as v13 computes them.
**Early stopping stays unweighted in every arm**, matching v13, so the arms
differ in fitting pressure and not in when fitting halts.

Primary scores use capped RUL. Paired bootstrap intervals resample UAVs with
10,000 draws and the same bootstrap seed as PE_35, PE_36 and PE_39, drawn once
and shared across all arms so every comparison rests on the same resamples.
Multiplicity adjustment covers the **three** comparisons against v13. Two
secondary contrasts are declared in `settings.toml` and adjusted over their own
family; they isolate what each factor buys against no weighting at all.

## Run manually

From the repository root, start the **20 model fits**:

```powershell
.venv/Scripts/python.exe -u 2_architecture_experiments/1_pipeline_experiments/experiments/PE_40/run.py
```

The command resumes completed, checksummed fits if interrupted. No production
model or submission is changed.

After training finishes, create the chart and fit-diagnostics table:

```powershell
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_40/plot_results.py
```

Validation only, with **no training**:

```powershell
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_40/run.py --check
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_40/test_weighting.py
```

`--check` validates the feature contract, causal prefix summaries, disjoint UAV
partitions, matched endpoints, and that the declared arms actually equalise UAV
totals; it writes `weight_diagnostics.csv` with the per-fold weight ranges and
UAV totals for every arm, so the arms can be inspected before any model is fit.

Results are written to `runs/run_1/reporting/report.md`, with per-arm effects in
`paired_weighting.csv`, the interaction in `interaction.json`, the declared
contrasts in `secondary_contrasts.csv`, and all predictions and per-fold, seed
and region metrics alongside.

## Reading the result

This changes the weight on training rows and nothing else. It is **not** a test
of `current20`, which also changes how many rows exist — a separate arm would be
needed for that, and it would move two things at once.

PE_35 through PE_39 all returned inconclusive at effect sizes near ±0.1 cycles
on a ~10.1-cycle baseline. Treat any interval here that crosses zero the same
way. An inconclusive result is still useful: it would say the two pipelines'
different choices do not matter, which is the answer to the Q&A question that
prompted this study.
