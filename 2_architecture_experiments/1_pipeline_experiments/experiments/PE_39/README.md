# PE_39: removing telemetry 05 and 24 together from v13

PE_35 screened eight unclear channels one at a time and closed with an explicit
warning: *do not combine individual removals without evaluating that
combination*. Telemetry 05 and 24 are the two that earned a second look. In
PE_35 they were among the cheapest to drop (05 at `+0.0755` cycles, 24 the
cheapest of all at `+0.0088`), and in PE_37 they were the only two channels
whose addition to the reduced baseline made things **worse** (`+0.0477` and
`+0.0588`). They are the one pair for which the evidence points the same way
from both directions. This study evaluates that pair as a pair.

| Arm | Removed channels | Features |
| --- | --- | ---: |
| baseline | None | 153 |
| drop_05 | telemetry 05 | 150 |
| drop_24 | telemetry 24 | 150 |
| drop_05_24 | telemetry 05 and 24 | 147 |

Each removal deletes the channel's raw value, historical mean and historical
sample standard deviation — both are weak-tier, so three features each. The
headline comparison is `drop_05_24` against the unchanged baseline; the two
single removals ride along as matched arms so the combination is interpretable.

## Fixed protocol

The same main-XGBoost protocol as PE_35 and PE_36: v13's 153-feature contract,
frozen historical tiers, cap-125 fitting targets, 3,000 maximum trees, depth 5,
learning rate 0.015, row/column subsampling 0.8, lambda 1, model seed 0, one CPU
thread, and 100-round early stopping. No hyperparameter search, specialist,
blend or additional model family.

All four arms share five deterministic UAV-grouped folds, the same inner 10% UAV
holdout for early stopping, and ten test-cutoff assignments per UAV (1,000
paired endpoints). Each fold has 72 fitting, eight stopping and 20 scoring UAVs.
Feature weights use fitting UAVs only; test-cutoff sample weighting uses v13's
formula. The test file supplies only UAV IDs and cycles. Models are not refit
after early stopping.

Primary scores use capped RUL; uncapped and error-region metrics are reported
separately. Paired bootstrap intervals resample UAVs with 10,000 draws and the
same bootstrap seed as PE_35 and PE_36, drawn once and shared across all arms so
every comparison rests on the same resamples. Multiplicity adjustment covers the
**three** comparisons against the baseline. Two secondary contrasts are declared
in `settings.toml` — the marginal cost of removing the second channel once the
first is gone — and adjusted over their own family.

## The interaction term

The report answers one question the single-channel screens cannot:

```
interaction = (pair - baseline) - (drop_05 - baseline) - (drop_24 - baseline)
```

A positive interaction means the pair costs more than its parts predict — the
two channels carry information the other does not replace. A negative one means
they are redundant with each other, so the second removal is nearly free once
the first has happened. The interval comes from the same shared UAV bootstrap
and is written to `reporting/additivity.json`. It is reported unadjusted and is
descriptive: only the baseline comparisons decide anything.

## Reproduction check

PE_35 already fitted `baseline`, `drop_05` and `drop_24` under this exact
protocol. PE_39 **refits** them rather than importing them, following PE_36's
precedent, and then compares its own predictions against PE_35's cell by cell.
The outcome is recorded in `reporting/reproduction_check.json` and summarised in
the report. If the shared arms differ while inputs, settings and library
versions are all identical, the runner raises: the protocol is meant to be
deterministic, and silent drift would invalidate every comparison in the
register. A difference under a changed library version is reported, not raised.
Point `reproduction_check` at another run, or set it to `""`, to skip this.

## Run manually

From the repository root, start the **20 model fits**:

```powershell
.venv/Scripts/python.exe -u 2_architecture_experiments/1_pipeline_experiments/experiments/PE_39/run.py
```

The command resumes completed, checksummed fits if interrupted. No production
model or submission is changed.

After training finishes, create the chart and fit-diagnostics table:

```powershell
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_39/plot_results.py
```

Validation only, with **no training**:

```powershell
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_39/run.py --check
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_39/test_paired_removal.py
```

The check validates feature counts and order, that the pair removes exactly the
union of the two single removals and nothing else, causal prefix summaries,
disjoint UAV partitions and matched endpoints. Source, data, settings and
runtime fingerprints guard resuming; change `study.run` before executing a
modified scientific configuration.

Results are written to `runs/run_1/reporting/report.md`, with per-arm effects in
`paired_ablation.csv`, the interaction term in `additivity.json`, the declared
contrasts in `secondary_contrasts.csv`, and all predictions and per-fold, seed
and region metrics alongside. The optional plot command writes
`paired_removal_effects.png`, its SVG version, and `fit_diagnostics.csv`.

## Reading the result

PE_35's single-channel intervals for these two channels were
`[-0.0165, +0.1683]` and `[-0.0585, +0.0764]` — both comfortably across zero. A
pair effect near the sum of two effects that small will also cross zero, and
that is a real outcome, not a failed experiment: it would say the two channels
can be dropped at no measurable cost, which is a simplification argument rather
than an accuracy one. A removal candidate still needs confirmation, and this
result covers exactly this pair. It does not license removing a third channel.
