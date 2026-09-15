# PE_41: which difference reference should v13 keep

v13 measures how far a channel has moved in two ways at once, and always both:

- `baseline_delta` — the current value minus that UAV's **first observed value**
- `last_minus_hist_mean` — the current value minus that UAV's **expanding mean**

Both answer the same question, "how far has this channel drifted from where this
UAV normally sits", and they differ only in what counts as normal: the very first
cycle, or everything seen so far. The EDA that motivated them could not say which
reference is the better one, so both were kept and nothing since has asked whether
that was necessary. This study asks.

| Arm | Removed | Features |
| --- | --- | ---: |
| baseline | nothing | 153 |
| drop_history_difference | `last_minus_hist_mean` | 144 |
| drop_baseline_difference | `baseline_delta` | 143 |
| drop_history_difference_matched | `last_minus_hist_mean` **and** `telemetry_07__baseline_delta` | 143 |
| drop_both | both families | 134 |

## Why there is a fifth arm

v13 does not give the two families the same reach. The strong tier receives all
twelve feature families; the medium tier receives seven, and those seven include
the baseline difference but **not** the history-mean difference. Telemetry 07 is
the only medium-tier channel, so it carries a baseline difference with no
history-mean counterpart.

The consequence is that `drop_baseline_difference` deletes **10** columns while
`drop_history_difference` deletes **9**. Comparing those two directly would
confound *which reference is better* with *which arm lost one more feature* — a
small confound on a study whose whole expected effect is small.

`drop_history_difference_matched` deletes the history-mean difference **and** that
one unpaired baseline column. It therefore holds 143 features against
`drop_baseline_difference`'s 143, and on the nine fully paired strong channels each
arm keeps exactly one reference and neither keeps anything on telemetry 07. That
contrast, declared as the first secondary contrast, is the one that answers
"which reference is better". The runner asserts the match rather than trusting it,
and the assertion survives a re-tiering: if a future v13 gave both families the
same reach, the matched arm would collapse onto the plain history removal and the
head-to-head would still be balanced.

## Fixed protocol

The same main-XGBoost protocol as PE_35, PE_36, PE_39 and PE_40: v13's 153-feature
contract, frozen historical tiers, cap-125 fitting targets, 3,000 maximum trees,
depth 5, learning rate 0.015, row/column subsampling 0.8, lambda 1, model seed 0,
one CPU thread, and 100-round early stopping. No hyperparameter search, specialist,
blend or additional model family.

All five arms share five deterministic UAV-grouped folds, the same inner 10% UAV
holdout for early stopping, and ten test-cutoff assignments per UAV (1,000 paired
endpoints). Each fold has 72 fitting, eight stopping and 20 scoring UAVs. Feature
weights use fitting UAVs only; test-cutoff sample weighting uses v13's formula. The
test file supplies only UAV IDs and cycles. Models are not refit after early
stopping. **The only thing that moves between arms is which difference columns are
present.**

Primary scores use capped RUL; uncapped and error-region metrics are reported
separately. Paired bootstrap intervals resample UAVs with 10,000 draws and the same
bootstrap seed as PE_35, PE_36, PE_39 and PE_40, drawn once and shared across all
arms so every comparison rests on the same resamples. Multiplicity adjustment covers
the **three** comparisons against the baseline that v13 could actually adopt. The
matched arm is not one of them; it exists to make the head-to-head fair, and it is
reported with the three declared secondary contrasts over their own family.

## The interaction term

```
interaction = (drop_both - baseline) - (drop_history - baseline) - (drop_baseline - baseline)
```

A **positive** interaction means the two references carry complementary information:
losing the second costs more than losing the first predicted, so keeping both is
doing real work. A **negative** one means they are substitutes — once one reference
is gone the other is nearly free to remove, which is the signature of two features
encoding the same thing. The interval comes from the same shared UAV bootstrap and
is written to `reporting/additivity.json`. It is reported unadjusted and is
descriptive: only the baseline comparisons decide anything.

## Reproduction check

PE_39 already fitted this exact `baseline` under this exact protocol. PE_41 **refits**
it rather than importing it and then compares its own predictions cell by cell. The
outcome is recorded in `reporting/reproduction_check.json` and summarised in the
report. If the shared arm differs while inputs, settings and library versions are all
identical, the runner raises: the protocol is meant to be deterministic, and silent
drift would invalidate every comparison in the register. A difference under a changed
library version is reported, not raised. Point `reproduction_check` at another run, or
set it to `""`, to skip this.

## Run manually

From the repository root, start the **25 model fits**:

```powershell
.venv/Scripts/python.exe -u 2_architecture_experiments/1_pipeline_experiments/experiments/PE_41/run.py
```

The command resumes completed, checksummed fits if interrupted. No production model
or submission is changed.

After training finishes, create the chart and fit-diagnostics table:

```powershell
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_41/plot_results.py
```

Validation only, with **no training**:

```powershell
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_41/run.py --check
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_41/test_difference_reference.py
```

`--check` validates the feature contract, that each arm removes exactly its declared
family and touches no other feature family, that dropping both is exactly the union
of the singles, that the head-to-head arms end at the same feature count, causal
prefix summaries, disjoint UAV partitions and matched endpoints. Source, data,
settings and runtime fingerprints guard resuming; change `study.run` before executing
a modified scientific configuration.

Results are written to `runs/run_1/reporting/report.md`, with per-arm effects in
`paired_ablation.csv`, the interaction in `additivity.json`, the declared contrasts
in `secondary_contrasts.csv`, and all predictions and per-fold, seed and region
metrics alongside. The optional plot command writes `difference_reference_effects.png`,
its SVG version, and `fit_diagnostics.csv`.

## Reading the result

PE_35 through PE_40 all returned inconclusive at effect sizes near ±0.1 cycles on a
~10.1-cycle baseline. Treat any interval here that crosses zero the same way, and
expect the head-to-head in particular to be small: these two columns are strongly
correlated by construction, since for a monotone channel the expanding mean sits
between the first value and the current one.

An inconclusive head-to-head is a real outcome, not a failed experiment. It would say
the two references are interchangeable, which is a **simplification** argument rather
than an accuracy one: v13 could keep whichever is cheaper to explain and drop the
other. The arm that decides whether anything can be dropped at all is `drop_both` —
if that one is clearly harmful while both singles are inconclusive, the honest reading
is that v13 needs *a* difference reference but not *two*, and the choice between them
is free.

This result covers exactly these two families under v13's frozen tiers. It does not
say which reference a differently tiered feature set would prefer, and it does not
license removing a third feature family.
