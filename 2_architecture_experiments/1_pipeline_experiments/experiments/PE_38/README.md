# PE_38: telemetry 07 representation study with v13 XGBoost

Telemetry 07 is the one channel that is both degradation-associated and
state-like: Phase 0 found it increasing with age in every UAV while flatlining
across about 97% of rows. In the data it takes exactly four values on a uniform
`0.0859487` grid, so it is a genuinely discrete operating state, not a
continuous sensor that happens to be noisy. v13 currently treats it as an
ordinary medium-tier sensor with seven numeric features. This study asks which
representation it should actually receive.

| Arm | Telemetry 07 representation | Features |
| --- | --- | ---: |
| baseline | v13 as shipped: medium-tier numeric (7) | 153 |
| numeric_strong | strong-tier numeric (12) | 158 |
| state_only | discrete-state block (11) | 157 |
| medium_plus_state | medium numeric + state block | 164 |
| strong_plus_state | strong numeric + state block | 169 |
| drop_07 | channel removed entirely | 146 |

The six arms cross numeric depth against the state block and bracket both with
controls. Every other channel, including Telemetry 16, is untouched in all six.

## The two representations

**Numeric.** The deeper numeric arm is v13's own strong-tier recipe, obtained by
moving the channel between tiers rather than by writing new feature code, so it
adds exactly `last_minus_hist_mean`, `roll5_mean`, `roll5_std`, `roll20_mean`
and `roll20_std`. The runner asserts that the seven shared columns are
numerically identical to the baseline's.

**State.** Eleven causal columns, each reading only cycles 1..n of that UAV:

- `state` — current state index
- `state_n_unique` — distinct states seen so far
- `state_n_transitions`, `state_transition_rate` — changes so far, and per cycle
- `state_run_length`, `state_run_fraction` — length of the current run
- `state_max_so_far` — highest state reached
- `state_frac_l0` … `state_frac_l3` — share of observed cycles spent in each state

Rates divide by the within-UAV observation count, which `prepare` verifies is
equal to `flight_cycle` in this dataset.

Note what `state_only` does and does not remove. For a tree the ordinal state
code is an order-preserving relabel of the raw value, so that arm keeps the
channel's current level and drops only its numeric history; `drop_07` is the
arm that removes the channel outright.

## Fixed protocol

The same main-XGBoost protocol as PE_35 and PE_36: v13's 153-feature contract,
frozen historical tiers, cap-125 fitting targets, 3,000 maximum trees, depth 5,
learning rate 0.015, row/column subsampling 0.8, lambda 1, model seed 0, one CPU
thread, and 100-round early stopping. No hyperparameter search, specialist,
blend or additional model family.

All six arms share five deterministic UAV-grouped folds, the same inner 10% UAV
holdout for early stopping, and ten test-cutoff assignments per UAV (1,000
paired endpoints). Each fold has 72 fitting, eight stopping and 20 scoring UAVs.
The test file supplies only UAV IDs and cycles. Models are not refit after early
stopping, as in PE_35 and PE_36.

**State levels are derived per fold from the fitting UAVs only**, and any value
outside that fold's grid is assigned to the nearest known state. Scoring UAVs
therefore contribute to neither the encoding, the feature weights, the fit, nor
early stopping. Each fold's grid is recorded in `registration.json` and in every
cell's metadata; `--check` fails loudly if a fold's fitting UAVs do not recover
the declared four states. Every state column is named `telemetry_07__…`, so it
inherits the channel's own correlation-based feature weight with no change to
v13's weighting code.

Primary scores use capped RUL; uncapped and error-region metrics are reported
separately. Paired bootstrap intervals resample UAVs with 10,000 draws and the
same bootstrap seed as PE_35 and PE_36. Multiplicity adjustment covers the
**five** comparisons against the baseline. Four secondary contrasts are declared
in `settings.toml` and adjusted over their own family; they locate any effect
but do not decide anything. This is a development screen with fixed historical
tiers, not independent confirmation.

## Run manually

From the repository root, start the **30 model fits**:

```powershell
.venv/Scripts/python.exe -u 2_architecture_experiments/1_pipeline_experiments/experiments/PE_38/run.py
```

The command resumes completed, checksummed fits if interrupted. No production
model or submission is changed.

After training finishes, create the chart and fit-diagnostics table:

```powershell
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_38/plot_results.py
```

Validation only, with **no training**:

```powershell
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_38/run.py --check
.venv/Scripts/python.exe 2_architecture_experiments/1_pipeline_experiments/experiments/PE_38/test_representations.py
```

`--check` validates feature counts and order, the identity of the shared numeric
columns across tiers, per-fold state grids, disjoint UAV partitions, matched
endpoints, and that a truncated trajectory reproduces every numeric **and**
state feature exactly. It writes setup artifacts under `runs/run_1/`; it does
not create fitted cells or claim study completion. Source, data, settings and
runtime fingerprints guard resuming; change `study.run` before executing a
modified scientific configuration.

Results are written to `runs/run_1/reporting/report.md`, with per-arm effects in
`paired_representations.csv`, the declared contrasts in
`secondary_contrasts.csv`, and all predictions and per-fold/seed/region metrics
alongside. The optional plot command writes `representation_effects.png`, its
SVG version, and `fit_diagnostics.csv`.

## Reading the result

A winning arm is a candidate, not a decision. PE_35 through PE_37 all returned
inconclusive at effect sizes near ±0.1 cycles on a ~10.1-cycle baseline, so
treat any interval that crosses zero here the same way. The arms change one
channel only: a result for Telemetry 07 does not transfer to Telemetry 16 or
license combining this change with other pending ones.
