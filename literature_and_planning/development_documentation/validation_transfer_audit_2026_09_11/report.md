# Validation transfer audit — 11 September 2026

The historical endpoint suite already matches the empirical test history-length
distribution exactly. History length alone does not explain the leaderboard
ranking reversal. The completed comparison does reveal an adaptation in PE_31's
simpler baseline and substantial differences in calibration support, scoring,
and final fitting. These are testable explanations, not established causes.

This audit uses existing PE_31 predictions, raw training trajectories, observable
test features, source code and saved model contracts. No test RUL labels or
locked evaluation scenarios were loaded. No new predictor was trained.

## Matched comparison already completed

PE_31 evaluated Run 7 and its reference-protocol adapter on exactly the same
outer-held UAVs and endpoints: two five-fold partitions, with candidate seeds
averaged at prediction level. The endpoint identity check passed. The 100 UAVs
are reused across partitions; this is not 1,000 independent test subjects.

| Suite | Run 7 pooled R² / RMSE | PE_31 reference pooled R² / RMSE |
|---|---:|---:|
| Historical | 0.89304 / 10.8910 | 0.86941 / 12.0336 |
| Nominal | 0.90110 / 10.5793 | 0.88767 / 11.2743 |
| Unrestricted | 0.34786 / 65.7173 | 0.37736 / 64.2139 |

These use raw labels and equal total weight per UAV. Seed zero alone gives
reference historical R² 0.86975, so averaging the additional candidate seed
does not explain the reversal. Both raw and capped-label scoring are identical
on the historical suite because its true labels are already at most 125.

**Limitation found:** PE_31 `campaign_models.reference_protocol` selected its
blend using one uniform feasible cutoff per training UAV. The submitted script
uses the empirical test cutoff distribution with a greedy one-to-one feasible
assignment. PE_31 therefore tested an adapted reference, not the unchanged
submission recipe inside cross-validation. Its raw-score comparison is valid
for that adapter but does not settle the reproduced script's ranking.

See `matched_metrics.csv` for seed-zero and two-seed results, capped-label
diagnostics and age-band reweighting. None of the retrospective alternatives
are new promotion evidence.

## History and endpoint sampling

| Profile | Rows | Mean history | Median | History ≤100 | Distance to test in cycles |
|---|---:|---:|---:|---:|---:|
| Actual test | 100 | 165.96 | 148 | 21.0% | 0 |
| Historical | 500 | 165.96 | 148 | 21.0% | 0 |
| Nominal | 300 | 179.85 | 165 | 10.7% | 14.00 |
| Unrestricted | 300 | 129.92 | 122 | 30.3% | 36.04 |

Distance is one-dimensional Wasserstein distance. Historical repeats the full
empirical cutoff distribution five times. Reweighting its fixed age bands to
test frequencies changes neither model's score. The nominal and unrestricted
suites do not preserve that marginal distribution: per-UAV feasibility filters
and independent draws change it. Reweighting them changes scores but does not
reverse their within-suite model ranking in this audit.

The historical scenario configuration uses bipartite assignment with true RUL
between 1 and 125. Thus matching cutoff lengths does not establish matching
degradation states: historical endpoint assignment is conditional on training
lifetimes/labels. Test labels and the true test RUL distribution remain unknown.
The original script's seed-1000 empirical calibration assignment instead has
21% of training endpoints with raw RUL above 125, subsequently capped for
calibration. This difference remains even though cutoff marginals match.

## Sensor distributions and causality

Raw endpoint sensor distributions differ despite the historical age match.
Among sensors used by Run 7, the Wasserstein distance divided by historical
IQR is approximately 0.93 for telemetry_22, 0.69 for telemetry_18, 0.54 for
telemetry_15 and 0.50 for telemetry_01. Differences also occur within fixed
age bands; see `endpoint_sensor_drift.csv`. These are descriptive distances,
not significance tests or proof of harmful drift. Small IQRs and small
within-band test samples can inflate ratios. Repeated endpoints are dependent.

The test prediction disagreement is recorded for every UAV, without labelling
either model's prediction as correct. The simpler prediction minus Run 7
averages +4.13 cycles at histories 51–100, -3.79 at 101–150, and -4.76 at
151–200. The four test histories at most 50 cycles have a +10.71 average gap.
These differences identify observable disagreement, not test errors.

Perturbing future sensor rows and all RUL labels left the original script's
prefix features and Run 7's prefix features unchanged in an actual-UAV check.
The original script relies on positional label alignment after sorting features;
the current train CSV is already sorted, so that assumption holds for this
submission. The new baseline sorts explicitly. These checks do not claim that
all prior feature/model selection was independent of development outcomes.

## Preprocessing, calibration and final fitting

| Step | Run 7 | Submitted simpler script |
|---|---|---|
| Inputs | Frozen 298 selected engineered features | 266 features from 22 nonconstant sensors |
| Prefixes | 20 per UAV; each UAV total weight 1 | Every cycle; unit row weights |
| Feature processing | Existing prefix builder; winning XGB/ExtraTrees contracts use no signal compression or fault-mode partition | Causal expanding/rolling features, missing values filled with zero; no PCA or scaling |
| Fit targets | Capped at 125 | Capped at 125 |
| Internal calibration | Raw-label historical endpoints; four grouped OOF folds choose family weight and fit a residual model | Five grouped stopping fits per family; capped labels on a single empirical-cutoff draw choose XGB/CatBoost weight |
| Internal score caveat | Development data repeatedly used in research | Same held groups supply stopping labels and the script's printed stable score; capped `RUL_raw` is not actually raw |
| Final fitting | Six XGB/ExtraTrees members trained on all 100 UAVs, with residual correction | XGB and CatBoost fitted on 90 UAVs, with 10 stopping UAVs never refitted |
| Final postprocessing | Residual correction, nonnegative predictions, no extra safety offset | Weighted XGB/CatBoost predictions, no residual model |

The reproduction log records XGB share 0.70, XGB best iteration 1980 and
CatBoost best iteration 2998 (zero-based). Its internal stable scores, 0.9436
and 0.9452, cannot be compared directly with unbiased outer raw-label R² or
Kaggle R². The final-fit log's wording “all training data” is inaccurate; the
actual 90/10 grouped stopping code is preserved in the reproduced submission.

These findings motivate retaining each pipeline's original fitting procedure
in a controlled comparison rather than silently harmonizing away differences.

## Implemented next comparison and verification

PE_32 now defaults to a separate `run_2`, adding `simple_reproduction` beside
Run 7 and all four existing LightGBM recipes. All methods use identical outer
splits, endpoints, raw labels and equal-UAV scoring. The simpler baseline
preserves its original feature order, hyperparameters, seed zero, capped inner
objectives, blend grid and final stopping split. It uses training-side empirical
cutoffs rather than PE_31's uniform cutoff calibration.

The 100-cutoff assignment requires a declared adaptation for 80 training UAVs:
draw 80 empirical cutoffs without replacement, apply the original greedy
assignment, and reject infeasible complete draws with a fixed maximum of 1,000.
Every one of the 15 declared outer-training sets passed on the first draw.
Assignments and rejection counts are saved. No outer-held labels/lifetimes are
used. This is a closer nested protocol reproduction, not a claim of byte-for-byte
equivalence to a 100-UAV full-data execution.

The baseline is fitted once per outer fold and reused across LightGBM seeds.
It adds 60 base fits to screening and at most 120 to confirmation. A candidate
must satisfy the existing Run 7 gate and have strictly better historical
mean-fold RMSE than the simpler baseline. Paired bootstrap comparisons against
both baselines are reported. All original validation verdicts are preserved.

Verification: 14 bounded-comparison tests passed in total. Thirteen passed in
the sandbox; the Windows spawn checkpoint test passed when rerun outside the
sandbox after local process-pipe creation was denied. Checks cover held-label
and future-row isolation, baseline cache reuse, comparison reporting, decision
gates, LightGBM/TCN behavior and existing checkpoint integrity. PE_32 run_2 input
validation passed. **The new full training comparison has not been run**; the
table above contains completed PE_31 results only.

Run from the repository root:

```powershell
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_32\run.py
```

Do not redefine validation until it happens to agree with the leaderboard.
Retain the historical suite and explicit stress diagnostics. The next result
should tell us whether the more faithful simpler baseline changes the local
ranking; it cannot guarantee R² above 0.9 on Kaggle.

Reproduce the read-only-model audit with
`2_architecture_experiments/1_pipeline_experiments/audit_validation_transfer.py`.
Its input hashes and CSV diagnostics are saved in this directory.
