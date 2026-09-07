# Experiments toward Kaggle R² > 0.9

Research date: 7 September 2026. Scope confirmed by the user: unseen-UAV Kaggle performance, building on PE_1–PE_13, with literature from any year.

**Recommendation.** Carry the promoted Run 7 plus full-feature TabPFN blend into multi-seed confirmation, and give highest priority to a complete outer refit of PE_15's forecast-history candidate. The TabPFN gain is too small to close the Kaggle gap by itself; forecast history is the only completed screen with a large enough effect to do so if it survives nested evaluation. Do not expand the failed population, residual-coverage, compact-feature, or broad neural searches.

Implementation update, 7 September 2026: PE_14–PE_19 are complete as experiment-owned workflows. PE_18 promoted a small full-feature TabPFN blend; PE_15 produced a larger forecast-history screening candidate that still needs complete nested outer refitting. No new locked evaluation was opened and no Kaggle submission was made. The earlier September 1 strategy and implementation plan are historical: several of their proposed experiments have now failed, and Phase 3 Run 7 is already complete.

**Where the project actually stands**

| Evidence | R² | RMSE | Interpretation |
| --- | ---: | ---: | --- |
| Phase 3 Run 6, mean development fold | 0.89274 | 10.6332 | Previous production configuration |
| Phase 3 Run 7, mean development fold | **0.90041** | **10.2361** | Current residual-corrected ensemble |
| Run 7, pooled 500 development predictions | **0.90446** | **10.2929** | Recomputed from saved predictions |
| PE_18 Run 7 + full TabPFN blend, mean development fold | **0.90258** | **10.1035** | Promoted by the declared 1%/4-of-5 practical gate |
| PE_18 Run 7 + full TabPFN blend, pooled development predictions | **0.90668** | **10.1725** | Small complementary gain; bootstrap interval crosses zero |
| Run 7, conditional UAV-bootstrap 95% interval | **0.87779–0.92772** | 9.0679–11.4521 | Uncertainty in fixed predictions, excluding training/selection uncertainty |
| Run 6 Kaggle public score | **0.86741** | Unknown | Confirmed by user-supplied screenshot |
| Run 7 Kaggle public score | **0.87652** | Unknown | Best score in the supplied screenshots; +0.00911 versus Run 6 |

Sources: [Run 6 report](C:/Users/joest/UAV-RUL-Estimation/3_final_model_training_and_inference/runs/run_6/7_post_run_reporting/report_summary.json), [Run 7 report](C:/Users/joest/UAV-RUL-Estimation/3_final_model_training_and_inference/runs/run_7/7_post_run_reporting/report_summary.json), [historical public result](C:/Users/joest/UAV-RUL-Estimation/literature_and_planning/development_documentation/pipeline_experiments.md:388), and [new diagnostics](C:/Users/joest/UAV-RUL-Estimation/literature_and_planning/development_documentation/r2_research_2026_09_07/diagnostic_summary.json).

The user supplied Kaggle submission screenshots after the initial research. All 14 distinct displayed submissions are transcribed in [the score record](C:/Users/joest/UAV-RUL-Estimation/literature_and_planning/development_documentation/r2_research_2026_09_07/kaggle_scores.csv); repeated entries visible in both screenshots are recorded once. The recording date is not an inferred submission date. These are user-supplied public scores, not a live lookup or private-test result.

On the same public target set, Run 7's improvement from **0.86741 to 0.87652** corresponds to **6.87% lower MSE**, or **3.50% lower RMSE**. That closely matches its **3.52% pooled development RMSE improvement**. This supports the residual-corrected ensemble as the new baseline, while it does not isolate the effect of each configuration change or establish private-set performance.

From Run 7's public score, reaching R² 0.9 now requires approximately **19.02% lower MSE**, equivalent to **10.01% lower RMSE** on that same target set. This follows from `R² = 1 − MSE/Var(y)`: the required RMSE ratio is `sqrt(0.1 / (1 − 0.87652))`. Absolute public RMSE remains unknown. The remaining mean-development/public R² gap is **0.02389**. The PE_14–PE_19 priorities remain appropriate; Run 7's submission/score confirmation is now complete.

The conditional paired bootstrap interval for Run 7 minus Run 6 RMSE is **−1.252 to +0.446 cycles**. The point estimate favors Run 7, but this analysis does not establish a reliably positive improvement across new UAV samples. Moreover, all 100 training UAVs have influenced previous development choices; resplitting them produces a robustness check, not a genuinely untouched test set.

**What the errors suggest**

| Run 7 region | Rows | RMSE | Bias: prediction minus truth | Share of total squared error |
| --- | ---: | ---: | ---: | ---: |
| True RUL 1–25 | 44 | 4.85 | +1.87 | 2.0% |
| True RUL 26–50 | 58 | 8.28 | +2.05 | 7.5% |
| True RUL 51–75 | 87 | 11.33 | +2.64 | 21.1% |
| True RUL 76–100 | 118 | 11.63 | +4.92 | 30.1% |
| True RUL 101–125 | 193 | 10.39 | −4.54 | 39.3% |
| Observed history ≤100 cycles | 105 | **12.80** | −0.67 | **32.5%** |

RUL 51–125 contributes 90.5% of error and 79.6% of observations. Short histories contribute disproportionately: 21% of observations produce 32.5% of error. The ten highest-error UAVs contribute 38.7% of total squared error. Inspect their trajectories to form hypotheses, but do not remove them or create ID-specific corrections.

There are **444 distinct `(UAV, cutoff)` pairs among 500 scenario rows**, across 100 independent UAV identities. Removing repeated endpoints descriptively changes pooled R² only to 0.90494, so duplication does not explain the headline score. It does affect observation weighting and precision. The same UAV must remain the unit of bootstrap resampling. Narrow-band R² is unstable because the target variance is small; use band RMSE, bias and error contribution when deciding what to fix. [Error-band evidence](C:/Users/joest/UAV-RUL-Estimation/literature_and_planning/development_documentation/r2_research_2026_09_07/error_bands.csv).

**Do not repeat these as if they were new hypotheses**

The experiment register records failed promotion gates for dense neural sequences (Run 7), hybrid sequence/tabular networks (PE_10 and Run 8), personalized onset targets (PE_8), drift feature pruning (PE_9), raw/soft-tail targets (PE_5), and censored/horizon models (Run 9). Earlier experiments also tested dense tree prefixes, baseline normalization, PCA/median compression and unsupervised fault-mode experts. PE_13's selected safety subtraction reduced near-failure overprediction but worsened overall RMSE. PE_11 residual correction was the successful accuracy treatment; PE_12 retained it under test-like weighting. [Complete local experiment register](C:/Users/joest/UAV-RUL-Estimation/literature_and_planning/development_documentation/pipeline_experiments.md).

CatBoost and DTW retrieval adapters already exist. A CatBoost rerun is a matched-control refresh, not a new architecture. Raw-target DTW performed poorly under an older, broader target distribution; a new retrieval experiment would need a different health representation and the current evaluation contract.

**Proposed additions, in execution order**

IDs PE_14–PE_19 below are implemented. Their numerical gates remain prospective rules rather than guarantees of leaderboard improvement.

| Experiment and status | Hypothesis and comparison | Priority / first budget | Evidence needed to advance |
| --- | --- | --- | --- |
| **PE_14: complete, no promotion** | Run 7 reached pooled nominal R² 0.9011, but won 11/15 folds and its bootstrap interval crossed zero; unrestricted pooled R² was 0.6188 | Completed 60 outer cells | Nominal gain was encouraging but did not pass the stability gate |
| **PE_15: screening complete** | Existing features versus filtered sensor level/rate features versus previous-prediction features; combine only after independent tests | Five recipes completed | Forecast history improved mean RMSE 13.0% with 5/5 wins; requires complete outer confirmation |
| **PE_16: complete, no promotion** | Regularized HGB improved mean RMSE 0.63% with 4/5 wins; all other residual/coverage variants were weaker | Completed 8 recipes | Missed the 1% gate and bootstrap interval crossed zero |
| **PE_17: screening complete, rejected** | Current ensemble versus added per-sensor terminal-distance/rate and partially pooled health-trajectory features | Two recipes completed | Both regressed; control retained |
| **PE_18: complete, promoted for confirmation** | Full-feature TabPFN blend reached pooled R² 0.9067 and improved mean RMSE 1.30% with 4/5 wins; standalone and compact models were weaker | Completed 4 direct cells plus nested small blends | Passed the practical gate; bootstrap interval crossed zero, so confirm on new grouped splits |
| **PE_19: complete, no promotion** | TCN blend reached pooled R² 0.9062 but improved mean RMSE only 0.76% with 3/5 wins; LSTM regressed | Completed 2 temporal challengers | Tree control retained |

The numerical gates are proposed practical thresholds, not literature guarantees. Use the same five held-out UAV groups and the same endpoints within every paired comparison. Confirm selected recipes over two additional grouped split seeds. Four wins out of five is a screening heuristic, not a significance test.

**PE_14 — verify the full pipeline before interpreting another 0.9 score**

Two project-specific issues make this necessary:

1. **Scenario support is selected using true RUL.** The active `early_and_middle` profile assigns test-like cutoffs only where `1 ≤ RUL ≤ 125`. Matching the cutoff marginal therefore does not establish that the validation joint distribution matches hidden test endpoints. Evaluation uses raw labels, but only inside this selected support. Keep this historically useful profile as the nominal comparison. Also evaluate frozen candidates on new assignments under the same profile and on a stress profile with feasible cutoff assignment but no upper-RUL restriction. Report these separately; changing the target support changes the R² denominator. [Actual profile](C:/Users/joest/UAV-RUL-Estimation/2_architecture_experiments/1_pipeline_experiments/_internal/shared_settings.toml:84).

2. **The PE_11 development report is not a fully nested estimate of its complete stack.** The residual stage holds out inner fold A but fits on OOF predictions from folds B/C/D whose base models may have trained on A. Additionally, the blend value for a residual-training row in B was selected using labels in A/C/D. Disjoint UAV IDs at the final residual fit do not remove these indirect dependencies. The source hyperparameters were also selected using inner validation outcomes. [Blend/residual implementation](C:/Users/joest/UAV-RUL-Estimation/2_architecture_experiments/1_pipeline_experiments/evaluate_bagging_residual.py:47), [base member construction](C:/Users/joest/UAV-RUL-Estimation/2_architecture_experiments/1_pipeline_experiments/run_bagged_tree_members.py:112).

The Phase 3 Run 7 adapter is stronger: it filters calibration endpoints to training UAVs, makes internal grouped base predictions, fits the correction there, and predicts external held-out UAVs. Do not infer that Run 7 has the same direct evaluation dependency as the PE_11 report. Its remaining limitation is that configurations and the overall workflow were chosen using reused development data. [Phase 3 ensemble fitting](C:/Users/joest/UAV-RUL-Estimation/2_architecture_experiments/2_model_architecture_study/4_model_adapters/models/tabular/residual_corrected_tree_ensemble.py:233).

For an evaluation fold H, remove H before choosing any hyperparameters, fitting preprocessing, generating residual-training predictions, fitting blend weights, or fitting corrections. Generate inner OOF predictions entirely inside the remaining UAVs, fit the full stack, then predict H once. Reusing a precomputed global OOF table for both stack construction and external evaluation is insufficient. This design follows the distinction between stacking's internal prediction generation and external model evaluation in the [scikit-learn stacking documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html) and the [nested-CV guide](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

Add a dependency test: changing labels or future observations of H must leave every training-side artifact unchanged. Also report unique endpoint counts and use explicit equal-UAV weights when calibration scenarios have different numbers of distinct endpoints.

**PE_15 — filter measurements and reuse previous forecasts**

The feature builder already contains trailing means/slopes at windows 5/20/50 and full-history summaries. Therefore simply adding another rolling mean is not the proposed novelty. [Current feature definitions](C:/Users/joest/UAV-RUL-Estimation/1_dataset_construction/5_prefix_feature_engineering/build_prefix_features.py:32).

Test two separate mechanisms against the unchanged Run 7 recipe:

- **Sensor-state features:** append one-sided exponentially filtered level/rate features with half-lives 3 or 10, or a local-linear-trend Kalman filter with noise settings fitted only on training UAVs. Include slope uncertainty, current innovation, and disagreement between filtered and raw levels. Begin with continuous degradation channels 13/19/21/22/25/28; retain originals and the discrete-state features for 07/16. Use prefix replay: no full-trajectory smoothing before truncation and no validation-UAV parameter fitting.
- **Prediction-history features:** for endpoint `t`, obtain predictions from the same held-out-UAV base model at `t−k`, for `k ∈ {0,2,5,10,20}` where available. For uncapped-compatible earlier estimates, form `z_k = predicted_RUL(t−k)−k`. Test a small causal pooling filter separately from adding current-versus-past forecast differences, dispersion and trend to the residual head.

Past forecasts near the cap are not valid translated raw-RUL estimates: an earlier value of 125 may represent much more than 125 remaining cycles. Flag or exclude those translated values, keep a fallback to the current forecast, and predefine a near-cap sensitivity check. Do not hard-enforce monotone predicted RUL: new telemetry can legitimately revise expected remaining life upward. Our saved endpoints show increasing predictions in 11.0% of 344 successive observed endpoint pairs; this is a diagnostic, not proof of an error.

Kalman filtering of RUL predictions has direct precedent in an aircraft bleed-valve study, with benefits dependent on model and life stage, especially near end of life. It does not establish benefits for this UAV dataset or its short histories. [Baptista et al., RESS, 2019](https://www.sciencedirect.com/science/article/pii/S095183201731075X). A recent C-MAPSS study also compares signal smoothing methods; its architecture-level results do not isolate the gain expected here. [Han et al., PLOS One, 2026](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0340645).

Implementation: extend the prefix feature builder for sensor filtering; add training-fold forecast-history generation around the residual ensemble adapter. Verify identical endpoint features after modifying all later rows. Select a combination only if the two individual branches add value.

**PE_16 — estimate residuals more reliably with only 100 UAVs**

PE_11 already uses uncertainty, cutoff, base prediction, and eight telemetry baseline-delta/slope features. Run 7 already fits its residual model on development-like endpoints belonging to training UAVs. Neither should be presented as a new idea.

The new tests are:

- Compare the current seven-leaf, 100-iteration histogram-gradient residual model with a ridge/small spline mean correction on the same inputs, and with a more strongly regularized tree head.
- Compare correction strengths `λ ∈ {0, 0.5, 1}` in `prediction = base − λ × estimated_residual`, selecting λ inside training UAVs. Zero is the indispensable no-correction control.
- Hold the base learner's `current20` training prefixes fixed. Increase only residual-calibration coverage from five scenario endpoints to twenty distinct endpoints per training UAV, respecting the nominal support and balancing total UAV weight. Include inverse-duplicate weighting for the existing calibration table as a separate small ablation. More endpoints improve coverage, not the number of independent UAVs.

Use staged comparisons rather than the full Cartesian product. First screen head/strength with current coverage, then test coverage on the best two heads. Generate calibration endpoints within each training partition; never fit on the evaluated UAV's earlier labeled prefixes. Continue to measure errors against raw RUL.

For R², the desired point prediction minimizes squared error. A median or conservative quantile correction does not generally do so. Run 7 already uses no additional quantile calibration; preserve that accuracy control. [Mean versus quantile regression](https://scikit-learn.org/stable/modules/linear_model.html#quantile-regression).

**PE_17 — add a model of degradation rate without changing labels**

For selected continuous channels, estimate robust healthy and terminal reference distributions from training UAVs. Update a low-dimensional population trajectory model using each query prefix's observed level and rate. Start with partially pooled intercepts/slopes; consider mild curvature only if needed. Feed terminal-distance, rate, forecast time-to-threshold, and uncertainty into the existing tree/residual estimator.

A simple diagnostic feature is `remaining_sensor_distance / degradation_rate`, with explicit flags for weak slopes, wrong directions, and extrapolation outside training support. Use a population fallback where the prefix cannot identify a rate. The terminal reference must be learned exclusively from training UAVs; fitting it from the query UAV's eventual terminal readings would leak its future.

This differs from PE_8: it retains the cap-125 fitting control and raw evaluation labels instead of replacing labels with an estimated onset target. It also retains individual sensor features, unlike the failed PCA-only representations. Stop before full tuning if training-only leave-one-UAV-out checks show that terminal levels vary too much for reliable thresholds.

Empirical-Bayes updating of population degradation models provides a methodological basis, but the cited work concerns crack-growth/simulated degradation, not these anonymous channels. The proposed small parametric version is our adaptation. [Zhou, Serban and Gebraeel, 2011](https://arxiv.org/abs/1107.5712).

If this produces a useful health representation, an optional follow-up can revisit similarity retrieval with that representation and cap-125 evaluation. Compare to raw-representation retrieval on the same endpoints. Exclude the entire query UAV from the library. The classical method uses training run-to-failure patterns; it does not justify retrieving external hidden benchmark labels. [Wang, Yu, Siegel and Lee, 2008](https://doi.org/10.1109/PHM.2008.4711421).

**PE_18 — a pretrained tabular prior, with a bounded budget**

Test a pinned local TabPFN-3 regression checkpoint on the existing 298 features and on a compact 32–64-feature view selected inside training folds. Use the same capped fitting target, raw evaluation labels, and fixed prefix counts per UAV as controls. Compare direct predictions and their contribution to the Run 7 ensemble. Include the existing CatBoost adapter as a small matched-control refresh; do not start a broad search across tree families.

TabPFN-3's May 2026 official report describes synthetic-only pretraining and released weights. This is a different source of inductive assumptions from training a neural encoder on 100 UAVs. The reported general tabular benchmarks do not guarantee RUL performance. Pin package/checkpoint versions and resolve local weight access before execution; competition telemetry remains local. Use regression means for the R² comparison. [Official TabPFN-3 report](https://priorlabs.ai/technical-reports/tabpfn-3), [model card](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/README.md).

Feature selection and all model context rows must exclude validation UAVs. Thousands of correlated prefixes are still only 100 systems. A compact kernel-ridge/Gaussian-process residual model is a reasonable later alternative, but should not expand the initial four-cell screen.

**PE_19 — let a weak model earn a small ensemble role**

PE_7's standalone accuracy gate prevented testing whether a worse temporal estimator helps the ensemble. Standalone R² is not sufficient to decide this. With base error `e_b`, challenger error `e_c`, and `C = E[e_b e_c]`, a small challenger weight can reduce MSE when `C < E[e_b²]`. The weight must be selected inside training data and the complete combination evaluated on held-out UAVs.

The new saved-prediction screen aligned all 2,000 development rows by outer fold, inner fold, UAV, scenario, cutoff and raw target:

| Base | Challenger | Optimistic fitted weight | Pooled RMSE before → after | Optimistic gain |
| --- | --- | ---: | ---: | ---: |
| PE_3 calibrated blend | LSTM | 7.0% | 11.5309 → 11.4682 | 0.54% |
| PE_11 residual-corrected | LSTM | 10.7% | 11.0404 → 10.8645 | **1.59%** |
| PE_11 residual-corrected | TCN | 9.6% | 11.0404 → 10.8733 | **1.51%** |

These weights were fitted and scored on the same reused development labels. They are **optimistic diagnostics, not validation results or deployable weights**. They also concern PE_11 development predictions, not new Phase 3 Run 7 temporal predictions. They justify at most a two-challenger, small-weight nested test; they do not support another broad temporal search. Start with weights `{0, 0.05, 0.10, 0.15}` and allow zero to win. [Reproducible screen](C:/Users/joest/UAV-RUL-Estimation/literature_and_planning/development_documentation/r2_research_2026_09_07/optimistic_blend_screen.csv).

**Common experiment contract and stopping rules**

1. Freeze Run 7 and retain Run 6 as the historical control. Store exact feature, checkpoint, prediction and cutoff provenance. Run 7's public score is now recorded as 0.87652 from the user's screenshot; no duplicate submission is needed to establish that baseline.
2. Keep fitting-target cap 125 as the nominal control. Evaluate against unchanged raw RUL labels. Report nominal and unrestricted-support stress results separately.
3. Keep all windows/prefixes of one UAV in a single evaluation group. Nest feature learning, base-model selection, early stopping, residual construction and blend selection. New split seeds measure robustness but do not erase prior use of the dataset.
4. Screen only the declared small recipes. Report pooled R²/RMSE, mean-fold metrics, paired differences, UAV-bootstrap intervals, short-history RMSE, target-band error shares and overprediction severity. Do not optimize a deployed rule using true test RUL bands.
5. Promote cheap additions at ≥1% mean paired RMSE improvement and four of five fold wins; require ≥2% for substantial new complexity. Confirm the selected recipes on two more grouped split seeds, report all results, and reject gains concentrated in one UAV or one seed. Report safety tradeoffs separately from the R² decision.
6. Combine only independently supported changes and re-evaluate the full combination. Seek a meaningful development margin above 0.9, for example around 0.92 under the fixed nominal profile, while treating that as a planning margin rather than a calibrated predictor of Kaggle performance. Do not force a lower confidence bound above 0.9 by repeated selection on the same data.
7. Freeze the final procedure before any genuinely held-back confirmation. Previously examined locked artifacts are historical evidence, not a reusable fresh holdout. Kaggle public feedback must not become the inner optimization loop.

**Execution status:** PE_14–PE_19 are complete. PE_18's full-feature TabPFN
blend passed the declared promotion gate, but its small gain and bootstrap
interval require multi-seed confirmation. PE_15's forecast-history candidate
still requires complete outer confirmation before a final combination. Run 7's
public result is confirmed; no duplicate submission is needed.

**Reproduce this report's new calculations**

```powershell
& .\.venv\Scripts\python.exe `
  .\literature_and_planning\development_documentation\r2_research_2026_09_07\analyze_saved_predictions.py
```

The script reads saved development predictions, verifies paired endpoint alignment, computes 3,000 UAV bootstrap resamples, and writes the adjacent JSON/CSV diagnostics. It does not retrain models or use test targets. The research framework and experiment fields are saved in `outline.yaml` and `fields.yaml` alongside this report.
