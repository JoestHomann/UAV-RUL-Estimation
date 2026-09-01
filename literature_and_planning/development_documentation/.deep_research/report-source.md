# Research source: reaching public R2 above 0.9

Research date: 2026-09-01

## Question and scope

Determine an evidence-backed, repository-specific route from the current best
Kaggle public `R2 = 0.86741` to `R2 > 0.9`. Preserve the competition boundary:
do not obtain or use the original NASA test RUL labels. Conservative prediction
remains a secondary operational objective, but the immediate research question
is the accuracy gap.

## Repository evidence

- Phase 3 Run 6 is the current best submission (`R2 = 0.86741`). It uses a
  calibrated 50/50 XGBoost-ExtraTrees blend, `screened_drift_pruned`, cap 125,
  and conditional quantile calibration at `q = 0.55`.
- Run 6 mean-fold development performance is `R2 = 0.89274`,
  `RMSE = 10.6332`, `MAE = 7.8348`, and bias `= -1.0718`.
- To move from public `0.86741` to `0.9` at fixed target variance requires an
  approximately `13.16%` RMSE reduction because
  `sqrt((1 - 0.9) / (1 - 0.86741)) = 0.86845`.
- Selected-candidate pooled OOF metrics are `R2 = 0.89736` and
  `RMSE = 10.6685` over 500 development endpoints.
- OOF error is concentrated in the ambiguous middle of the capped target:
  RMSE is `11.09` at true RUL 51-75, `11.16` at 76-100, and `12.11` at
  101-125. It is only `3.68` at RUL 0-25.
- OOF performance is weakest at short histories: cutoff at most 100 has
  `R2 = 0.184` and `RMSE = 12.49`; cutoffs 201-300 have `R2 = 0.952` and
  `RMSE = 8.21`.
- PE_5 leaderboard results reject uncapping as the primary remedy: hard cap 125
  scored `0.85609`, soft tail `0.83609`, weighted raw `0.83482`, and raw
  `0.82866`.
- Existing neural experiments used only 20 prefixes per UAV. Dense tabular
  prefixes were tested and rejected, but dense sliding-window sequence training
  was not tested. Therefore the existing LSTM/TCN/CNN results do not test the
  sample-construction protocol used by successful C-MAPSS sequence papers.

## New repository diagnostic

Compared each of the five 100-row development scenarios with the 100 unlabeled
test endpoints using the 298 `screened_drift_pruned` features. Cutoff
distributions are exactly matched. A repeated cross-validated ExtraTrees domain
classifier reached mean AUCs `0.785-0.843`; a stricter 10-fold OOF run reached
`0.821, 0.817, 0.845, 0.853, 0.835` (mean `0.834`). Cutoff-only classification
was effectively chance.

The most stable shifts include telemetry 16 state cardinality, telemetry 15/16
window-20 standard deviation, and telemetry 15/16 slope features. However,
Run 6 error decreases rather than increases with test-domain propensity:
correlation with absolute residual is `-0.184`, and the highest propensity
quintile has local `RMSE = 8.37`. This diagnoses covariate shift but does not
establish that importance weighting will improve target performance. The shift
could reflect an easier target composition, conditional shift, or a proxy that
does not align with public-label difficulty.

## Primary-source evidence ledger

| Source | Evidence | Limits for this project |
| --- | --- | --- |
| NASA C-MAPSS official dataset | Full training trajectories and truncated test histories define the prognostic task. | Original test targets are an experimental-integrity boundary. |
| Li, Ding, Sun (2018), RESS | A DCNN using normalized raw signals and time-window sample preparation performed strongly on C-MAPSS. | Published benchmark protocol is not identical to this Kaggle leaderboard. |
| Multi-scale LSTM (2021) | Uses sliding windows and multi-scale sequences to map degradation features to RUL. | Architecture alone does not isolate the effect of dense sample construction. |
| Hossain et al. (2026), arXiv | OOF stack of LSTM/CNN/CNN-LSTM/CNN-GRU with XGBoost meta-learner reports FD003 `R2 = 0.906`, `RMSE = 8.613`. | Four-day-old unreviewed preprint; potential protocol differences; treat as feasibility evidence only. |
| Arunan et al. (2024), Control Engineering Practice | Individual device change points used for RUL labels improved two multi-condition cases by 5.6% and 7.5%. | Strongest evidence is FD002/FD004-like multi-condition data, not single-condition FD003. |
| Sugiyama et al. (2007), JMLR | Importance-weighted CV can correct model selection under covariate shift assumptions. | Requires stable density ratios and unchanged conditional target distribution; neither is established here. |
| Le Xuan et al. (2024), RESS | Self-supervised domain adaptation incorporates unlabeled target-domain information and improves C-MAPSS results. | More complex than a diagnostic; competition rules must permit transductive use of test telemetry. |
| XGBoost official documentation | Supports named monotonic constraints; `hist` constraints may require larger `max_bin`. | Raw sensor direction can depend on fault mode, so constraints need a reliable health index. |
| Change-point and trajectory-similarity literature | Individualized degradation onset and learned health trajectories can improve RUL estimation. | Existing raw multichannel DTW failure does not test a learned monotonic health index, but this branch is less direct than dense sequence learning. |

## Claim gap matrix

| Claim | Repository evidence | External evidence | Confidence |
| --- | --- | --- | --- |
| The cap should be removed | Direct leaderboard evidence contradicts it. | Piecewise caps are standard but not universal. | High confidence: do not prioritize uncapping. |
| More tree tuning can close 0.033 R2 | Multiple tree searches and blends plateau around 0.865-0.867 public. | No source suggests hyperparameter search alone supplies a 13% RMSE reduction. | High confidence: low-value path. |
| Dense temporal learning is the most promising missing test | Existing sequence models saw only sparse 20-prefix data; errors are largest where temporal degradation state is ambiguous. | Multiple primary C-MAPSS methods depend on dense sliding windows; one new stack reports FD003 above 0.9. | Medium-high; benchmark comparability remains uncertain. |
| Domain shift contributes to the leaderboard gap | Domain AUC is 0.834 after cutoff matching. | Covariate-shift and domain-adaptation literature supports the mechanism. | Medium; local propensity does not predict harder rows. |
| OOF stacking can cross 0.9 | Fixed tree averaging is already near its ceiling; no strong sequence component exists yet. | New unreviewed FD003 stack reports 0.906. | Medium-low until a complementary dense model passes local gates. |
| Personalized degradation onset can improve the middle-RUL region | Mid-RUL and short-history errors dominate; fixed cap onset is global. | Change-point labeling improves related C-MAPSS subsets. | Medium. |
| GAN augmentation should be next | No repository evidence of a sample-scarcity mechanism that GANs solve. | Literature benefits are conditional on domain or training deficiency. | Low. |

## Reconciled conclusion

The route to `R2 > 0.9` is not another cap, calibration scalar, or broader tree
search. The required public improvement corresponds to a roughly 13% RMSE
reduction. The only untested intervention with that scale of plausible benefit
is dense, grouped, sequence-window learning followed, only if complementary,
by leakage-free OOF stacking with the current tree blend. Domain diagnostics
and individualized degradation onset are the two supporting experiments most
likely to improve generalization. Safety calibration should be frozen during
accuracy experiments and re-applied only after the predictive model is chosen.
