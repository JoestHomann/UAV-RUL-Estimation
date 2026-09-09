# FD003 codebase research and proposed experiments

Research date: 9 September 2026. “CMAPSS fold 3” is interpreted as **C-MAPSS FD003**, not the third cross-validation fold. This is a static source audit and experiment proposal; no external models were trained and PE_31 was not changed. Dataset identity with the competition has not been established, so NASA sensor numbers must not be copied onto our telemetry columns.

## Recommendation

Prioritize a **bounded LightGBM comparison (PE_32)** against Run 7. Our weak TCN results do not support a major new neural campaign: contrastive learning is now an **optional two-fold futility pilot (PE_33)**. Temporal-distance pretraining and cross-time sensor graphs remain deferred ideas. This priority revision incorporates the user's correction about our existing TCN evidence; external loss ablations alone do not outweigh it.

Implementation and run instructions: [PE_32](../../../2_architecture_experiments/1_pipeline_experiments/experiments/PE_32/README.md) and [PE_33](../../../2_architecture_experiments/1_pipeline_experiments/experiments/PE_33/README.md). The source audit below is unchanged; implemented budgets and gates supersede the initial broader proposal.

No inspected implementation establishes that our public R² will exceed .90. On an unchanged leaderboard target set, moving from .87652 to .90 requires about **19.0% less squared error, or 10.0% less RMSE**. From .87877 it requires **17.5% less squared error, or 9.18% less RMSE**:

`RMSE_new / RMSE_old = sqrt((1 - R2_new) / (1 - R2_old))`.

This calculation does not require the hidden labels. It also explains why repeated 0.5–2% local gains are unlikely to close the gap individually.

## What the source code actually supports

“Reported” below means the author's result, not a reproduction by us. Scores across rows have different targets and evaluation populations and must not be ranked as one leaderboard.

| Codebase | Relevant evidence | Audit verdict / transfer |
|---|---|---|
| [FSGRI / Dual-Mixer](https://github.com/fuen1590/PhmDeepLearningProjects) | FD003 LSTM normalized RMSE .1116 → .0938 with FSGRI; Dual-Mixer .0887 → .0887; DAMCNN worsens. | Best concrete loss ablation. Engine-separated validation and train-only scaling verified; all-window normalized evaluation differs from ours. |
| [Self-supervised SSL](https://github.com/tilman151/self-supervised-ssl) | Temporal-distance pretraining before RUL fine-tuning. FD003 fully labeled baseline 13.88 ± .58; outcomes depend on the pretraining scenario. | Promising secondary mechanism; paper and runner inspected, full preprocessing path not verified. |
| [FC-STGNN](https://github.com/Frank-Wang-oss/FCSTGNN) | Paper reports FD003 RMSE 11.52 ± .19; connects sensors across different times. | Distinct from our static sensor graph, but released training code selects minimum test RMSE. Borrow mechanism, not reported performance. |
| [Interpretable tree ensemble](https://github.com/hkmtcn/interpretable-rul-maintenance) | FD003 notebook outputs R² .9904 and MSE 94.2051 for LightGBM + CatBoost + GBR. | Headline output is random cycle validation with engine ID retained, not unseen-engine endpoint performance. |
| [Turbofan RUL MLOps](https://github.com/virgil-castillo/turbofan-rul-mlops) | Reports final-endpoint FD003 GRU RMSE 14.22 ± .17 and LSTM 13.98 ± .30. | Useful evaluation reference. Its FD003 feature screen does not support indiscriminate feature expansion. |
| [Kalman-DVAE](https://github.com/StarMarco/Kalman_DVAE) | Explicit latent state dynamics and predictive uncertainty. | Random-window validation; different aggregate metric paths; no justification for another full filtering campaign. |
| [PCLSSN](https://github.com/zhirongzhong/PCLSSN) | Physics-consistent liquid state-space network; author paper includes FD003. | Released runner defaults to IGBT, reads prepared MAT splits and averages trajectory metrics. FD003 endpoint reproduction is unverified. |
| [Frequency-masked Embedding Inference](https://github.com/USTBInnovationPark/Frequency-masked-Embedding-Inference) | Frequency masking and EMA encoder pretraining. | Paper's FD003 regression uses 13,196 windows and a RUL-ratio task; not evidence of competitive raw endpoint RUL. |
| [C-MAPSS prediction toolkit](https://github.com/Jiajun-H/cmaps-rul-prediction) | Reports FD003 LSTM RMSE 12.20 / R² .789 and attention variant 12.49 / .779. | Supports caution about architectural complexity. README/results inspected; complete training protocol not audited. |
| [Kaggle LSTM notebook](https://www.kaggle.com/code/yahyamomtaz/rul-prediction-using-lstm-for-aircraft-engine) | Relevant notebook found through Kaggle search. | Accessible page exposed metadata, not executable cells; FD003 and evaluation could not be verified. Excluded from performance evidence. |

### 1. FSGRI: the useful part is the learning signal

The [paper's Table 4](https://arxiv.org/html/2401.16462v1#S4.T4) contains both improvements and failures on FD003. The LSTM improvement is roughly 16%, but the best Dual-Mixer gains nothing from the loss. This supports testing the training mechanism on a fixed backbone, rather than assuming the entire proposed architecture will win.

The [loss implementation](https://github.com/fuen1590/PhmDeepLearningProjects/blob/master/models/RULPrediction/ContrastiveModules.py) combines MSE with weighted InfoNCE. It augments an anchor with Gaussian noise and weights negatives using RUL differences. The intended effect is to make the learned representation reflect degradation progression.

[Preprocessing](https://github.com/fuen1590/PhmDeepLearningProjects/blob/master/dataset/cmapss.py) separates engines before scaling, but generates every test window and normalizes capped labels. It also keeps engine 1 out of validation. [Training](https://github.com/fuen1590/PhmDeepLearningProjects/blob/master/train/trainable.py) monitors validation loss; the best checkpoint is restored when patience triggers, but not unconditionally at the epoch limit. [Defaults](https://github.com/fuen1590/PhmDeepLearningProjects/blob/master/models/RULPrediction/experiments.py) select FD004 and disable contrastive training.

**Adaptation:** use our grouped partitions, fixed cycle-based target scale, prefix masks and raw endpoint evaluation. Select the epoch on training-side UAVs, then refit for that epoch count. Match augmentation and sample exposure in controls so that a gain can be attributed to the loss.

### 2. The apparent R² .99 tree result does not solve our problem

The [FD003 notebook cell](https://github.com/hkmtcn/interpretable-rul-maintenance/blob/main/notebooks/LGMB_CatBoost_GBR.ipynb) constructs X by removing only RUL, leaving engine ID and cycle. It applies random 80/20 row splitting and reports metrics against that validation subset. Its stored MSE 94.2051 corresponds to RMSE 9.706, matching the headline 9.71.

Cycles from the same engines therefore occur on both sides of the split; engine identity provides information unavailable for a new UAV. The notebook also contains grouped analyses, and the separate [PGTS protocol](https://github.com/hkmtcn/interpretable-rul-maintenance/blob/main/notebooks/PGTS_PROTOCOL.md) explicitly groups whole engines. Those additional checks do not change the provenance of the headline output. Do not reproduce the random-row/ID recipe.

LightGBM remains a reasonable cheap candidate because its tree construction differs from our existing implementations, but **the .99 result is not supporting evidence that it will improve us**.

### 3. FC-STGNN: interesting relationships, optimistic selection code

The [paper](https://arxiv.org/pdf/2309.05305.pdf) reports FD003 RMSE 11.92 without its full cross-time graph construction versus 11.52 for the complete model. Its graph links different sensors at different timestamps; our existing sensor_graph_tcn first mixes sensors at each timestep using a fold-fitted correlation graph.

However, [Train_model](https://github.com/Frank-Wang-oss/FCSTGNN/blob/main/FC_STGNN/main_RUL.py) evaluates test data at validation-improving epochs and then selects `np.argmin(test_RMSE)`. The [loader](https://github.com/Frank-Wang-oss/FCSTGNN/blob/main/FC_STGNN/data_loader_RUL.py) also shuffles windows into validation after scaling. Thus the released workflow cannot supply an unbiased unseen-engine estimate. Test-label checkpoint selection must be removed, as must overlapping-engine validation.

**Adaptation:** one small graph comparison with identical inputs, parameter budget and training protocol, changing only same-time versus cross-time sensor connections. This is a conditional experiment, not the first large run.

### 4. Temporal-distance pretraining: a second specific learning signal

The [author paper](https://papers.phmsociety.org/index.php/ijphm/article/download/3096/1891) learns the time separation between two windows before fine-tuning on RUL. Table 4 reports a fully labeled FD003 baseline of 13.88 ± .58; the 80%-degradation pretraining scenario reaches 13.05 ± .53, while the 90% scenario reaches 13.83 ± .49. These are different pretraining scenarios, not a guaranteed six-percent improvement.

The [runner](https://github.com/tilman151/self-supervised-ssl/blob/master/src/run_pretraining.py) exposes metric and autoencoder modes. Only the runner and paper were verified in this audit.

We already have complete labeled histories, so the value would come from a better learning objective rather than newly acquired labels. Pretrain only on the current training UAVs; use the same encoder, fine-tuning data and budget as a supervised control. Elapsed time is an imperfect degradation proxy during the capped healthy plateau.

### Other findings that change priorities

The [FD003 feature screen](https://github.com/virgil-castillo/turbofan-rul-mlops/blob/main/docs/feature_family_screen_report.md) favors raw inputs for its GRU at sequence length 60: adding a feature family does not beat raw there. Some gains discussed in that repository occur on other subsets and must not be generalized to FD003. This reinforces the need for matched ablations; it does not invalidate useful engineered features in our tree models.

[Kalman-DVAE preprocessing](https://github.com/StarMarco/Kalman_DVAE/blob/main/kalman_ruls/data/dataprep.py) randomly holds out overlapping windows after normalization. Its [testing code](https://github.com/StarMarco/Kalman_DVAE/blob/main/testing.py) has all-time RMSE and a final-endpoint helper named RMSE that averages absolute errors for scalar endpoints. Label-conditioned filtered diagnostics are distinct from its unconditioned predictions. Do not treat every plotted curve or metric name as a deployable endpoint result.

[PCLSSN's runner](https://github.com/zhirongzhong/PCLSSN/blob/main/scripts/run_experiments.py) has validation-based checkpoint selection, but its supplied IGBT configuration and trajectory-averaged reporting leave a substantial FD003 reproduction gap. [FEI's paper](https://arxiv.org/html/2412.20790v2) uses external SleepEEG pretraining and a single-channel, ratio-based C-MAPSS transfer task. Both remain research references rather than ready replacements.

## Proposed experiments

E2 is implemented as PE_32; the reduced E1 pilot is implemented as PE_33. E3/E4 remain proposals. Complete PE_31 and review its traceable submission replay before starting another large campaign. Preserve unchanged Run 7 as a control even if PE_31 identifies another candidate.

### E1 — Degradation-aware contrastive training: optional futility pilot (PE_33)

**Question:** do our sequence models fail partly because endpoint MSE alone does not sufficiently organize the degradation representation?

Freeze one small TCN using our existing residual-block implementation, its 50-cycle input, channels, age side inputs, masks, training rows and UAV weights. Do not simultaneously introduce Dual-Mixer. Compare three arms:

1. Current supervised MSE.
2. Identical sampled windows and noise augmentation, MSE only.
3. Arm 2 plus weighted contrastive loss, coefficient .1.

Use `min(RUL,125)/125` for the training MSE scale; retain raw RUL for evaluation. These coefficients are proposed starting values, not transferred performance guarantees. Sample within training engines, balance total contribution by UAV, and keep sample exposure matched across arms. Noise must be scaled from training data and excluded from discrete channels, age and padding. Do not invent a time-warped RUL label.

Compare each arm standalone and as a fixed 10% blend with Run 7 on two predeclared outer folds, one model seed, at most 30 epochs per fit. If augmented MSE matches the contrastive arm, the information gain is augmentation, not contrastive learning. Require at least 5% standalone RMSE improvement over both controls and at least 1% complete-blend improvement with two fold wins, plus nominal/stress safeguards. Stop after the pilot regardless of outcome; success only justifies proposing further work.

### E2 — Regularized LightGBM: next bounded comparison (PE_32)

Keep the 298-feature Run 7 representation, cap125, existing prefix policy and total UAV weighting fixed. Compare four bounded LightGBM configurations: 7 or 15 leaves crossed with minimum leaf counts 20 or 80. Choose boosting duration on separate training-side UAVs. Document actual weight scale; do not copy unexamined regularization defaults across sparse/dense policies.

Evaluate standalone and convex-blended predictions with Run 7, with all blend choices inside training folds. HistGradientBoosting has already been studied, so this is incremental learner diversity, not a wholly new modeling direction. If it supplies no paired ensemble gain, stop; do not launch a large LightGBM hyperparameter sweep.

### E3 — Temporal-distance pretraining: second representation experiment

Reuse E1's encoder and input contract. Compare supervised training against temporal-distance pretraining followed by identical supervised fine-tuning. Add an equal-total-training-budget supervised control so that extra optimization is not mistaken for pretraining value. Cap pretraining at one fixed budget and one declared pair sampler.

Use only training-fold engines, with per-engine pair balancing. Do not pretrain on outer-held or competition-test telemetry. Retain the healthy plateau in the analysis rather than choosing its handling after seeing outer outcomes. This is lower priority because we have no extra unlabeled fleet and the published gain depends on data availability.

### E4 — Cross-time sensor graph: conditional extension

Compare an existing same-time sensor-graph encoder with a matched version allowing cross-time edges within the observed 50-cycle window. Keep the head, target, sampler and approximate parameter count fixed. Use one decay setting, such as the author's .7, rather than a graph/grid search.

This tests delayed cross-sensor relationships that summary features or same-time mixing may miss. It does not repeat the question “are GNNs better?” Proceed only if the small neural controls converge and E1/E3 show meaningful standalone or ensemble utility. All windows must end at the requested cutoff; no graph may cross UAV boundaries.

## Evaluation and stopping contract

Use PE_31's exported endpoint definitions and grouped evaluation principles in a new frozen campaign. A cached control is reusable only when its complete fit contract, fold identities and endpoints match.

- Keep historical, fresh nominal and unrestricted raw-label suites separate. Report pooled R² and RMSE with their sample counts; do not mix their different target variances.
- Fit scalers, sampling rules, stopping choices, contrastive pairs, model choices and blend weights entirely inside training UAVs. Generate training-side out-of-fold predictions for blending. Validation labels do not enter pretraining.
- For E2, allow four declared recipes in a five-fold, two-model-seed screen. Select at most one challenger and confirm it on two other grouped partitions and two model seeds. E1 is limited to three arms, two folds and one seed; it has no automatic confirmation. E3/E4 remain deferred.
- Those are recipe-evaluation budgets, **not estimator-fit counts**. Preflight must account for all inner blending, stopping and refitting work, reusing exact matching controls, before launching. Do not run all four families simultaneously.
- Adopt a prospective promotion gate consistent with PE_31: at least 2% mean-fold RMSE improvement for the complete model; historical pooled R² at least .90; paired UAV-bootstrap 95% RMSE-change interval below zero; nominal regression at most 1% and unrestricted regression at most 5%.
- Inspect history ≤100, RUL regions and per-UAV SSE contributions. Labels define diagnostic groups only, never inference routing. The implementation exports region metrics and predictions; any later removal-of-one-UAV sensitivity analysis must be labelled exploratory and must not delete that UAV from the primary score.
- Report residual covariance and actual blend gain. Low residual correlation alone does not establish usefulness.
- Freeze the chosen artifact and submission hash before one leaderboard confirmation. The original simpler-script replay is still needed to connect its historical .87877 to a reproducible file.

All existing UAVs have influenced our research decisions. New partitions and bootstrap intervals provide conditional robustness evidence; they do not create an untouched external test set. These restrictions are motivated by actual code-audit findings and our PE_28 reversal, not merely by convention.

## Experiments not worth repeating now

PE_15 filtering's apparent gain failed complete confirmation in PE_20 (1.56% RMSE regression). PE_26's short-history specialist failed promotion. PE_27 did not support broad UAV-subset bagging. PE_28's compact feature winner reversed during confirmation; PE_29 weight-scale correction and PE_30 dense equal-UAV training did not help. PE_31 is already examining sampling, weight scale and capacity interactions.

Likewise, generic LSTM/GRU/TCN, multiscale CNN, static sensor graphs and tabular-sequence hybrids already exist in our architecture studies. Reopening them requires a new controlled mechanism, such as E1 or E3. Do not count more windows, random-row validation, raw-versus-capped evaluation changes, NASA-score optimization, or original FD003 test answers as progress toward our competition goal.

The current sequence is: **finish PE_31 and its submission provenance check → bounded PE_32 → optional PE_33 pilot**. Further neural work depends on evidence from that pilot. Neither the literature nor our results justify calling a large contrastive campaign the strongest next investment.

## Evidence and limits

This report combines public source inspection with our local PE_15/20/26–31 documentation and architecture settings. Public search covered GitHub, Kaggle and author papers without a date restriction. It is a targeted review, not an exhaustive census. No upstream results were rerun. Kaggle code access and several complete reproduction paths remain unverified.

[code_audit.json](code_audit.json) records inspected source paths and available Git blob hashes. [outline.yaml](outline.yaml) and [fields.yaml](fields.yaml) record scope, research objects and review fields. No training data, official test labels or pretrained model weights were imported into this project.
