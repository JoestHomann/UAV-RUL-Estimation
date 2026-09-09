# PE_33 — optional contrastive TCN futility pilot

Our TCN's weak prior result makes a large contrastive campaign unjustified. This
pilot asks whether a changed learning signal is worth further investigation.
PE_32 never launches it. Run it separately, after reviewing higher-priority work:

```powershell
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_33\run.py
```

Add `--only validate_inputs` for no-fit input verification. The command resumes
completed cells. Changed code/settings/data require a new `pipeline.run`; do not
run two copies against the same directory.

## Fixed three-arm ablation

1. TCN with MSE.
2. Same TCN, Gaussian augmentation plus MSE.
3. Same augmentation and MSE, plus degradation-aware contrastive loss at weight .1.

All use the existing TCN residual-block implementation: 50-cycle left-padded
prefixes, 22 telemetry channels, two blocks of width32, kernel3, dilation base2,
dropout .1, and two age side inputs. This is a new fixed small control, not a
claim to reproduce the historical architecture search's selected checkpoint.
The comparison keeps the backbone fixed across all three arms.

Robust scalers see only the current fitting UAVs and only cycles observed in
their fitting prefixes. Sequences stop at each requested cutoff; later telemetry,
test telemetry and held-out labels are excluded from training. The 20 existing
prefixes per UAV remain the anchors. Equal prefix counts and within-UAV negative
sampling give equal total training contribution per UAV. MSE is averaged across
the matched sampled exposures; its scale is held constant across arms.

Positives reuse the anchor, with scaled Gaussian noise (SD .05) in continuous
telemetry 13,19,21,22,25,28 only. Padding, discrete channels and age remain intact.
A negative is sampled from the same training UAV where the capped normalized
RUL difference exceeds .08. If none exists, the anchor gets no contrastive term.
All arms see the same anchor/positive/negative windows and forward-pass shapes.
This avoids confusing extra supervised examples or BatchNorm exposure with a
contrastive benefit.

For L2-normalized telemetry embeddings, the positive logit is cosine similarity
divided by temperature .2. The negative logit adds the log of capped RUL distance.
The two-logit cross-entropy is the auxiliary loss. Age is excluded from this
embedding, although the regression head retains age inputs. This is a bounded
adaptation inspired by [FSGRI's loss](https://github.com/fuen1590/PhmDeepLearningProjects/blob/master/models/RULPrediction/ContrastiveModules.py),
not a reproduction of its multi-negative benchmark.

Adam uses learning rate .001, batch128 and gradient clipping5. Fitting MSE uses
`min(RUL,125)/125`. A separate 20% of training UAVs selects the epoch using raw
historical endpoint RMSE, with patience5 and a 30-epoch maximum. Refit from the
same seed on all training UAVs for the selected epoch count, including when the
best epoch precedes the limit. No final-epoch fallback replaces that choice.

## Hard limit and decision

Only outer folds0 and1 of the predeclared partition run, with one model seed.
There are at most **12 neural fits**, including stopping fits and refits, and two
Run 7 evaluations (60 XGBoost/ExtraTrees base-estimator fits plus calibration).
The pilot uses one worker; neural training is CPU-only with four threads. Run 7
retains its existing device policy and may use CUDA.

Each arm is evaluated standalone and in a **fixed 10% TCN / 90% Run 7 blend**.
No blend search or inner blending fits are performed. The pilot says
`worth_followup` only if all conditions pass:

- Contrastive standalone mean-fold historical RMSE improves by at least 5%
  against both MSE and augmentation-only controls.
- Its fixed complete blend improves historical mean-fold RMSE over Run 7 by at
  least 1%, winning both held-out folds.
- Nominal RMSE regression is at most 1%; unrestricted regression is at most 5%.

Otherwise the verdict is `stop_pilot`. Even success stops after these two folds;
there is no automatic expansion, promotion or submission. A failed gate is a
reason to stop this recipe, not proof that contrastive learning cannot work.
A passing gate only justifies proposing a separately registered larger study.

## Outputs

`runs/run_1/reporting/winner_manifest.json` holds the verdict and every gate.
`stages/pilot/reporting/` holds standalone/blended predictions, separate suite
metrics, per-fold results, paired comparisons and history/RUL-region diagnostics.
`cells/` records scaler/training membership through the frozen input contract,
stopping/refit UAVs, selected epoch and both training curves. Input provenance and
actual fit counts are in `reporting/`.

Use the matched controls to interpret any gain: improvement over plain MSE alone
does not show that contrastive learning helped. A better standalone TCN that
still damages Run 7's blend does not justify a major new campaign. Two folds and
one seed provide a futility check, not a dependable estimate of leaderboard gain.
