# PE_34: adaptive UAV weighting pilot

Test whether giving more training emphasis to poorly predicted UAVs improves
the complete Run 7 predictor. This is a single reweighting pass with three
fixed arms: unchanged Run 7, 1.5x relative weight and 2x relative weight for the
hardest quarter of training UAVs. The simpler pipeline is an additional matched
baseline. Features, fitting cap125, model capacities and iterations remain fixed.

## Difficulty and leakage boundaries

For each actual member-fitting set, three grouped folds train unchanged Run 7
and predict that set's historical endpoints out of fold. Rank UAVs by their
mean squared error (equivalently RMSE) over those endpoints. Select the largest
`ceil(0.25 * N)` scores; exact ties use ascending UAV ID. Multiply those UAVs'
prefix weights by 1.5 or 2 and renormalize to the original total weight for
that actual fitting set. Relative weight increases do not change global weight
scale or allow long histories to dominate through row count.

Difficulty must be computed again for each of Run 7's four calibration member
fits using only that fit's available training UAVs. A single outer-level set of
weights would let calibration-held labels influence those fits. Final members
use difficulty estimated on the full outer-training set. The unweighted
difficulty models do not invoke adaptive weighting recursively.

Only the base estimators' training weights change. Run 7's family-blend and
residual-calibration objectives retain their original weights and procedure,
using OOF predictions from the adaptively weighted member fits. This pilot
does not also reweight the residual head or add adaptive blend routing.

The two weighted arms share difficulty estimates. All evaluation scores retain
equal total weight per UAV; the difficult subgroup's score is not the objective
used to decide success. Outer-held labels are never used in difficulty fitting.

## Evaluation and budget

- Five outer folds on PE_32's screening partition, split seed 20270910.
- Same historical, nominal and unrestricted endpoints as PE_32, scored separately.
- Frozen Run 7 member seeds 13, 37 and 73; no extra candidate-seed sweep.
- Verified reuse of PE_32's 20 Run 7 evaluations and five simpler evaluations:
  **660 base fits saved**, including the outer-training difficulty fits.
- At most 60 additional unchanged Run 7 evaluations for internal difficulty:
  **1,800 base fits**. Ten weighted full evaluations add **300 base fits**.
- At most **2,100 new XGBoost/ExtraTrees base fits**, excluding small residual
  heads. No new CatBoost fits. This is a substantive nested pilot.
- Two outer workers, four numerical CPU threads each; the frozen Run 7 GPU
  policy is preserved. Cells and difficulty decisions are checkpointed.

These are previously inspected UAVs and the already used PE_32 partition.
This is an exploratory pilot, not untouched confirmation. Reusing predictions
does not make any previously inspected data fresh. Even a successful arm must
be separately confirmed; there is no automatic expansion or production change.

An arm is worth confirmation only if it improves historical mean-fold RMSE
by at least 2% over Run 7, wins at least four of five folds, has an upper 95%
paired-UAV bootstrap RMSE-change bound below zero, beats the simpler baseline's
historical RMSE, and regresses by no more than 1% nominal / 5% unrestricted.
The bootstrap is conditional on fitted predictions and excludes adaptive-history
uncertainty. Report R² but do not claim a local threshold establishes Kaggle R².

## Run and inspect

From the repository root:

```powershell
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_34\run.py
```

Add `--only validate_inputs` for a preflight. Preflight validates and copies
compatible baseline cells and freezes their input registration without fitting
new models. Re-run the full command to resume. Only one process may hold the
run lock. Changed scientific settings/code require a new run directory.

Artifacts under `runs/run_1/`:

- `reporting/input_verification.json`: readiness, verified reuse and fit budget.
- `reporting/reused_cells.csv`: source paths and SHA-256 for copied PE_32 cells.
- `cells/`: reused and newly fitted unchanged Run 7 prediction checkpoints.
- `adaptive_cells/`: difficulty rankings, fit/held UAV provenance, normalized
  weights and complete weighted-model predictions. Completed difficulty work is
  reused when resuming an interrupted weighted model.
- `reporting/progress.json`: running/completed/failed status and finished folds.
- `reporting/fit_costs.csv`: completed fit counts, separating reused work.
- `stages/pilot/reporting/`: summary, regions, fold scores and paired comparisons
  against both baselines.
- `reporting/winner_manifest.json`: final pilot verdict, never automatic promotion.

Preparation verification includes unit tests for weight conservation, stable
UAV ranking, nested calibration isolation, outer-label perturbation, shared
difficulty caches, tamper rejection and pilot decision gates.

## Experimental Phase 3 submission

The pilot completed without promotion. The `adaptive_uav_1_5` arm was the
better treatment at mean-fold RMSE 10.2101, but it missed the preregistered 2%
historical-gain and paired-bootstrap gates. Phase 3 Run 8 therefore deploys it
only as a labeled leaderboard probe; Phase 3 Run 7 remains the retained model.

The current workspace has already completed Run 8 Step 1. Continue from the
repository root with:

```powershell
& .\.venv\Scripts\python.exe .\3_final_model_training_and_inference\run_phase_3.py --from-step 2
```

The verified Kaggle file will be written to
`3_final_model_training_and_inference/runs/run_8/6_submission_verification/artifacts/submission.csv`.
Run 8 recomputes fit-local grouped difficulty inside every residual-calibration
fit. Its Step 2 and Step 4 are consequently expensive and are resumable after
an interruption.
