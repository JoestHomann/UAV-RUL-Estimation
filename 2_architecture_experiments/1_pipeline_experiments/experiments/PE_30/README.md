# PE_30: all-cycle training with the original simple recipe

PE_29's weight adjustment made mean fold RMSE 0.32% worse and failed its first gate. Its all-cycle stage therefore never ran. PE_30 tests that remaining hypothesis independently, without changing PE_29's verdict or its registered files.

## Comparison

The candidate `original_dense` uses all 24,720 observed training cycles, including 100 terminal RUL-zero rows, with the original PE_28 XGBoost–CatBoost recipe and 266-feature B representation. The baseline uses 2,000 existing prefixes (20 per UAV). Each candidate row receives weight `1 / number_of_cycles_in_its_UAV`; both policies retain total weight one per UAV. More rows provide more training cutoffs, not more independent UAVs.

The original XGBoost regularization stays fixed: `reg_lambda = 1`, default `min_child_weight = 1`, and default `reg_alpha = gamma = 0`. PE_29's scaling correction is not applied. CatBoost parameters, fitting cap 125, seeds, inner fold count, and blend selection procedure remain those registered by PE_28. Each inner early-stopping fit uses the corresponding training sampling policy, and all stopping/calibration UAVs stay inside the outer training fold. Blend calibration and outer evaluation retain the existing sparse development endpoints and raw RUL labels.

Features use only observations at or before the row's cutoff. Full trajectory length determines training labels and training weights only; it is never a feature or inference input. All-cycle sampling changes both training density and the cutoff distribution, including terminal-cycle coverage. The experiment does not isolate these effects from one another, and it does not reproduce the alternative script's unit-row weighting that gives longer-lived UAVs more total influence.

## Fixed stages

| Stage | Comparison | New recipe evaluations | Gate |
|---|---|---:|---|
| Density | Dense candidate versus saved original sparse simple recipe | 5 | At least 2% lower mean fold RMSE and 4/5 fold wins |
| Reference | Same dense predictions versus saved original Run 7 | 0 | At least 2% lower mean fold RMSE and 4/5 fold wins |
| Confirmation | Fixed dense candidate versus freshly fitted unchanged Run 7 | 20, only if both gates pass | At least 2% lower mean fold RMSE, 8/10 wins, pooled R² ≥ 0.90, paired UAV-bootstrap 95% RMSE-change interval wholly below zero |

Both screening reports are written even if the density gate fails; comparing the saved predictions costs no additional fits. The run uses five new recipe evaluations initially and at most 25 overall, reusing ten verified PE_28 screening cells. Each simple evaluation includes 18 base estimator fits. Each Run 7 evaluation includes 30 base estimator fits and a residual head. Maximum cost is 570 base estimator fits plus ten residual heads. There is one candidate, no grid search and no fallback after failed confirmation.

Screening uses PE_28's seed 20270107 and is exploratory: these endpoints informed the hypothesis. Confirmation uses seeds 20270317 and 20270327, five outer folds each, after freezing the screening decision. These are new partitions of previously examined UAVs, not independent external data. Local R² ≥ 0.90 does not establish a leaderboard score ≥ 0.90. Test labels and locked evaluation labels are not opened. Production remains Run 7 until a separate final-model decision.

## Run

From the repository root in PowerShell:

```powershell
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_30\run.py
```

Two worker processes fit separate folds concurrently. Each worker has a four-thread numerical-library limit; the inherited simple recipe uses one XGBoost thread and four CatBoost threads. Feature construction runs in the parent before fitting. Preflight constructs and checks the complete dense feature matrix, so startup is longer than PE_29's. The full launcher constructs it once for preflight and once for evaluation.

Check inputs and features without fitting:

```powershell
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_30\run.py --only validate_inputs
```

Rerun the original command to resume completed cells. Source hashes, package versions, settings, feature/target/weight values and endpoint identities are checked before reuse. Unregistered or corrupted checkpoints are rejected. Worker failure stops new dispatch and allows already-running cells to finish checkpointing. `--force` recomputes cells only under an unchanged registration; changed experiments require a new `pipeline.run` directory.

Read `runs/run_1/reporting/winner_manifest.json` for the overall verdict, `reporting/selected_configuration.json` for both screening gate outcomes, and `stages/density`, `stages/reference`, and `stages/confirmation` for their reporting folders. Each contains metrics, paired decisions, regional errors and predictions, or an explicit skipped-stage verdict. Cell audit records include training rows, weights, actual recipe parameters, calibration membership and fit time.
