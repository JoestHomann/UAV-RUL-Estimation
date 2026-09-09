# PE_29: weight scaling before training density

PE_28's compact feature winner failed confirmation. Its XGBoost–CatBoost recipe also lost to Run 7 locally, despite the alternative script's reported public score of 0.87877. PE_29 tests a specific confound: the alternative script uses unit row weights, whereas our 20-prefix data assigns each row weight 0.05. Copying XGBoost's regularization parameters unchanged changes their strength relative to the training loss. This is a hypothesis about the discrepancy, not an established explanation of the leaderboard result.

## Fixed comparisons and gates

| Stage | Comparison | New recipe evaluations | Gate |
|---|---|---:|---|
| Audit | Adjusted XGBoost + unchanged CatBoost versus PE_28's saved simple B baseline, on the same 2,000 prefixes and 266 features | 5 | At least 2% lower mean fold RMSE and 4/5 fold wins |
| Density, only after audit passes | All 24,720 cycles versus the adjusted 2,000-prefix recipe | 5 | Same gate; select dense only if it passes, otherwise retain sparse |
| Reference | Selected candidate versus saved Run 7 on those same screening folds | 0 | Same gate |
| Confirmation, only after reference passes | One frozen candidate versus freshly fitted Run 7 | 20 | At least 2% lower mean fold RMSE, 8/10 wins, pooled R² ≥ 0.90, paired UAV bootstrap upper bound for RMSE change below zero |

The first stage reuses ten verified PE_28 cells (five simple B and five Run 7). The run needs five new recipe evaluations initially and at most thirty overall. Each simple recipe evaluation includes nested stopping, calibration and refits: 18 base estimator fits. A Run 7 evaluation includes 30 base estimator fits and a residual head. These are not thirty individual model fits. Failed gates stop dependent stages; failed confirmation does not trigger a fallback candidate.

## What changes

For uniformly scaled weights `w' = c*w`, XGBoost's gradient and Hessian sums scale by `c`. Scaling `reg_lambda`, `reg_alpha`, `gamma` and `min_child_weight` by the same factor preserves their scale relative to the loss (subject to numerical and implementation details). For `c = 0.05`, the copied recipe's effective settings become `reg_lambda = 0.05`, `min_child_weight = 0.05`, `reg_alpha = 0`, `gamma = 0`. There is no parameter search. See the [XGBoost parameter definitions](https://xgboost.readthedocs.io/en/latest/parameter.html).

CatBoost, feature definitions, target cap 125, model seeds, inner selection and stopping procedures remain those registered by PE_28. All scoring uses raw endpoint RUL. Calibration and stopping UAVs stay inside each outer training fold.

The density stage assigns each cycle weight `1 / number_of_cycles_in_its_UAV`, retaining total weight one per UAV and the same adjusted XGBoost parameters. It includes RUL-zero terminal cycles and changes both the number and distribution of training cutoffs. It therefore tests the all-cycle sampling policy, not density in isolation. It does not reproduce the alternative script's unit-row policy, which gives longer-lived UAVs more total weight. Features are computed from observed prefixes only.

Screening uses PE_28's seed 20270107. Confirmation uses new seeds 20270217 and 20270227, with five outer folds each. Screening results are exploratory because these endpoints informed the hypothesis. New confirmation partitions still reuse the same UAV population; they are not untouched external evidence. Neither test labels nor locked evaluation labels are used. No Kaggle submission or production replacement happens automatically.

## Run and resume

From the repository root in PowerShell:

```powershell
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_29\run.py
```

Two worker processes evaluate independent folds concurrently. Each has a four-thread numerical-library limit; inherited XGBoost uses one thread and CatBoost four. All-cycle feature construction occurs once in the parent and is conditional on the audit gate. Worker failure stops new dispatch and lets already-running cells finish their checkpoints.

Rerun the same command after interruption. Completed cells are verified and reused. Source data, PE_28 registered code, package versions, reference predictions, PE_29 code and configuration are checked before reuse. Checkpoints also hash training feature values, targets and weights. Changes to a registered contract are rejected; use a new run directory for a changed experiment.

Preflight without training:

```powershell
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\run_weight_scale_audit.py --config .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_29\settings.toml --check
```

Read `runs/run_1/reporting/winner_manifest.json` for the final verdict, `reporting/selected_configuration.json` for the frozen candidate, and `stages/<stage>/reporting/` for fold metrics, pooled metrics, regional errors and paired bootstrap comparisons. Cell audit records include the actual XGBoost parameters, weight range, training row count and fit timing.
