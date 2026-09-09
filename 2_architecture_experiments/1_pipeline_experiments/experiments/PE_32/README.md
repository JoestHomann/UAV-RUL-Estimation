# PE_32 — bounded LightGBM comparison

Test whether LightGBM improves the complete predictor relative to the retained
Run 7 residual-corrected XGBoost/ExtraTrees ensemble. This is the next tree
comparison; PE_33 is a separate, optional neural pilot.

Run from the repository root in PowerShell:

```powershell
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_32\run.py
```

This validates inputs, runs the screen, and runs confirmation only if the screen
passes. Add `--only validate_inputs` for a preflight with no model fitting.
The full command resumes completed fit cells after an interruption. Only the
interrupted model fit is repeated. Settings, code, dependencies and input hashes
are registered before fitting; changes require a new `pipeline.run` directory.
Do not start two copies against the same run directory.

## Frozen comparison

- Same 298 features, 20 training prefixes per UAV, row weight 0.05, and cap125
  fitting target as Run 7. Actual row weights are retained, not rescaled to one.
- Four recipes: 7 or 15 leaves crossed with minimum leaf counts 20 or 80.
  Learning rate .03, feature fraction .9, L2 1, minimum Hessian mass .001.
- Up to 2,000 boosting rounds. A deterministic 20% of the current training UAVs
  supplies historical stopping endpoints; patience is 75 rounds. Refit on all
  current training UAVs for the chosen number of rounds. Stopping and evaluation
  use raw endpoint RUL; regression fitting uses capped RUL.
- Five outer folds on the screening partition; two LightGBM seeds, averaged at
  prediction time. Run 7 retains its own frozen three-member recipe.
- For each recipe and candidate seed, select a blend weight from
  `[0, .1, .25, .5, 1]` using three grouped inner folds of the outer-training UAVs.
  The historical RMSE objective is subject to nominal/stress constraints. Zero
  weight retains Run 7. Outer labels never select a fold's blend weight.

The screen reports all standalone recipes and their complete blends. At least
2% mean-fold historical RMSE improvement and four of five fold wins are required
to continue, with at most 1% nominal and 5% unrestricted RMSE regression. Freeze
the best eligible blend recipe; confirm only that recipe on two additional
five-fold partitions with the same two-seed procedure. Blend weights continue to
be selected inside each confirmation training fold using the frozen rule.

Final review requires at least 2% mean-fold historical RMSE improvement, eight of
ten fold wins, historical pooled R² >= .90, an upper 95% paired UAV-bootstrap
RMSE-change bound below zero, and the same nominal/stress constraints. Failure
retains Run 7. Passing creates a review verdict; it does not replace production
or write a submission automatically.

## Cost and parallel execution

Two outer tasks run in separate Windows processes; numerical thread pools and
LightGBM are limited to four CPU threads per worker. Run 7 keeps its existing
device policy, so its XGBoost component can still use CUDA. Set `max_workers = 1`
before beginning a new run if running alongside PE_31 would compete for resources.

| Stage | LightGBM fits, including stopping/refit | Run 7 evaluations | Run 7 base-estimator fits |
|---|---:|---:|---:|
| Screen | at most 320 | 20 | 600 |
| Confirmation, only after screen passes | at most 160 | 40 | 1,200 |

Each Run 7 evaluation includes its existing internal calibration and residual
correction. Base-estimator counts cover XGBoost/ExtraTrees, not the smaller
calibration heads. Controls are cached across all recipes and candidate seeds.
No refit is performed per endpoint draw. The grid is small; the nested control
fits still make this a substantive run, not a quick four-fit benchmark.

## Results and interpretation

Results are under `runs/run_1/` (or the configured run name):

- `reporting/input_verification.json`: input readiness and fit budget.
- `reporting/pre_registration.json`: frozen provenance, created when fitting starts.
- `reporting/screen_selection.json`: frozen recipe choice and prediction checksum.
- `reporting/winner_manifest.json`: overall verdict; `fit_costs.csv`: actual work.
- `stages/screen/reporting/` and, if eligible, `stages/confirmation/reporting/`:
  summary, fold and seed metrics, predictions, paired comparisons and regions.
- `cells/`: checked predictions, stopping audits, and training-only blend choices.

Historical (500 rows), nominal (300), and unrestricted (300) suites are scored
separately with equal total weight per UAV. Endpoint generation follows PE_31.
The bootstrap resamples entire UAVs and is conditional on the fitted predictions.
All these UAVs have influenced previous research: different partitions measure
robustness and do not create an untouched external test set. Local R² >= .90
does not establish leaderboard R² >= .90.

To run individual stages directly:

```powershell
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\run_bounded_comparison.py --config .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_32\settings.toml --workflow PE_32 --stage screen
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\run_bounded_comparison.py --config .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_32\settings.toml --workflow PE_32 --stage confirm
```

Confirmation revalidates screening checkpoints and the frozen choice. It refuses
an incomplete screen. The deleted alternative script is explicitly not required:
this experiment builds only our Run 7 representation and does not replay that script.
