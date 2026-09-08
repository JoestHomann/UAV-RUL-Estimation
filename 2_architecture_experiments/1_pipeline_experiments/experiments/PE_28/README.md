# PE_28: feature representation and model comparison

PE_28 crosses ten frozen feature sets with two recipes: unchanged Run 7's
residual-corrected XGBoost/ExtraTrees ensemble, and an XGBoost/CatBoost blend
based on the implementation in `other_pipelines`. All twenty configurations
use the same 2,000 training prefixes, equal total training weight per UAV,
cap-125 fitting targets, raw evaluation labels, and 500 development endpoints.
Only training data is read; the experiment does not read test or locked tables.

## Feature sets

| ID | Representation | Features | Main comparison |
| --- | --- | ---: | --- |
| A | Existing drift-pruned representation | 298 | Unchanged Run 7 control |
| B | Other implementation: 22 sensors, windows 5/10/20 | 266 | Alternative representation |
| C | B restricted to our 14 sensors | 170 | C versus B: sensor coverage |
| D | B with windows 5/20/50 | 266 | D versus B: window choice |
| E | Six compact summaries on all 22 sensors | 134 | E versus B: summary complexity |
| F | B using the first available ten cycles as the baseline | 266 | F versus B: first observation versus early average |
| G | B with windows 5/10/20/50 | 310 | G versus B/D: complementary windows |
| H | E restricted to our 14 sensors | 86 | Completes coverage-by-complexity comparison |
| I | B restricted to sensors 07/13/19/21 | 50 | Four sensors already used by Run 7's residual correction |
| J | B plus 5-cycle and 20-cycle slopes on every sensor | 310 | J versus B: recent degradation rates |

Compact sets retain latest value, change from the first observation, full-history
slope, five-cycle mean, 20-cycle mean and 20-cycle sample standard deviation.
All sets retain cycle count and log cycle count. Baseline F uses only the first
`min(10, observed_history)` samples. Slopes are least-squares slopes; windows
shorten to the available prefix and single-observation standard deviations/slopes
are zero. No feature reads observations after its cutoff.

Sensor membership is frozen from the previously inspected implementations; no
ranking or correlation is recomputed using outer-held labels. The 22-sensor list
is 01/02/04/05/06/07/09/10/11/12/13/15/16/18/19/21/22/23/24/25/26/28.
The 14-sensor list is 01/06/07/13/15/16/18/19/21/22/23/25/26/28.
These are development-informed hypotheses, not independently discovered sensors.

A reuses the exact existing training and development matrices. Its original
calibration endpoints, features and labels are checked against the Run 7 source.
B-J rebuild features at exactly those endpoints from raw training trajectories;
labels, metadata and weights remain attached by UAV/cutoff identity. Canonical
names map equivalent features to Run 7's eight residual inputs. The residual
head receives the selected representation too, including its baseline definition.
No hidden extra sensor columns are supplied to the reduced sets.

## Model selection stays inside training UAVs

Run 7 keeps its selected tree parameters, cap, member seeds, four internal
grouped calibration folds, blend grid, residual head and nonnegative output
policy. Only the feature matrix and matching calibration source vary.

The simple recipe fixes the reference XGBoost/CatBoost structural parameters.
For each of four internal UAV folds:

1. Hold one group out for OOF calibration predictions.
2. From the remaining training UAVs, deterministically reserve 20% for early
   stopping. Fit on the others using capped targets and equal per-UAV weights.
3. Refit each model on all UAVs in that internal training partition using the
   selected number of iterations, then predict the held OOF calibration endpoints.

Select the XGBoost blend weight from 0 to 1 in increments of 0.05 using these
raw-label OOF predictions with equal total calibration weight per UAV. Choose
final iteration counts by the rounded median of the internal counts, refit both
models on **all outer-training UAVs**, and evaluate once on outer-held UAVs.
The outer-held labels never select early stopping, iterations or blend weights.

This deliberately changes the reference script's evaluation, all-cycle sampling,
unequal UAV weighting, and final 90%-UAV fit to the matched experimental contract.
It tests representations and recipes, not an exact rerun of its Kaggle submission.

## Stages and budget

The screen uses five grouped folds with seed 20270107. All twenty configurations
are evaluated. A challenger must improve mean-fold RMSE by at least 2% and win
at least 4/5 folds. Among passing challengers, choose the lowest mean RMSE;
ties use fewer features, then the lexical configuration name. Freeze that one
configuration before confirmation. Screening bootstrap intervals and all matched
contrasts are exploratory, not independent confirmation or multiple-testing-adjusted
significance claims.

Confirmation compares only that challenger with unchanged Run 7 on seeds
20270117 and 20270127 (ten folds). Promotion requires all of:

- At least 2% mean-fold RMSE improvement.
- At least 8/10 fold wins.
- Pooled R2 at least 0.90.
- A paired UAV-bootstrap 95% RMSE-change interval wholly below zero.

If no screen candidate passes, confirmation is skipped. A failed confirmation
does not trigger another challenger. Confirmation summaries exclude the screen.
The seeds reuse previously examined UAVs, so this is a robustness check, not an
untouched test set or a promise of public R2 above 0.9. Cheaper near-ties remain
visible in the cost table; they are not automatically promoted as accuracy gains.

The screen costs **100 complete recipe evaluations**; confirmation adds at most
**20**. These are not individual tree fits: each Run 7 evaluation includes 30
base estimator fits plus its residual head, and each simple evaluation includes
18 estimator fits. The screen therefore entails 2,400 base estimator fits plus
50 residual-head fits. Full training is a substantial workload.

Configuration/fold fits run in **two spawned worker processes** by default,
controlled by `max_workers = 2` in the workflow settings. Each worker owns one
cell checkpoint. The main process alone selects the screen winner and writes
stage reports after the required fits complete; completion order does not change
the report order or selection rule. At most two fits are submitted at once.

XGBoost uses CUDA when available; both workers share the GPU. ExtraTrees retains
its single worker. Each simple CatBoost fit uses four CPU threads, and numerical
thread pools are bounded at four per process. Two workers are an initial setting
for the local eight-logical-CPU / 8 GB RTX 3070 machine, not a measured speedup
guarantee. Set `max_workers = 1` for serial execution before starting a new run.
Settings are registered, so changing workers after training starts requires a new
run directory under the current resume contract. On a worker failure, no further
cells are dispatched; other active cells finish and checkpoint before the error
is returned. PE_28 does not change execution settings in existing scientific runs.

## Commands and resume

From the repository root:

```powershell
# Verify features, endpoint labels, source calibration and partitions; no fitting.
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_28\run.py --only validate_inputs

# Full gated comparison. Run the same command to resume after interruption.
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_28\run.py
```

`--list` shows the commands; `--status` shows completed steps. Each completed
configuration/fold evaluation is saved as one atomic checksummed JSON under
`runs/run_1/cells/`. An interruption within a recipe repeats that unfinished
recipe, while completed cells are reused after validating endpoints, labels,
configuration, feature names and training/calibration UAV membership.

`reporting/pre_registration.json` freezes settings, source data, source/model code,
the reference script, Python, and relevant package versions. Changing them
requires a new `[pipeline].run` directory. `--force` clears completed cells only
under the same registered contract; it does not bypass source-hash checks.

## Outputs

- `reporting/input_verification.json`: feature counts and workload readiness.
- `reporting/feature_catalog.json`: exact ordered input names for each set.
- `reporting/outer_folds.csv` and `inner_folds.csv`: generated grouped partitions.
  The inner table records partition construction; model calibration uses its own
  GroupKFold within each outer training partition, matching Run 7's recipe.
- `cells/*.json`: held predictions, membership, runtime, and simple-recipe
  stopping/OOF membership, iteration counts and selected weight.
- `stages/screen/reporting/` and `stages/confirmation/reporting/`: fold/pooled
  metrics, error-region and overprediction metrics, UAV-bootstrap decisions,
  readable comparison plot, `summary_with_costs.csv`, and
  `matched_contrasts.csv` for the 32 predeclared comparisons available in screening.
- `reporting/selected_configuration.json`: frozen screen choice.
- `reporting/winner_manifest.json`: final stage outcome and retained production model.

Run 7 remains deployed. This runner does not train a production replacement,
open locked evaluation, create a submission, or contact Kaggle.
