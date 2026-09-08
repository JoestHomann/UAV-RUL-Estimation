# PE_27: UAV-subset ensemble

PE_27 tests whether Run 7 improves when averaged across different training-UAV
subsets. It uses **subsampling without replacement**, rather than resampling
individual prefixes. Each member receives every existing training prefix of
its selected UAVs, with the original total weight of one per UAV. The features,
cap-125 fitting targets, raw evaluation labels, tree settings, three internal
member seeds per tree family, and residual-correction recipe remain fixed.

In each five-fold evaluation, the control uses all 80 training UAVs. A subset
member uses 64 of those 80. All members exclude the 20 outer-held UAVs. Its
calibration endpoints are restricted to its own 64 UAVs, and the unchanged
residual adapter performs its four internal grouped folds entirely there.
Every complete model has six tree members; "four" and "eight" below count
complete Run 7 systems, not individual trees.

Subsets depend only on sorted UAV IDs and fixed seeds. No labels or feature
importance determine membership. All eight subsets within a fold are distinct;
the first four are exactly retained during expansion. Predictions are uniformly
averaged, without selecting weights or adding the full-data control to the mean.

## Stages and fixed decision rules

| Stage | Evaluation | Fits added | Requirement to continue |
| --- | --- | ---: | --- |
| `screen_4` | Five folds, seed 20261207; control versus four-member mean | 25 | At least 2% mean-fold RMSE improvement and 4/5 wins |
| `screen_8` | Same endpoints; reuse the first four members and control | 20 | At least 2% mean-fold RMSE improvement and 4/5 wins |
| `confirmation_8` | Ten folds, seeds 20261217 and 20261227 | 90 | At least 2% mean-fold RMSE improvement, 8/10 wins, pooled R2 at least 0.90, and a UAV-bootstrap 95% RMSE-change interval wholly below zero |

Failure at either screen ends the run normally and skips remaining stages.
Only the eight-member confirmation can promote a candidate for subsequent
final-model work. Confirmation metrics exclude the screening seed. There is no
post-hoc fallback to a four-member winner. The screening intervals are reported
but do not gate expansion; they are not treated as confirmation evidence.
These are new splits of previously examined UAVs, not a new independent dataset.
The earlier 4.09% saved-prediction averaging gain is motivation, not an expected
gain: these subset members train on fewer UAVs than the paired control.

## Run and resume

From the repository root:

```powershell
# Check source data, target/weight policies and all planned subsets without fitting.
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_27\run.py --only validate_inputs

# Run the gated workflow; the same command resumes after interruption.
& .\.venv\Scripts\python.exe .\2_architecture_experiments\1_pipeline_experiments\experiments\PE_27\run.py
```

`--list` prints commands and `--status` shows completion. The first screen costs
25 complete model fits; the maximum is 135. Each completed fit is atomically
checkpointed. Resume verifies the exact held endpoints, raw labels, finite
nonnegative predictions, and training/calibration UAV hashes before skipping it.
It does not reuse older PE_21/24/26 predictions as prospective evidence.

`pre_registration.json` freezes the settings, source files, model/runner code,
Python and package versions. Changing them requires a new `[pipeline].run`;
`--force` restarts fitting only under the same registered contract.

## Outputs

The root `runs/run_1/reporting/` contains input coverage, grouped assignments,
`member_uavs.csv` (the complete subset plan), `fold_predictions.csv` (completed
fits and membership provenance), and the final `winner_manifest.json` specifying
completed/skipped stages. Each `runs/run_1/stages/<stage>/reporting/` contains
mean-fold/pooled metrics, paired decisions, UAV-bootstrap intervals, regional and
overprediction metrics, predictions and a comparison figure.

Run 7 remains the production model. This experiment does not open locked
evaluation, load test inputs/labels, train a production replacement or submit to
Kaggle. Implementation and preflight can complete without running the scientific
training workload.
