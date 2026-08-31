# Step 5: Inner model selection

## Purpose

This step automatically selects one configuration inside each enabled model
family and outer fold. It uses the four inner UAV folds and the five development
scenarios defined in Phase 1. It never loads locked validation or test data.

The output is not an architecture ranking. For example, the selected XGBoost
configuration is the best XGBoost candidate for that outer fold, but Step 5
does not compare it with the selected Random Forest, MLP, TCN, or LSTM.

## Selection unit

One independent study is identified by:

    model family + outer fold + settings version

For that study, the runner:

1. keeps the current outer-validation UAVs outside model selection;
2. generates distinct candidates from the recorded settings search space;
3. treats the tabular feature set or sequence lookback as part of a candidate;
4. evaluates every candidate on all four inner UAV folds;
5. refits preprocessing and the model separately in every inner fold;
6. calculates RMSE on the five development scenarios;
7. averages the four inner-fold RMSE values;
8. selects the lowest mean RMSE inside that family and outer fold.

Mean and cycle-only baselines have no tunable choices and therefore receive one
candidate. Other enabled families receive the settings budget of 20 distinct
candidates.

## Automatic tuning

"candidate_space.py" maps the Step 1 search definitions to Optuna suggestions.
It supports fixed, categorical, continuous uniform, log-uniform, and
integer-sequence choices. The feature-set and lookback alternatives are sampled
through the same candidate object as model hyperparameters.

"InnerModelSelectionRunner" creates an in-memory Optuna "TPESampler" study with
search seed 13 for each family/outer-fold pair. Studies run with one worker so
candidate order and model seeds remain repeatable. Duplicate resolved
configurations are rejected and do not consume the distinct-candidate budget.

The Ridge/Elastic Net family uses a conditional space. Optuna samples only the
parameters active for the selected variant; the unused keys receive fixed valid
defaults because the Step 4 adapter requires one complete resolved dictionary.

## Leakage boundary

The runner can request only these public methods:

- Step 2 "get_inner_selection_split" for tabular candidates;
- Step 3 "get_inner_selection_split" for sequence candidates;
- Step 3b "get_inner_selection_split" for trajectory candidates.

There is no call to either locked-outer split method. Every returned split is
checked for disjoint training and validation UAVs, an available RUL target,
training sample weights, and exactly five development scenarios.

Preprocessing remains fold-specific. Step 4 robust scalers are fitted from the
current inner-training rows, and Step 3 sequence scalers are fitted from the
current inner-training UAV histories.

## Training duration

XGBoost, CatBoost, and neural candidates use their inner-validation data for
early stopping. For the selected configuration, Step 5 records the median best tree
or epoch count across the four inner folds. If the median lies halfway between
two integers, it is rounded upward.

Step 6 will use this value as a fixed retraining duration. Consequently, locked
outer-validation targets cannot choose the tree or epoch count.

## Files

- "candidate_space.py" resolves representation and hyperparameter candidates.
- "inner_model_selection.py" loads inner splits, runs studies, calculates the
  objective, and writes checkpoints and consolidated artifacts.
- "run_inner_model_selection.py" provides the command-line interface.

## Generated artifacts

Everything below is written to "runs/run_<n>/5_inner_model_selection/", where
"n" is the "run_number" in the architecture study settings. That number is only
ever changed by hand, so interrupting a study and resuming it later continues
inside the same folder. "--output-dir" still accepts an explicit path.

The "studies/" directory contains separate files for every completed
or attempted family/outer-fold study:

- "...__candidates.csv" contains one row per complete candidate;
- "...__inner_folds.csv" contains its four fold-level results;
- "...__selected.json" contains the within-family selected configuration;
- "...__status.json" says whether the individual study is running,
  interrupted, failed, or complete. Interrupted studies retain their candidate
  checkpoints but remain explicitly ineligible for locked evaluation.

The consolidated Step 5 files are:

- "candidate_results.csv" with every completed candidate;
- "inner_fold_results.csv" with every candidate/fold score and training fact;
- "selected_configurations.csv" with one selected configuration per completed
  family/outer-fold study;
- "selection_manifest.json" with completion counts, protocol settings, and an
  explicit statement that architecture selection is not automatic.

The manifest remains "partial" until all fifteen enabled families have completed
all five outer folds. Later steps must require a "complete" manifest.

## TensorBoard monitoring

"runs/run_<n>/tensorboard_logs/step_5/<family>/outer_fold_N/study_progress"
holds the search curve: "search/candidate_rmse" carries one point per completed
candidate, with that candidate's hyperparameters attached as text. This is the
view to watch during a run -- whether tuning is still improving, and whether it
has plateaued.

Per-fit "train/loss" and "val/rmse" curves are switched off by default, because
a study fits (candidate budget x inner fold count) models and one prefix per fit
makes the scalar panel unreadable. Set "PHASE2_TENSORBOARD_FIT_CURVES=1" to
publish them below ".../outer_fold_N/fit_progress" while debugging a single
architecture. Every finished quantity -- development RMSE, MAE, R2, bias, the
age-band breakdown, timings, and the selected candidate -- lives in this step's
CSV artifacts, not in TensorBoard.

These are development metrics from the five permitted scenarios. No locked or
test target is accessed by Step 5 monitoring.

Generated artifacts remain visible locally and are ignored by Git.

## Running Step 5

Run the entire inner-selection stage from the repository root:

    py 2_architecture_experiments\2_model_architecture_study\5_inner_model_selection\run_inner_model_selection.py

Because the complete study includes many neural fits, individual studies can
be run separately while writing to the same artifact directory:

    py 2_architecture_experiments\2_model_architecture_study\5_inner_model_selection\run_inner_model_selection.py --family xgboost --outer-fold 1

Repeat "--family" or "--outer-fold" to request several values. A rerun of the
same family/fold study replaces that study's generated checkpoints. Other
completed studies are preserved and reconsolidated.

The candidate budget cannot be changed from the command line. This prevents a
quick diagnostic run from being confused with the settings-defined study.
