# Step 6: Locked outer evaluation

## Purpose

This step measures how each fully tuned architecture generalizes to unseen
UAVs. It retrains the configuration selected for each family and outer fold on
all 80 outer-training UAVs, then predicts the 20 held-out UAVs across the 20
locked validation scenarios.

Step 6 produces predictions and efficiency facts. It does not tune a model,
change a feature set or lookback, rank architectures, or choose a winner.

## Mandatory Step 5 gate

"evaluation_gate.py" runs before either data adapter is constructed. It requires:

- Step 5 "selection_manifest.json" to have status "complete";
- all 40 enabled family/outer-fold studies to be complete;
- exactly one selected configuration for each enabled family and outer fold;
- matching Step 1 and Step 5 contract versions;
- valid feature sets, lookbacks, hyperparameters, and retraining durations;
- Step 5 to report no locked-data or test-data access;
- architecture selection to remain manual.

If any requirement fails, no locked split is loaded. There is no command-line
option to bypass this gate.

## Retraining procedure

For one selected family, outer fold, and seed, the runner:

1. loads the selected Step 5 configuration;
2. obtains the matching locked-outer split from Step 2 or Step 3;
3. verifies 80 training UAVs, 1,600 training prefixes, 20 held-out UAVs, 20
   locked scenarios, 400 validation endpoints, disjoint UAV groups, and equal
   total training weight per UAV;
4. reconstructs the Step 4 adapter with the selected hyperparameters;
5. fits using only the outer-training dataset;
6. predicts the locked validation dataset after fitting;
7. saves the model, prediction rows, training facts, and inference facts.

Locked validation data is passed to "predict" only. The call to "fit" receives
"None" as its validation argument, so locked targets cannot affect model
training or early stopping.

## Fixed training duration

XGBoost and neural models receive "outer_retraining_iterations" from Step 5.
That value is the rounded median of the four inner-fold best durations. Step 6
checks that the completed tree or epoch count equals this fixed value.

No early stopping is performed against locked targets.

## Retraining seeds

Random Forest, XGBoost, MLP, TCN, LSTM, and any enabled Transformer are marked
as stochastic by their Step 4 adapters. They are retrained with seeds 13, 37,
and 73. The best seed is never selected.

Mean and cycle-only baselines, regularized linear models, and optional RBF-SVR
are deterministic and run once with seed 13.

With the current eight enabled families, the complete stage contains 90
family/fold/seed runs and 36,000 locked prediction rows.

## Files

- "evaluation_gate.py" validates every prerequisite without importing a data
  adapter or loading locked data.
- "locked_outer_evaluation.py" performs retraining, prediction, validation,
  model persistence, checkpointing, and artifact consolidation.
- "run_locked_outer_evaluation.py" provides the command-line interface.

## Generated artifacts

Each family/fold/seed run creates:

- "models/...joblib" containing the fitted model and its preprocessing;
- "runs/...__predictions.csv.gz" containing 400 locked predictions;
- "runs/...__run.json" containing training and efficiency facts;
- "runs/...__status.json" marking the run as running, failed, or complete.

Consolidated artifacts are:

- "locked_predictions.csv.gz" with all complete prediction rows;
- "model_runs.csv" with training time, inference time, parameter count, model
  size, selected configuration, and fixed duration for every complete run;
- "locked_evaluation_manifest.json" with completion counts and protocol facts.

The manifest remains "partial" until all 90 required runs are complete. Step 7
must reject a partial manifest.

Generated artifacts remain visible locally and are ignored by Git.

## Running Step 6

After Step 5 is complete, run all locked evaluations from the repository root:

    py 2_model_architecture_study\6_locked_outer_evaluation\run_locked_outer_evaluation.py

Individual completed-plan studies can be run separately:

    py 2_model_architecture_study\6_locked_outer_evaluation\run_locked_outer_evaluation.py --family xgboost --outer-fold 0

Filters reduce only the runs executed by that command. They do not weaken the
gate: all Step 5 studies must already be complete before any filtered locked
evaluation can start.

## Current state

Step 5 is currently partial at 10 of 40 studies. Therefore, the Step 6 gate is
closed and no locked prediction or model artifact has been generated.
