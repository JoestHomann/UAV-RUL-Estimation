# Step 4: Model adapters

## Purpose

This step implements every architecture declared in the architecture study
settings behind one shared interface. The later tuning and evaluation runner can
therefore exchange model families without containing separate fitting and
prediction logic for each library.

Step 4 does not tune hyperparameters, run the outer-fold study, compare model
scores, or select an architecture. It receives one already resolved candidate,
fits it, produces RUL predictions, and preserves the fitted model when asked.

## Implemented families

- "MeanBaselineAdapter" predicts the sample-weighted training-fold mean RUL.
- "CycleOnlyBaselineAdapter" reproduces the Phase 1 weighted linear baseline
  using only "feature__flight_cycle".
- "RegularizedLinearAdapter" implements Ridge and Elastic Net after robust
  scaling fitted only on the supplied training rows.
- "RandomForestAdapter" fits sample-weighted trees to unscaled tabular values.
- "ExtraTreesAdapter" fits sample-weighted trees with randomized split
  thresholds to the same unscaled tabular values.
- "XGBoostAdapter" fits sample-weighted boosted trees to unscaled values,
  automatically uses CUDA only when both the installed XGBoost build and the
  active machine support it, and supports inner-fold early stopping or a fixed
  outer-retraining tree count.
- "CatBoostAdapter" fits sample-weighted CPU CatBoost trees over the same
  numerical inputs, with inner-fold early stopping and fixed outer-retraining
  tree counts. CPU execution is intentional because CatBoost GPU training is
  not deterministic enough for this study.
- "TrajectoryDTWKNNAdapter" retrieves similar training-UAV trajectories with
  constrained multivariate DTW and returns a distance-weighted average of their
  cycle-wise remaining lives. It is deterministic and uses no fixed-window
  feature set or lookback.
- "MLPAdapter" applies training-fold robust scaling before a PyTorch MLP.
- "TCNAdapter" applies causal dilated convolutions to Step 3 sequence windows.
- "MultiScaleCNNAdapter" combines three causal convolution branches with
  different receptive fields and masked mean/maximum pooling.
- "SensorGraphTCNAdapter" fits a top-k absolute-correlation graph from the
  active training fold, performs sensor message passing, and then applies a
  causal residual TCN.
- "LSTMAdapter" uses packed valid sequence lengths in one temporal direction.
- "TransformerAdapter" implements the conditional masked Transformer.
- "RBFSVRAdapter" implements the optional robust-scaled RBF-SVR.

All fifteen declared families are enabled for Run 4. Their included,
conditional, and optional status labels preserve the study rationale; the
separate `study.enabled` table controls what actually runs.

## Shared interface

Every adapter exposes the following operations:

- "fit(training_data, validation_data)" fits preprocessing and the model;
- "predict(data)" returns one finite RUL value per row and clips values at the
  settings' lower prediction boundary;
- "save(path)" stores the fitted preprocessing, estimator, configuration, and
  training summary together;
- "load(path)" restores a trusted local model artifact.

The Step 2 and Step 3 dataset objects already carry RUL targets and sample
weights. The model seed and resolved hyperparameters are supplied when the
factory creates an adapter, so those values do not have to be passed again to
"fit".

"TrainingSummary" records training and validation row counts, elapsed fitting
time, completed epochs or iterations, the selected stopping point when
available, validation RMSE, and the trainable parameter count when meaningful.

The factory can also attach the neutral training monitor. Adapter modules
never import TensorBoard. The shared neural loop reports "train/loss" and,
where a development fold exists, "val/rmse" each epoch; the XGBoost adapter
reports the same two tags on sampled boosting rounds through its official
callback interface, and CatBoost publishes its structured training history
through the same monitor boundary after fitting. Atomic-fit estimators expose
no optimization iterations and therefore publish no curve; everything about
their fits is recorded in the Step 5 and Step 6 artifacts, which is where every
family's finished numbers live.

Every adapter also declares whether its family is stochastic. Random Forest,
Extra Trees, XGBoost, CatBoost, and the neural adapters use the three
retraining seeds from the settings; deterministic baselines and linear or
kernel models need one run.

## Preprocessing ownership

Preprocessing stays with the model artifact that depends on it:

- Ridge, Elastic Net, MLP, and RBF-SVR fit robust tabular scaling from the
  active training rows only.
- Random Forest and XGBoost use unchanged numeric feature values.
- Extra Trees and CatBoost use unchanged numeric feature values. CatBoost does
  not receive categorical columns because this dataset's engineered inputs are
  numerical.
- Trajectory DTW-kNN receives the fold-scaled variable-length trajectories and
  the complete reference library built from active training UAVs only. It does
  not use tabular feature sets or fixed lookbacks.
- TCN, multi-scale CNN, sensor-graph TCN, LSTM, and Transformer require the
  fold-scaled telemetry returned by Step 3. They additionally robust-scale the
  two age side features using only the training rows supplied to "fit". The
  sensor graph is also fitted from training-fold telemetry only.

Keeping fitted scalers inside their model adapters ensures that later
predictions use exactly the transformation learned during training.

## Training policy

All families use the supplied Phase 1 sample weights. Neural adapters share
weighted mean squared error, AdamW, deterministic seeds, gradient clipping,
batch size, maximum epoch count, and early-stopping patience from the settings.
They run on CPU with one data-loading worker for a simple repeatable baseline.

XGBoost, CatBoost, and neural models support two distinct training modes:

- inner-fold candidate fitting uses validation-based early stopping;
- outer-fold retraining receives a fixed tree or epoch count from the later
  runner, which will use the median best duration found in the inner folds.

Random Forest, Extra Trees, XGBoost, and CatBoost each use one internal worker
or CPU thread. Candidate-level
parallelism belongs to the later runner, avoiding nested parallel execution.

## Files

- "base.py" defines the common adapter, training summary, persistence, input
  validation, sample-weight, and metric helpers.
- "model_registry.py" maps settings family names to adapter classes and creates
  configured adapters.
- "build_model_registry.py" writes the traceable registry artifact.

Every architecture has one implementation module below "models":

| Category | Model module | Adapter |
| --- | --- | --- |
| Baseline | "models/baselines/mean_baseline.py" | "MeanBaselineAdapter" |
| Baseline | "models/baselines/cycle_only_baseline.py" | "CycleOnlyBaselineAdapter" |
| Tabular family | "models/tabular/regularized_linear.py" | "RegularizedLinearAdapter" |
| Tabular variant | "models/tabular/ridge.py" | Ridge construction |
| Tabular variant | "models/tabular/elastic_net.py" | Elastic Net construction |
| Tabular | "models/tabular/random_forest.py" | "RandomForestAdapter" |
| Tabular | "models/tabular/extra_trees.py" | "ExtraTreesAdapter" |
| Tabular | "models/tabular/xgboost.py" | "XGBoostAdapter" |
| Tabular | "models/tabular/catboost.py" | "CatBoostAdapter" |
| Tabular | "models/tabular/rbf_svr.py" | "RBFSVRAdapter" |
| Trajectory | "models/trajectory/trajectory_dtw_knn.py" | "TrajectoryDTWKNNAdapter" |
| Neural | "models/neural/mlp.py" | "MLPAdapter" |
| Neural | "models/neural/tcn.py" | "TCNAdapter" |
| Neural | "models/neural/multiscale_cnn.py" | "MultiScaleCNNAdapter" |
| Neural | "models/neural/sensor_graph_tcn.py" | "SensorGraphTCNAdapter" |
| Neural | "models/neural/lstm.py" | "LSTMAdapter" |
| Neural | "models/neural/transformer.py" | "TransformerAdapter" |

The neural modules reuse only two support files:

- "models/neural/neural_base.py" owns the common weighted PyTorch training,
  early-stopping, deterministic-seeding, and inference loop.
- "models/neural/sequence_base.py" owns validation and preparation shared by
  all five sequence-family inputs.

This structure keeps architecture-specific layers and hyperparameter handling
inside the model's own file while preventing copied training logic from
diverging between neural families.

Ridge and Elastic Net remain one comparison family because the architecture
study settings select between them within the same regularized-linear search.
Their estimator definitions nevertheless live in separate files; the family adapter
contains only their shared robust scaling, fitting, prediction, and reporting.

## Generated artifact

Run from the repository root:

    py 2_model_architecture_study\4_model_adapters\build_model_registry.py

The command creates "artifacts/model_registry.json". It records all fifteen model
families, their enabled status, representation, adapter class, permitted
configuration fields, common behavior, settings version, and installed model
library versions. It contains no timestamp or hash and makes no performance or
winner claim.

Generated artifacts remain visible locally and are ignored by Git.

## Boundary for Step 5

Step 5 will sample resolved candidates from the settings, request the correct
Step 2 or Step 3 split, construct the matching adapter through
"ModelAdapterFactory", and collect its predictions and training summary. The
model adapters deliberately contain no candidate-ranking or cross-architecture
selection logic.
