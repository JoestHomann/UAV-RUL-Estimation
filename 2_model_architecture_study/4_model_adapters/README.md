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
- "XGBoostAdapter" fits sample-weighted boosted trees to unscaled values and
  supports inner-fold early stopping or a fixed outer-retraining tree count.
- "MLPAdapter" applies training-fold robust scaling before a PyTorch MLP.
- "TCNAdapter" applies causal dilated convolutions to Step 3 sequence windows.
- "LSTMAdapter" uses packed valid sequence lengths in one temporal direction.
- "TransformerAdapter" implements the settings' disabled conditional masked
  Transformer so it can be enabled without changing the adapter layer.
- "RBFSVRAdapter" implements the disabled optional robust-scaled RBF-SVR.

The eight required families are enabled by the current settings. Transformer
and RBF-SVR are implemented but remain disabled and cannot be constructed by
the factory unless the caller explicitly allows disabled families.

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
callback interface. Atomic-fit estimators expose no optimization iterations and
therefore publish no curve; everything about their fits is recorded in the Step
5 and Step 6 artifacts, which is where every family's finished numbers live.

Every adapter also declares whether its family is stochastic. Random Forest,
XGBoost, and the neural adapters use the three retraining seeds from the
settings; deterministic baselines and linear or kernel models need one run.

## Preprocessing ownership

Preprocessing stays with the model artifact that depends on it:

- Ridge, Elastic Net, MLP, and RBF-SVR fit robust tabular scaling from the
  active training rows only.
- Random Forest and XGBoost use unchanged numeric feature values.
- TCN, LSTM, and Transformer require the fold-scaled telemetry returned by
  Step 3. They additionally robust-scale the two age side features using only
  the training rows supplied to "fit".

Keeping fitted scalers inside their model adapters ensures that later
predictions use exactly the transformation learned during training.

## Training policy

All families use the supplied Phase 1 sample weights. Neural adapters share
weighted mean squared error, AdamW, deterministic seeds, gradient clipping,
batch size, maximum epoch count, and early-stopping patience from the settings.
They run on CPU with one data-loading worker for a simple repeatable baseline.

XGBoost and neural models support two distinct training modes:

- inner-fold candidate fitting uses validation-based early stopping;
- outer-fold retraining receives a fixed tree or epoch count from the later
  runner, which will use the median best duration found in the inner folds.

Random Forest and XGBoost each use one internal worker. Candidate-level
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
| Tabular | "models/tabular/xgboost.py" | "XGBoostAdapter" |
| Tabular | "models/tabular/rbf_svr.py" | "RBFSVRAdapter" |
| Neural | "models/neural/mlp.py" | "MLPAdapter" |
| Neural | "models/neural/tcn.py" | "TCNAdapter" |
| Neural | "models/neural/lstm.py" | "LSTMAdapter" |
| Neural | "models/neural/transformer.py" | "TransformerAdapter" |

The neural modules reuse only two support files:

- "models/neural/neural_base.py" owns the common weighted PyTorch training,
  early-stopping, deterministic-seeding, and inference loop.
- "models/neural/sequence_base.py" owns validation and preparation shared by
  TCN, LSTM, and Transformer sequence inputs.

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

The command creates "artifacts/model_registry.json". It records all ten model
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
