# UAV Remaining Useful Life Estimation

This repository develops a remaining-useful-life model for UAV telemetry. The
workflow is divided into four phases that must be executed in order:

1. Phase 0 investigates telemetry quality and predictive evidence.
2. Phase 1 creates leakage-safe training and validation datasets.
3. Phase 2 tunes and compares model architectures on fixed UAV-grouped splits.
4. Phase 3 selects the winner, performs final tuning and training, and creates
   test-set predictions for submission.

Phase 2 produces comparison tables and figures but does not automatically rank
architectures or select a winner. Phase 3 begins with that manual research
decision before any final tuning or test inference.

## Repository structure

| Folder | Phase | Purpose |
| --- | --- | --- |
| [data](data/) | Input | Raw training and test CSV files |
| [0_data_analysis](0_data_analysis/) | Phase 0 | Broad and core telemetry analysis |
| [1_dataset_construction](1_dataset_construction/) | Phase 1 | Leakage-safe folds, prefixes, features, and validation artifacts |
| [2_model_architecture_study](2_model_architecture_study/) | Phase 2 | Model tuning, locked evaluation, and architecture comparison |
| [3_final_model_training_and_inference](3_final_model_training_and_inference/) | Phase 3 | Winner selection, final tuning and training, test inference, and submission verification |
| [literature_and_planning](literature_and_planning/) | Documentation | Research notes and phase documentation |

## Requirements

- Windows PowerShell
- A supported Python installation with the "py" launcher
- Enough disk space for generated datasets, fitted models, and prediction files
- Substantial execution time for Phase 2 model tuning
- Additional execution time for the Phase 3 final configuration search

All commands below assume that PowerShell is open in the repository root. In
PowerShell, a local executable must start with ".\". Therefore, use
".\.venv\Scripts\python.exe", not ".venv/Scripts/python.exe".

## Initial environment setup

Create the virtual environment once:

    py -m venv .venv

Install the shared Phase 2 dependencies and SciPy, which is additionally used
by Phase 0:

    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r 2_model_architecture_study\requirements.txt
    .\.venv\Scripts\python.exe -m pip install "scipy>=1.15,<2"

Check that the intended interpreter is available:

    .\.venv\Scripts\python.exe --version

Activation is optional because every command in this manual calls the virtual
environment interpreter directly.

## Input data

Before running a phase, confirm that these files exist:

- "data/train.csv" contains UAV identifiers, flight cycles, 28 telemetry
  channels, and RUL targets.
- "data/test.csv" contains UAV identifiers, flight cycles, and the 28 telemetry
  channels without RUL.

The phase scripts validate the detailed schema and stop if required columns,
finite values, unique keys, or history constraints are violated.

## Phase 0: Data analysis

Phase 0 examines data quality, temporal behavior, feature redundancy,
train/test drift, anomalies, and initial channel roles. It does not construct
model-validation datasets.

### Run the core analysis

Run the complete core suite:

    .\.venv\Scripts\python.exe 0_data_analysis\core_data_analysis\run_all.py

The suite runs temporal/RUL analysis, representative trajectories, redundancy
analysis, train/test drift, anomaly diagnostics, and channel classification.

Outputs are written below:

    0_data_analysis\core_data_analysis\figures\

Every quantitative plot has corresponding CSV evidence in its analysis
subfolder.

### Run the broad descriptive plots

Broad descriptive scripts can be executed individually. For example:

    .\.venv\Scripts\python.exe 0_data_analysis\broad_data_review\plot_descriptive_statistics.py

To run every broad script whose name starts with "plot_" in PowerShell:

    Get-ChildItem 0_data_analysis\broad_data_review\plot_*.py | ForEach-Object { & .\.venv\Scripts\python.exe $_.FullName }

Outputs are written below:

    0_data_analysis\broad_data_review\figures\

More details are available in the [Phase 0 README](0_data_analysis/README.md).

## Phase 1: Dataset construction

Phase 1 converts the reviewed data into fixed UAV-grouped folds, test-like
validation scenarios, causal training prefixes, engineered features,
fold-fitted preprocessing artifacts, baseline predictions, and leakage checks.

Run the complete ten-step workflow:

    .\.venv\Scripts\python.exe 1_dataset_construction\run_all.py

The steps execute in dependency order and stop on the first failure. Each step
writes its outputs into its own numbered folder under "artifacts" so the data
flow remains traceable.

The phase is complete only when the final report has status "passed":

    (Get-Content 1_dataset_construction\10_automated_leakage_checks\artifacts\verification_report.json -Raw | ConvertFrom-Json).status

Expected output:

    passed

Do not start Phase 2 if this verification fails. Investigate the failed
assertion and regenerate Phase 1 before continuing.

More details, including every generated artifact, are available in the
[Phase 1 README](1_dataset_construction/README.md).

## Phase 2: Model architecture study

Phase 2 uses one entry point for its seven steps. It builds and verifies the
architecture study settings, prepares the tabular and sequence adapters,
registers the models, performs within-family tuning, evaluates locked outer
folds, and creates the final architecture-comparison tables and figures.

### Check current progress

The status command is read-only and can be used at any time:

    .\.venv\Scripts\python.exe 2_model_architecture_study\run_phase_2.py --status

### Run the complete phase

Start TensorBoard in a second PowerShell terminal:

    .\.venv\Scripts\python.exe 2_model_architecture_study\tensorboard_monitoring\launch_tensorboard.py

Open the printed address, normally "http://localhost:6006". The dashboard can
be started before or during training and can be restarted independently.

For a new Phase 2 run, execute:

    .\.venv\Scripts\python.exe 2_model_architecture_study\run_phase_2.py

The entry point uses the same virtual-environment interpreter for every child
step and stops immediately if a prerequisite, TensorBoard dependency, event
write, or validation gate fails. TensorBoard logging is always active and does
not require a command-line option.

### Resume an interrupted run

To continue from model tuning without rebuilding Steps 1–4:

    .\.venv\Scripts\python.exe 2_model_architecture_study\run_phase_2.py --from-step 5

Completed Step 5 family/fold studies are preserved. Step 6 also preserves
complete family/fold evaluations; an interrupted stochastic family/fold is
rerun as one complete unit.

If Step 5 is already complete and only locked evaluation and comparison remain:

    .\.venv\Scripts\python.exe 2_model_architecture_study\run_phase_2.py --from-step 6

To run only the preparation stages and stop before expensive tuning:

    .\.venv\Scripts\python.exe 2_model_architecture_study\run_phase_2.py --through-step 4

### Force a complete expensive rerun

The following command deliberately replaces completed Step 5 studies and Step
6 evaluations:

    .\.venv\Scripts\python.exe 2_model_architecture_study\run_phase_2.py --from-step 5 --force

Use "--force" only when a complete rerun is genuinely intended.

### Phase 2 execution rules

- Do not run two Phase 2 training commands simultaneously. They would write to
  the same checkpoint folders.
- The read-only "--status" command may be used from another terminal.
- Step 6 remains locked until all Step 5 studies are complete.
- Step 7 remains locked until all Step 6 runs are complete.
- Step 6 TensorBoard runs show training progress and timing but no locked
  predictive metrics; Step 7 publishes the final locked comparison.
- Locked results are used for comparison, not for another tuning cycle.
- Step 7 saves all architecture results and plots without selecting a winner.

The expensive work is checkpointed below the corresponding Step 5 and Step 6
"artifacts" folders. Generated model and result files are intentionally ignored
by Git but remain visible locally.

More details are available in the
[Phase 2 README](2_model_architecture_study/README.md).

## Phase 3: Final model training and inference

Phase 3 records the manually selected Phase 2 winner, tunes only that family on
the five development folds, freezes the training contract, fits all 100
training UAVs, unlocks test inference and submission verification, and
automatically creates model-agnostic post-run figures.

The only human-edited Phase 3 configuration is the TOML below. Configure it
before starting a run and leave it unchanged when resuming:

    3_final_model_training_and_inference\1_winning_architecture_selection\phase_3_settings.toml

Validate the settings and referenced Phase 2 run:

    .\.venv\Scripts\python.exe 3_final_model_training_and_inference\1_winning_architecture_selection\verify_phase_3_settings.py

Run the training/development-only implementation checks:

    .\.venv\Scripts\python.exe 3_final_model_training_and_inference\verify_phase_3_implementation.py

Inspect progress or run the complete phase:

    .\.venv\Scripts\python.exe 3_final_model_training_and_inference\run_phase_3.py --status
    .\.venv\Scripts\python.exe 3_final_model_training_and_inference\run_phase_3.py

Resume an interrupted final search without discarding completed candidates:

    .\.venv\Scripts\python.exe 3_final_model_training_and_inference\run_phase_3.py --from-step 2

Regenerate only the post-run figures from a completed run:

    .\.venv\Scripts\python.exe 3_final_model_training_and_inference\run_phase_3.py --from-step 7

The test dataset is first opened by Step 5. Steps 1-4 use only the completed
Phase 2 artifacts plus training prefixes and development scenarios. More
details are available in the
[Phase 3 README](3_final_model_training_and_inference/README.md).

## Recommended full run order

For a clean execution, run these commands in order:

    .\.venv\Scripts\python.exe 0_data_analysis\core_data_analysis\run_all.py
    .\.venv\Scripts\python.exe 1_dataset_construction\run_all.py
    .\.venv\Scripts\python.exe 2_model_architecture_study\run_phase_2.py
    .\.venv\Scripts\python.exe 3_final_model_training_and_inference\run_phase_3.py

Broad Phase 0 descriptive plots are optional additions to the core analysis and
may be generated before Phase 1.

## Current project boundary

The repository now implements the full workflow through verified Kaggle
submission generation. The detailed
[Phase 3 protocol](literature_and_planning/development_documentation/phase_3_finalModelTrainingAndInference.md)
defines the final development-only search, frozen all-UAV training contract,
first test access, and submission checks.
