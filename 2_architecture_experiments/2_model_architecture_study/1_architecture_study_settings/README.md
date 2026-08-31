# Step 1: Architecture study settings

**Reads:** `architecture_study_settings.toml` and the Phase 1 artifacts listed in that settings file.

**Writes:** `artifacts/experiment_specification.json`

This step records the architecture-study settings before model implementation begins. The TOML file is the tracked, human-readable source of truth. The generated JSON contains the same resolved settings plus a concise summary of the Phase 1 structural checks.

The scripts have separate responsibilities:

- `verify_architecture_study_settings.py` validates the strict TOML schema and checks that the required Phase 1 artifacts exist and have the expected structure. It never writes files.
- `build_architecture_study_settings.py` calls the verifier and writes the deterministic JSON artifact only after validation succeeds.

Run both commands from the repository root:

```powershell
py 2_architecture_experiments\2_model_architecture_study\1_architecture_study_settings\verify_architecture_study_settings.py
py 2_architecture_experiments\2_model_architecture_study\1_architecture_study_settings\build_architecture_study_settings.py
```

Scientific settings cannot be overridden from the command line. The optional `--settings` and `--output-dir` arguments only change where the source is read and where the generated artifact is written.

This settings file is intentionally not frozen by code. When its values change, increment `settings_version` manually and regenerate all dependent Phase 2 artifacts.

Tabular feature-set names are validated against the selected Phase 1 catalog
and are not restricted to the four legacy literals. The declared
`representations.tabular_feature_sets` and `phase_1.expected_feature_sets` keys
must match exactly, keeping versioned extended-feature catalogs explicit.

For a fixed-size prefix policy, declare
`expected_prefixes_per_training_uav`. For an eligibility-limited policy such as
`prefix40_stratified`, omit that field and declare both
`minimum_prefixes_per_training_uav` and
`maximum_prefixes_per_training_uav`. These values, the artifact paths, and the
feature counts can be transcribed directly from the selected variant's
`phase_2_interface.json`.
