# Step 1: Architecture study settings

**Reads:** `architecture_study_settings.toml` and the Phase 1 artifacts listed in that settings file.

**Writes:** `artifacts/experiment_specification.json`

This step records the architecture-study settings before model implementation begins. The TOML file is the tracked, human-readable source of truth. The generated JSON contains the same resolved settings plus a concise summary of the Phase 1 structural checks.

The scripts have separate responsibilities:

- `verify_architecture_study_settings.py` validates the strict TOML schema and checks that the required Phase 1 artifacts exist and have the expected structure. It never writes files.
- `build_architecture_study_settings.py` calls the verifier and writes the deterministic JSON artifact only after validation succeeds.

Run both commands from the repository root:

```powershell
py 2_model_architecture_study\1_architecture_study_settings\verify_architecture_study_settings.py
py 2_model_architecture_study\1_architecture_study_settings\build_architecture_study_settings.py
```

Scientific settings cannot be overridden from the command line. The optional `--settings` and `--output-dir` arguments only change where the source is read and where the generated artifact is written.

This settings file is intentionally not frozen by code. When its values change, increment `settings_version` manually and regenerate all dependent Phase 2 artifacts.
