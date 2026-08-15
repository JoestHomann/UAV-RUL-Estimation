# Step 1: Experiment contract

**Reads:** `experiment_contract.toml` and the Phase 1 artifacts listed in that contract.

**Writes:** `artifacts/experiment_specification.json`

This step records the architecture-study settings before model implementation begins. The TOML file is the tracked, human-readable source of truth. The generated JSON contains the same resolved settings plus a concise summary of the Phase 1 structural checks.

The scripts have separate responsibilities:

- `verify_experiment_contract.py` validates the strict TOML schema and checks that the required Phase 1 artifacts exist and have the expected structure. It never writes files.
- `build_experiment_contract.py` calls the verifier and writes the deterministic JSON artifact only after validation succeeds.

Run both commands from the repository root:

```powershell
py 2_model_architecture_study\1_experiment_contract\verify_experiment_contract.py
py 2_model_architecture_study\1_experiment_contract\build_experiment_contract.py
```

Scientific settings cannot be overridden from the command line. The optional `--contract` and `--output-dir` arguments only change where the source is read and where the generated artifact is written.

The contract is intentionally not frozen by code. When its experiment settings change, increment `contract_version` manually and regenerate all dependent Phase 2 artifacts.
