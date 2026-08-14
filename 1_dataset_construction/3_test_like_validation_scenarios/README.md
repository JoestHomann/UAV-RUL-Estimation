# Step 3: Test-like validation scenarios

**Reads:** raw train/test data, Step 1 `test_fligh_cycles_cut_offs.csv`, Step 2 outer folds

**Writes:** `artifacts/locked_validation_scenarios.csv`, `artifacts/development_validation_scenarios.csv`, `artifacts/test_endpoints.csv`, `artifacts/scenario_config.json`

Each scenario assigns one test-like cutoff to every training UAV.
