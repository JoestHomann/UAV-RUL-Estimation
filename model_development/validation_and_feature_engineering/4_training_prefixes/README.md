# Step 4: Training prefixes

**Reads:** raw training data and Step 1 `train_flight_cycles.csv` and `test_fligh_cycles_cut_offs.csv`

**Writes:** `artifacts/training_prefixes.csv`, `artifacts/training_prefix_config.json`

The output contains 20 cutoffs per UAV and equal total sample weight per UAV.
