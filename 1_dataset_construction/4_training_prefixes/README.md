# Step 4: Training prefixes

**Reads:** raw training data and Step 1 `train_flight_cycles.csv` and `test_fligh_cycles_cut_offs.csv`

**Writes:** `artifacts/training_prefixes.csv`, `artifacts/training_prefix_config.json`

The legacy output contains 20 cutoffs per UAV and equal total sample weight per
UAV. The Run 5 `prefix40_stratified` policy requests up to 40 distinct eligible
cutoffs, allocates them across empirical test cutoff-age bands, and uses
`1 / actual_prefix_count` weights. One short-lived UAV supports 39 distinct
eligible cutoffs; every UAV still has total weight one.
