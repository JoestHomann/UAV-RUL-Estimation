# Step 1: Structural data audit

**Reads:** `data/train.csv`, `data/test.csv`

**Writes:** `artifacts/dataset_audit.json`, `artifacts/train_flight_cycles.csv`, `artifacts/test_fligh_cycles_cut_offs.csv`

This step validates the raw schema and history invariants. It does not modify the source CSV files.
