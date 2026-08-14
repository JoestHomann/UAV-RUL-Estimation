# Step 9: Cycle-only baseline

**Reads:** Step 2 outer folds, Step 5 feature tables, Step 8 metric definitions

**Writes:** predictions, coefficients, and grouped metrics under `artifacts/`

The locked predictions are generated only by models fitted without the held-out outer-fold UAVs.
