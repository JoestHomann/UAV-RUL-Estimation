# Step 5: Prefix feature engineering

**Reads:** raw train/test data, Step 3 scenarios/endpoints, Step 4 training prefixes

**Writes:** `artifacts/training_features.csv.gz`, `artifacts/development_validation_features.csv.gz`, `artifacts/locked_validation_features.csv.gz`, `artifacts/test_features.csv.gz`

Every row contains only features computable at its recorded cutoff.
