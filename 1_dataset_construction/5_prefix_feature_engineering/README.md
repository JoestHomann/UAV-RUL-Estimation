# Step 5: Prefix feature engineering

**Reads:** raw train/test data, Step 3 scenarios/endpoints, Step 4 training prefixes

**Writes:** `artifacts/training_features.csv.gz`, `artifacts/development_validation_features.csv.gz`, `artifacts/locked_validation_features.csv.gz`, `artifacts/test_features.csv.gz`

Every row contains only features computable at its recorded cutoff.

`--feature-profile legacy` reproduces the original 606 features. The `extended`
profile adds robust baseline/history/window summaries and explicit contrasts
between 5-, 20-, and 50-cycle trends. Both profiles retain the future-row
causality test, and the selected profile is written to
`feature_generation_config.json`.
