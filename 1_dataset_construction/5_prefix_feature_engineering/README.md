# Step 5: Prefix feature engineering

**Reads:** raw train/test data, Step 3 scenarios/endpoints, Step 4 training prefixes

**Writes:** `artifacts/training_features.csv.gz`, `artifacts/development_validation_features.csv.gz`, `artifacts/locked_validation_features.csv.gz`, `artifacts/test_features.csv.gz`

Every row contains only features computable at its recorded cutoff.

`--feature-profile legacy` reproduces the original 606 features. The `extended`
profile adds robust baseline/history/window summaries, explicit contrasts
between 5-, 20-, and 50-cycle trends, and direction-normalized degradation
onset/change-point features for the ten screened degradation channels. It also
adds robust-scaled history slopes and robust-scaled window slopes, deltas, and
last-minus-mean values. These use each prefix's early-life median/MAD scale and
support matched raw-versus-normalized feature experiments. Onsets
require three consecutive observations at least two robust baseline scales in
the degradation direction. Change points are positive prefix-local mean shifts
with at least five observations on each side. Both profiles retain the future-row
causality test, and the selected profile is written to
`feature_generation_config.json`.
