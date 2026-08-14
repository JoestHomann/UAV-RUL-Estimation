# Step 7: Fold-fitted preprocessing

**Reads:** Step 2 outer folds, Step 5 training features, Step 6 feature catalog

**Writes:** `artifacts/fold_scaler_parameters.csv.gz`, `artifacts/preprocessing_config.json`

Centers and scales are fitted separately on each outer-training partition. The
parameter table records `scale_method`, `iqr`, and `standard_deviation` so use of
the standard-deviation or unit-scale fallback remains auditable. It also records
the observed data range and feature-relative tolerance used to distinguish real
variation from floating-point noise.
