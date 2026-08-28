# Step 6: Feature sets

**Reads:** Step 5 training-feature columns

**Writes:** `artifacts/feature_catalog.csv`

The catalog identifies membership in explicitly named feature sets loaded from
`phase_1_settings.toml`. The legacy profile retains the original four sets; the
Run 5 profile adds control, robust, acceleration, compact, and complete-superset
alternatives. This step declares candidates and never selects a winner.
