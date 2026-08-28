# Step 6: Feature sets

**Reads:** Step 5 training-feature columns

**Writes:** `artifacts/feature_catalog.csv`

The catalog identifies membership in explicitly named feature sets loaded from
`phase_1_settings.toml`. The legacy profile retains the original four sets. The
extended profile adds control, robust, acceleration, compact, drift-ablation,
and complete-superset alternatives. `screened_drift_pruned` removes the twelve
drift-heavy telemetry 15/16 features identified by `FE_run_1`.
`screened_drift_replaced` adds robust global and local-window alternatives for
those channels to the pruned set. This step declares candidates and never
selects a winner.
