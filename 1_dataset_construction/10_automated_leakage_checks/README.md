# Step 10: Automated leakage checks

**Reads:** artifacts from Steps 1–7 and 9, including Step 1 `test_fligh_cycles_cut_offs.csv`

**Writes:** `artifacts/verification_report.json`

This final report verifies grouping, cutoffs, feature causality, saved preprocessing parameters, and baseline predictions.

For versioned profiles, it also discovers every cataloged feature set, checks
the configured feature profile during the causality test, and confirms that
variable prefix counts still give every UAV total sample weight one.
