# PE_23: final confirmed candidate fit and submission

PE_23 runs only after PE_22 promotes the combined history and TabPFN model. It
fits the history-aware Run 7 system on all 100 training UAVs, fits TabPFN on the
same full feature table, applies the OOF-selected frozen blend weight, and writes
`runs/run_1/reporting/submission.csv` with exactly `id,RUL` columns.

The history model and both component prediction files are checkpointed. The
manifest records package/checkpoint hashes, verifies finite nonnegative values
and the exact test ID set, and explicitly records that no test labels or test
metrics were used.
