# PE_24: regime-aware TabPFN blend confirmation

PE_24 tests whether the small global TabPFN gain from PE_18 and PE_21 can be
concentrated in regimes where TabPFN helps. It uses three fresh grouped split
seeds. Every outer fold refits Run 7 and TabPFN, and its gate is fitted only on
the four corresponding inner OOF partitions.

The fixed depth-two regression tree uses the Run 7 prediction, ensemble standard
deviation, ensemble range, tree-family disagreement, and TabPFN-minus-control
gap. It estimates a row-specific TabPFN weight constrained to 0–50%. The gate
target and sample weighting make each leaf minimize blend squared error while
giving each UAV equal total weight. No true-RUL band is used at inference.

Only `regime_tabpfn_blend` is eligible for promotion. It must win at least 12 of
15 folds, improve mean RMSE by at least 2%, achieve pooled R2 of at least 0.90,
and have a UAV-bootstrap 95% RMSE-delta interval wholly below zero. Standalone
TabPFN and the frozen PE_21 global-weight grid are diagnostic anchors.

The run has 75 resumable evaluation jobs and two fitted methods per job. Re-run
the same command after an interruption to continue from `fold_predictions.csv`.
The first execution writes `pre_registration.json`; changing a scientific
setting afterward requires a new `pipeline.run` directory.

Run 1 completed all 150 component fits without promotion. The regime blend
reduced mean RMSE from 10.3106 to 10.1151 (1.896%) and reached pooled R2 0.90660,
but won 11/15 folds and its UAV-bootstrap 95% RMSE-change interval was
[-0.5137, +0.0862]. Run 7 remains the control. PE_25 screens one restricted
TabPFN rule using these saved predictions; PE_26 independently tests an
early-history specialist.
