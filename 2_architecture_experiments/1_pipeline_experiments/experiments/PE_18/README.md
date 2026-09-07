# PE_18: pretrained tabular prior

This grouped outer-fold experiment evaluates CatBoost and TabPFN-3 on the same cap-125 training target and raw RUL metric as Run 7. Compact feature selection is fitted separately inside every training fold. Each small 0–25% challenger weight is selected from inner OOF predictions belonging to the outer training UAVs and applied once to the held outer fold; direct models require a 2% RMSE gain while blends require 1%.

TabPFN is an optional heavyweight dependency for the repository but required by this experiment. Install `tabpfn==8.5.0` and place `tabpfn-v3-regressor-v3_default.ckpt` in this experiment's `checkpoints` directory before running PE_18. The runner checks both before any model training. No competition data is sent to a remote API.

The completed experiment promoted `control_plus_tabpfn_full`. It reached
mean-fold R² 0.9026 and pooled R² 0.9067, reducing mean RMSE from 10.2361 to
10.1035 (1.30%) with four of five fold wins. The selected TabPFN weights ranged
from 5% to 25%. The paired UAV-bootstrap interval crossed zero, so the result
passes the declared practical gate but still needs confirmation on new grouped
split seeds. Standalone TabPFN and both compact-feature models were substantially
weaker than the control.
