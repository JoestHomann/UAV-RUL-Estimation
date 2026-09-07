# PE_18: pretrained tabular prior

This grouped outer-fold experiment evaluates CatBoost and TabPFN-3 on the same cap-125 training target and raw RUL metric as Run 7. Compact feature selection is fitted separately inside every training fold. Each small 0–25% challenger weight is selected from inner OOF predictions belonging to the outer training UAVs and applied once to the held outer fold; direct models require a 2% RMSE gain while blends require 1%.

TabPFN is an optional heavyweight dependency for the repository but required by this experiment. Install `tabpfn==8.5.0` and place `tabpfn-v3-regressor-v3_default.ckpt` in this experiment's `checkpoints` directory before running PE_18. The runner checks both before any model training. No competition data is sent to a remote API.
