# PE_16: residual refinement

This performs complete outer-UAV refits for eight residual-head contracts. The declared screen includes zero and half-strength correction, ridge and histogram-gradient heads, stronger regularization at fixed coverage, inverse-duplicate weighting for the existing five-scenario table, and twenty distinct terminal-support endpoints per UAV. The runner deterministically generates the twenty cutoffs inside `1 <= RUL <= 125`, then uses only the active fold's training UAVs for calibration; evaluated development UAVs remain excluded. Each candidate/fold is checkpointed.

The regularized five-scenario histogram-gradient head was best, improving mean
RMSE by 0.63% with 4/5 fold wins. It missed the 1% gate and its bootstrap
interval crossed zero, so the original residual head was retained.
