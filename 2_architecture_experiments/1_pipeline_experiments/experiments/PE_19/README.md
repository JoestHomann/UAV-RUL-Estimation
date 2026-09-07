# PE_19: marginal ensemble utility

This executes the small blend test that PE_7's standalone-accuracy gate previously blocked. The selected LSTM and TCN configuration for each outer fold is refitted on that fold's 80 training UAVs with its fixed retraining epoch count. Challenger weights are selected from inner OOF predictions belonging only to those 80 training UAVs and applied once to the 20 held UAVs; zero is always eligible. The workflow checkpoints temporal predictions and does not retune the models.

The TCN blend reached mean-fold R² 0.9025 and pooled R² 0.9062, but improved
mean RMSE by only 0.76% with 3/5 fold wins. The LSTM blend regressed by 0.83%.
Neither passed the 1% and 4/5 promotion gates, so the tree control was retained.
