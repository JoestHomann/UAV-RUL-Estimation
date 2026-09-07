# PE_21: multi-seed TabPFN blend confirmation

This repeats the frozen PE_18 full-feature TabPFN and Run 7 comparison on split
seeds 20260917 and 20260927. Each of the ten outer evaluations chooses its
0–25% TabPFN weight from four fully refitted inner UAV folds. No compact feature
view or new hyperparameter search is included.

The run checkpoints each control and TabPFN fit separately. It requires at least
eight of ten fold wins and a 1% mean-RMSE improvement for the blend.
