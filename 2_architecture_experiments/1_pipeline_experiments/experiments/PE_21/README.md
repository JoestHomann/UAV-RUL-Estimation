# PE_21: multi-seed TabPFN blend confirmation

This repeats the frozen PE_18 full-feature TabPFN and Run 7 comparison on split
seeds 20260917 and 20260927. Each of the ten outer evaluations chooses its
0–25% TabPFN weight from four fully refitted inner UAV folds. No compact feature
view or new hyperparameter search is included.

The run checkpoints each control and TabPFN fit separately. It requires at least
eight of ten fold wins and a 1% mean-RMSE improvement for the blend.

The completed confirmation retained the control. The blend reduced mean RMSE
from 10.7848 to 10.6640 (1.12%) and raised pooled R2 from 0.89459 to 0.89685,
but won only six of ten folds rather than the required eight. Its UAV-bootstrap
95% interval for challenger-minus-control RMSE was -0.335 to 0.102 cycles. The
small PE_18 blend effect therefore repeated in magnitude but remains too
uncertain for promotion. Standalone TabPFN was substantially worse, with pooled
R2 0.84462.
