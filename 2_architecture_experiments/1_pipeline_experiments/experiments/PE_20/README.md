# PE_20: nested forecast-history confirmation

This is the complete outer refit required after PE_15. For each held UAV group,
the six Run 7 tree members, their blend weight, and both residual heads are fit
without those UAVs. Forecasts at lags 2, 5, 10, and 20 are rebuilt causally from
the same fitted members. The four inner refits also produce leakage-safe
training-side predictions for PE_22 blend selection.

The run contains 25 resumable model cells: five outer evaluations and four inner
evaluations inside each outer fold. It promotes prediction history only at a 2%
mean-RMSE gain and four of five fold wins.

The completed confirmation did not promote prediction history. Mean RMSE rose
from 10.2361 to 10.3962 (a 1.56% regression), pooled R2 fell from 0.90446 to
0.90149, and prediction history won two of five folds. The UAV-bootstrap 95%
interval for challenger-minus-control RMSE was 0.016 to 0.310 cycles, so the
PE_15 screening gain did not survive the complete nested refit.
