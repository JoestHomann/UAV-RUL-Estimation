# PE_20: nested forecast-history confirmation

This is the complete outer refit required after PE_15. For each held UAV group,
the six Run 7 tree members, their blend weight, and both residual heads are fit
without those UAVs. Forecasts at lags 2, 5, 10, and 20 are rebuilt causally from
the same fitted members. The four inner refits also produce leakage-safe
training-side predictions for PE_22 blend selection.

The run contains 25 resumable model cells: five outer evaluations and four inner
evaluations inside each outer fold. It promotes prediction history only at a 2%
mean-RMSE gain and four of five fold wins.
