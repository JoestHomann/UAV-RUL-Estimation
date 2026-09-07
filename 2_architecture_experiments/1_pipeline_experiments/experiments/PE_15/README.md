# PE_15: causal filtering

This development-only screen compares the existing PE_11 correction with one-sided sensor filters and forecast-history innovations. Earlier near-cap forecasts are flagged and excluded from failure-cycle translation. Every correction is fitted on disjoint inner-fold UAVs.

Run `run.py --list` to inspect the command, then `run.py` to execute it.

The completed screen selected `prediction_history`: mean RMSE improved from 11.0243 to 9.5916 (13.0%) with five of five fold wins. The UAV-bootstrap paired RMSE-delta interval is -1.94 to -0.95 cycles. This remains a screening candidate until a complete outer refit confirms it.
