# PE_22: confirmed history and TabPFN combination

This inexpensive saved-prediction experiment is gated by successful PE_20,
PE_18, and PE_21 winner manifests. Within every original outer fold it chooses
the TabPFN weight using only PE_20 and PE_18 inner predictions, then evaluates
the combination once on held outer UAVs. If the combination passes its 1% and
four-of-five gate, it writes the frozen candidate contract consumed by PE_23.

The confirmation chain did not run PE_22 because neither PE_20 nor PE_21 passed
its frozen promotion gate. This is an expected gated stop, not a failed model
fit.
