# Temporal architecture Run 7 conclusion

Run 7 failed its development promotion gate and was stopped for mathematical
futility after 19 of 20 studies completed. This was not a pipeline crash.

- GRU completed five folds with mean selected inner RMSE 19.56.
- LSTM completed four folds with mean selected inner RMSE 19.26.
- TCN completed five folds with mean selected inner RMSE 20.85.
- Multiscale CNN completed five folds with mean selected inner RMSE 22.51.
- The required mean RMSE was at most 10.7.
- Giving unfinished `lstm__outer_04` an impossible RMSE of zero would still
  leave LSTM at mean RMSE 15.41.

Therefore no family qualifies for three-seed confirmation and no
`temporal_winner_manifest.json` should be created. PE_7 is blocked by design.

The leading interpretation is a representation bottleneck, not proof that
temporal information is useless: Run 7 used short recent raw windows, whereas
the tree baseline has engineered full-prefix summaries. PE_10 and architecture
Run 8 test that hypothesis with hybrid and multi-resolution inputs.
