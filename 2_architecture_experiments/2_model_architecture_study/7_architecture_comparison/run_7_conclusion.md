# Temporal architecture Run 7 conclusion

Run 7 completed all 20 development studies and 200 candidates normally, but
failed its development promotion gate. This was not a pipeline crash.

- LSTM was best at mean OOF R2 0.6531 and RMSE 19.57.
- GRU reached mean OOF R2 0.6424 and RMSE 19.83.
- TCN reached mean OOF R2 0.5969 and RMSE 21.08.
- Multiscale CNN reached mean OOF R2 0.5325 and RMSE 22.71.
- The required gates were mean R2 >= 0.89 and mean RMSE <= 10.7.
- Residual correlations with the tree control were 0.46-0.51, demonstrating
  complementarity but not enough standalone accuracy to make stacking viable.

Therefore no family qualifies for three-seed confirmation. The generated
`temporal_winner_manifest.json` correctly records `no_promotion` with no
winner, and PE_7 is blocked by design.

The leading interpretation is a representation bottleneck, not proof that
temporal information is useless: Run 7 used short recent raw windows, whereas
the tree baseline has engineered full-prefix summaries. PE_10 and architecture
Run 8 test that hypothesis with hybrid and multi-resolution inputs.
