# To Do's

## Questions to think about

- Engineering-wise a slightly negative bias would be ideal. This would make the models RUL estimation conservative. Can we force the model to be slightly conservative?
- Data Augmentation maybe? Risky but could improve performance
- Check notes from phase 0 again, and experiment with different dataset /feature sets (certain channels or features turned off or defined differently)
- Research if there are other models we could try out
- Implement regularization for tcn maybe? Or even other models?
- Median Polish for flight cycle vs each telemetry channels?
- ICA?
- Bagging maybe applicable? -> Estimate feature importance??
- Different feature scaling instead of robust scaling?
- GANs for increasing the dataset size?
- RNN architecture? Better Transformer architecture? For architecture study
- Increase or decrease capacity of the models? Reduce under- or overfitting?
- Why is the banana curved, not straight? LG Joest


-> Rerun pipeline experiments with temporal model e.g. LSTM / TCN as XGBoost and ExtraTrees directly use bootstrapping of the feature set to learn from all feature but then average out the learning, which limits performance if we actually know the best feature set. Those models might not sure the actual improve through choosing a different feature set because of this averaging