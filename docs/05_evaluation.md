# Evaluation

The main metric is RMSE between the actual and predicted `log1p(SalePrice)` on the fixed 292-row validation holdout. Every epoch's MSE and RMSE covers the entire training and validation folds in `eval()` and `no_grad()` mode. Early stopping selects the lowest validation RMSE, and the saved checkpoint is reloaded before final evaluation.

| Experiment | Hidden widths | Dropout | Learning rate | Weight decay | Best epoch | Validation RMSE |
|---|---|---|---:|---:|---:|---:|
| E0 | 64-32 | 0 | 0.001 | 0 | 3 | **0.133316** |
| E1 | 128-64-32 | 0.15, 0.10 | 0.001 | 0.0001 | 10 | 0.141518 |
| E2 | 256-128-64 | 0.20, 0.15 | 0.001 | 0.0001 | 4 | 0.136446 |
| E3 | 128-64-32 | 0.15, 0.10 | 0.0005 | 0.0001 | 11 | 0.142573 |
| E4 | 128-64-32 | 0.25, 0.15 | 0.001 | 0.001 | 24 | 0.138196 |

E0 wins on validation RMSE. Training loss alone is not used for selection. The 0.133316 value is a local holdout estimate, not a Kaggle score; choosing a configuration on this one holdout can optimistically bias it. Outlier candidates remain in the data. No cross-validation, feature engineering or scheduler was added in the baseline.

The user later reported a Kaggle score of **0.13543** for the submitted CSV. This result was not used to select or retrain the model.

`outputs/metrics/experiments.csv` is the machine-readable comparison; `e0_history.csv` through `e4_history.csv` record epoch metrics. Each experiment has four figures in `outputs/figures/`: train/validation RMSE, train/validation loss, actual versus predicted log-price, and residuals. `notebooks/03_mlp_experiments.ipynb` displays the real comparison and selected model's plots.
