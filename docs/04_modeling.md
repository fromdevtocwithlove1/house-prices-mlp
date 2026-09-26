# Modeling

The MLP predicts `log1p(SalePrice)` directly. E1 uses widths 128 → 64 → 32, BatchNorm and ReLU on the first two hidden layers, dropout 0.15 and 0.10, ReLU on the last hidden layer, and a linear scalar output. `src/model.py` accepts hidden widths and dropout from YAML so E0–E4 share one implementation.

Training uses MSE loss and AdamW, batch size 32, seed 42, a maximum of 500 epochs, and early stopping after 30 epochs without validation RMSE improvement. Each epoch records full-fold MSE and RMSE in evaluation mode. The best checkpoint is reloaded and its validation RMSE verified before use. Each experiment stores its own checkpoint and fitted preprocessor.

`device=auto` performs a CUDA forward/backward probe; it falls back to CPU when CUDA is unavailable or the probe fails. The project currently has a CPU-only PyTorch wheel, so CPU is expected. All experiments use the same fixed holdout and preprocessed training fold.

The output bias is initialized to the training fold's mean log price (about 12), because zero initialization made the baseline converge poorly. This uses training data only and leaves the planned architecture unchanged. The verified E1 baseline reached validation RMSE **0.141518** at epoch 10, with early stopping after epoch 40. Its reloaded best checkpoint reproduces the recorded RMSE.
