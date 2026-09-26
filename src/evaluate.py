"""Log-price metrics and evaluation figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .utils import root_path


def rmse_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()) ** 2)))


def predict_loader(model: torch.nn.Module, loader, device: torch.device) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for batch in loader:
            parts.append(model(batch[0].to(device)).cpu().numpy())
    return np.concatenate(parts).ravel()


def save_figures(history: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                 prefix: str, out_dir: str | Path = "outputs/figures") -> None:
    folder = root_path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for columns, suffix, ylabel in [
        (("train_rmse", "val_rmse"), "rmse", "RMSE (log-price)"),
        (("train_loss", "val_loss"), "loss", "MSE loss (log-price)"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        history.plot(x="epoch", y=list(columns), ax=ax)
        ax.set_ylabel(ylabel); ax.grid(alpha=0.2)
        fig.tight_layout(); fig.savefig(folder / f"{prefix}_{suffix}.png", dpi=150); plt.close(fig)
    actual, predicted = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual, predicted, s=16, alpha=0.6)
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    ax.plot([low, high], [low, high], color="red", linestyle="--")
    ax.set(xlabel="Actual log-price", ylabel="Predicted log-price")
    fig.tight_layout(); fig.savefig(folder / f"{prefix}_actual_predicted.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(predicted, actual - predicted, s=16, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--")
    ax.set(xlabel="Predicted log-price", ylabel="Residual (actual - predicted)")
    fig.tight_layout(); fig.savefig(folder / f"{prefix}_residuals.png", dpi=150); plt.close(fig)
